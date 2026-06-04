"""Daily quota for subsidized paid-tier model sessions.

Tracks per-user paid-tier model session starts against a daily cap derived from
the user's HF plan. MongoDB is the source of truth when configured; the
in-process dict remains the fallback for local/dev/test runs.

Unit: first paid-tier submit per session per UTC day, not raw messages. A
user who sends with a paid-tier model in any session consumes one quota
point for that day; continuing the same session on the same day doesn't
(`AgentSession.paid_counted_day` guards that). Model-level tier handling lives
in ``backend.routes.agent``; this module only tracks the per-plan daily cap.

Cap tiers:
  free user   → 0 included paid-tier sessions
  pro user    → PRO_DAILY_SESSIONS
"""

import asyncio
import os
from datetime import UTC, datetime

from agent.core.session_persistence import (
    NoopSessionStore,
    get_session_store,
    _reset_store_for_tests,
)

PRO_DAILY_SESSIONS: int = int(os.environ.get("PRO_DAILY_SESSIONS", "5"))

# user_id -> (day_utc_iso, count_for_that_day)
_paid_counts: dict[str, tuple[str, int]] = {}
_lock = asyncio.Lock()


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def current_quota_day() -> str:
    """Return the UTC date key used for today's paid-tier quota bucket."""
    return _today()


def daily_cap_for(plan: str | None) -> int:
    """Return the daily included paid-tier session cap for the given plan."""
    return PRO_DAILY_SESSIONS if plan == "pro" else 0


async def get_paid_used_today(user_id: str) -> int:
    """Return today's paid-tier session count for the user."""
    store = get_session_store()
    if getattr(store, "enabled", False):
        db_count = await store.get_quota(user_id, _today())
        return db_count or 0

    async with _lock:
        entry = _paid_counts.get(user_id)
        if entry is None:
            return 0
        day, count = entry
        if day != _today():
            # Stale day — drop the entry so the first increment starts fresh.
            _paid_counts.pop(user_id, None)
            return 0
        return count


async def increment_paid(user_id: str) -> int:
    """Bump today's paid-tier session count for the user. Returns the new value."""
    store = get_session_store()
    if getattr(store, "enabled", False):
        db_count = await store.try_increment_quota(user_id, _today(), cap=10**9)
        return db_count or 0

    async with _lock:
        today = _today()
        day, count = _paid_counts.get(user_id, (today, 0))
        if day != today:
            count = 0
        count += 1
        _paid_counts[user_id] = (today, count)
        return count


async def try_increment_paid(user_id: str, cap: int) -> int | None:
    """Atomically bump today's count if below *cap*.

    Returns the new count, or None when the user is already at the cap.
    """
    store = get_session_store()
    if getattr(store, "enabled", False):
        return await store.try_increment_quota(user_id, _today(), cap)

    async with _lock:
        today = _today()
        day, count = _paid_counts.get(user_id, (today, 0))
        if day != today:
            count = 0
        if count >= cap:
            return None
        count += 1
        _paid_counts[user_id] = (today, count)
        return count


async def refund_paid(user_id: str) -> None:
    """Decrement today's count — used when session creation fails after a successful gate."""
    store = get_session_store()
    if getattr(store, "enabled", False):
        await store.refund_quota(user_id, _today())
        return

    async with _lock:
        entry = _paid_counts.get(user_id)
        if entry is None:
            return
        day, count = entry
        if day != _today():
            _paid_counts.pop(user_id, None)
            return
        new_count = max(0, count - 1)
        if new_count == 0:
            _paid_counts.pop(user_id, None)
        else:
            _paid_counts[user_id] = (day, new_count)


def _reset_for_tests() -> None:
    """Test-only: clear the in-memory store."""
    _paid_counts.clear()
    _reset_store_for_tests(NoopSessionStore())
