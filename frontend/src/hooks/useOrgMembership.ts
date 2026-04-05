/**
 * Polls backend for org membership status.
 * When membership is detected, updates the user in the agent store
 * and closes any org-join popup that was opened.
 */
import { useEffect, useRef } from 'react';
import { useAgentStore } from '@/store/agentStore';

const POLL_INTERVAL_MS = 3000;

/**
 * @param enabled  Only poll when true (user is authenticated but not yet confirmed as org member)
 * @returns popupRef — assign `window.open()` result to `.current` so the hook can auto-close it
 */
export function useOrgMembership(enabled: boolean) {
  const user = useAgentStore((s) => s.user);
  const setUser = useAgentStore((s) => s.setUser);
  const popupRef = useRef<Window | null>(null);

  useEffect(() => {
    if (!enabled || user?.orgMember) return;

    let cancelled = false;

    const check = async () => {
      try {
        const res = await fetch('/auth/org-membership', { credentials: 'include' });
        if (!res.ok || cancelled) return;
        const data = await res.json();
        if (cancelled) return;
        if (data.is_member && user) {
          setUser({ ...user, orgMember: true });
          try { popupRef.current?.close(); } catch { /* cross-origin or already closed */ }
          popupRef.current = null;
        }
      } catch { /* backend unreachable — skip */ }
    };

    check();
    const id = setInterval(check, POLL_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [enabled, user?.orgMember, user, setUser]);

  return popupRef;
}
