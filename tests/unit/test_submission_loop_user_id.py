import asyncio

from agent.core.agent_loop import submission_loop


class _FakeConfig:
    save_sessions = False
    session_dataset_repo = "fake/repo"


class _FakeToolRouter:
    tools = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


def test_submission_loop_passes_user_id(monkeypatch):
    captured = {}

    class _FakeSession:
        def __init__(self, event_queue, **kwargs):
            captured["event_queue"] = event_queue
            captured["kwargs"] = kwargs
            captured["session"] = self
            self.config = kwargs["config"]
            self.is_running = True

        async def send_event(self, event):
            captured["ready_event"] = event

    async def _fake_process_submission(session, submission):
        captured["submission"] = submission
        return False

    async def body():
        submission_queue = asyncio.Queue()
        event_queue = asyncio.Queue()
        session_holder = [None]
        fake_submission = object()
        await submission_queue.put(fake_submission)

        monkeypatch.setattr("agent.core.agent_loop.Session", _FakeSession)
        monkeypatch.setattr(
            "agent.core.agent_loop.process_submission", _fake_process_submission
        )

        await submission_loop(
            submission_queue,
            event_queue,
            config=_FakeConfig(),
            tool_router=_FakeToolRouter(),
            session_holder=session_holder,
            hf_token="hf_token",
            user_id="alice",
            local_mode=True,
            stream=False,
        )

        assert session_holder[0] is captured["session"]
        assert captured["submission"] is fake_submission

    asyncio.run(body())

    assert captured["event_queue"] is not None
    assert captured["kwargs"]["hf_token"] == "hf_token"
    assert captured["kwargs"]["user_id"] == "alice"
    assert captured["kwargs"]["local_mode"] is True
    assert captured["kwargs"]["stream"] is False
