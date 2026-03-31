#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.20.0", "httpx>=0.27.0"]
# ///
"""
Sandbox Tools — Agent-native primitives for HF Space dev-mode sandboxes.

Architecture:
  - Creates a sandbox by duplicating a template Space (runs sandbox_server.py)
  - Waits for it to come online
  - Communicates via HTTPS to the Space's API
  - Optionally deletes the Space when done

Lifecycle:
    sb = Sandbox.create(owner="burtenshaw")         # duplicate, wait, connect
    sb = Sandbox.create(owner="burtenshaw",          # with options
                        hardware="t4-small",
                        private=True,
                        sleep_time=3600)
    sb = Sandbox.connect("burtenshaw/my-sandbox-abc") # attach to existing

    sb.bash("uv run train.py")
    sb.read("/app/train.py")
    sb.edit("/app/train.py", old_str="lr=1e-3", new_str="lr=1e-4")

    sb.delete()                                       # tear down when done

    # Or use as a context manager for automatic cleanup
    with Sandbox.create(owner="burtenshaw") as sb:
        sb.bash("python train.py")
    # Space deleted on exit

Tools: bash, read, write, edit, upload
"""

from __future__ import annotations

import io
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx
from huggingface_hub import CommitOperationAdd, HfApi

TEMPLATE_SPACE = "burtenshaw/sandbox"
HARDWARE_OPTIONS = [
    "cpu-basic",
    "cpu-upgrade",
    "t4-small",
    "t4-medium",
    "a10g-small",
    "a10g-large",
    "a100-large",
]
OUTPUT_LIMIT = 30000
LINE_LIMIT = 2000
DEFAULT_READ_LIMIT = 2000
DEFAULT_TIMEOUT = 240
MAX_TIMEOUT = 1200
WAIT_TIMEOUT = 600
WAIT_INTERVAL = 5
API_WAIT_TIMEOUT = 180

_DOCKERFILE = """\
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && \\
    apt-get install -y \\
      bash git git-lfs wget curl procps \\
      htop vim nano jq tmux \\
      build-essential && \\
    rm -rf /var/lib/apt/lists/*

RUN uv pip install --system fastapi uvicorn python-multipart

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \\
    PATH=/home/user/.local/bin:$PATH \\
    PIP_USER=1 \\
    HF_HUB_DISABLE_PROGRESS_BARS=1 \\
    TQDM_DISABLE=1 \\
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app
COPY --chown=user . /app

EXPOSE 7860

CMD ["python", "sandbox_server.py"]
"""

_SANDBOX_SERVER = '''\
"""Minimal FastAPI server for sandbox operations."""
import os, subprocess, pathlib, signal, threading
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

# Track active bash processes so they can be killed on cancel
_active_procs = {}  # pid -> subprocess.Popen
_proc_lock = threading.Lock()

class BashReq(BaseModel):
    command: str
    work_dir: str = "/app"
    timeout: int = 120

class ReadReq(BaseModel):
    path: str
    offset: Optional[int] = None
    limit: Optional[int] = 2000

class WriteReq(BaseModel):
    path: str
    content: str

class EditReq(BaseModel):
    path: str
    old_str: str
    new_str: str
    replace_all: bool = False

class ExistsReq(BaseModel):
    path: str

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/bash")
def bash(req: BashReq):
    try:
        proc = subprocess.Popen(
            req.command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=req.work_dir, start_new_session=True,
        )
        with _proc_lock:
            _active_procs[proc.pid] = proc
        try:
            stdout, stderr = proc.communicate(timeout=req.timeout)
            output = stdout + stderr
            if len(output) > 30000:
                output = output[:30000] + "\\n... (truncated)"
            return {"success": proc.returncode == 0, "output": output, "error": "" if proc.returncode == 0 else f"Exit code {proc.returncode}"}
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return {"success": False, "output": "", "error": f"Timeout after {req.timeout}s"}
        finally:
            with _proc_lock:
                _active_procs.pop(proc.pid, None)
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/kill")
def kill_all():
    """Kill all active bash processes. Called when user cancels."""
    with _proc_lock:
        pids = list(_active_procs.keys())
    killed = []
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed.append(pid)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
                killed.append(pid)
            except OSError:
                pass
    return {"success": True, "output": f"Killed {len(killed)} process(es): {killed}", "error": ""}

@app.post("/api/read")
def read(req: ReadReq):
    try:
        p = pathlib.Path(req.path)
        if not p.exists():
            return {"success": False, "output": "", "error": f"File not found: {req.path}"}
        if p.is_dir():
            return {"success": False, "output": "", "error": f"Is a directory: {req.path}"}
        lines = p.read_text().splitlines()
        start = (req.offset or 1) - 1
        end = start + (req.limit or len(lines))
        selected = lines[start:end]
        numbered = "\\n".join(f"{start + i + 1}\\t{line}" for i, line in enumerate(selected))
        return {"success": True, "output": numbered, "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/write")
def write(req: WriteReq):
    try:
        p = pathlib.Path(req.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(req.content)
        return {"success": True, "output": f"Wrote {len(req.content)} bytes to {req.path}", "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/edit")
def edit(req: EditReq):
    try:
        p = pathlib.Path(req.path)
        if not p.exists():
            return {"success": False, "output": "", "error": f"File not found: {req.path}"}
        content = p.read_text()
        if req.old_str not in content:
            return {"success": False, "output": "", "error": f"old_str not found in {req.path}"}
        if not req.replace_all and content.count(req.old_str) > 1:
            return {"success": False, "output": "", "error": f"old_str appears {content.count(req.old_str)} times. Use replace_all=true or provide more context."}
        if req.replace_all:
            new_content = content.replace(req.old_str, req.new_str)
        else:
            new_content = content.replace(req.old_str, req.new_str, 1)
        p.write_text(new_content)
        return {"success": True, "output": f"Edited {req.path}", "error": ""}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

@app.post("/api/exists")
def exists(req: ExistsReq):
    return {"success": True, "output": str(pathlib.Path(req.path).exists()).lower(), "error": ""}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
'''


@dataclass
class ToolResult:
    success: bool
    output: str = ""
    error: str = ""

    def __str__(self):
        if self.success:
            return self.output or "(no output)"
        return f"ERROR: {self.error}"

    def to_dict(self) -> dict:
        return {"success": self.success, "output": self.output, "error": self.error}


@dataclass
class Sandbox:
    """
    A handle to an HF Space sandbox.

    Use Sandbox.create() to spin up a new one, or Sandbox.connect() to
    attach to an existing running Space.
    """

    space_id: str
    token: str | None = None
    work_dir: str = "/app"
    timeout: int = DEFAULT_TIMEOUT
    _owns_space: bool = field(default=False, repr=False)
    _base_url: str = field(init=False, repr=False)
    _client: httpx.Client = field(init=False, repr=False)
    _hf_api: HfApi = field(init=False, repr=False)
    _files_read: set = field(init=False, repr=False, default_factory=set)

    def __post_init__(self):
        slug = self.space_id.replace("/", "-")
        # Trailing slash is critical: httpx resolves relative paths against base_url.
        # Without it, client.get("health") resolves to /health instead of /api/health.
        self._base_url = f"https://{slug}.hf.space/api/"
        self._client = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self.token}"} if self.token else {},
            timeout=httpx.Timeout(MAX_TIMEOUT, connect=30),
            follow_redirects=True,
        )
        self._hf_api = HfApi(token=self.token)

    # ── Lifecycle ─────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        owner: str,
        *,
        name: str | None = None,
        template: str = TEMPLATE_SPACE,
        hardware: str = "cpu-basic",
        private: bool = False,
        sleep_time: int | None = None,
        token: str | None = None,
        wait_timeout: int = WAIT_TIMEOUT,
        log: "Callable[[str], object] | None" = None,
    ) -> Sandbox:
        """
        Create a new sandbox by duplicating the template Space.

        Generates a unique space name, duplicates the template, waits for it
        to come online, then returns a connected Sandbox.

        Args:
            owner: HF username or org (e.g. "burtenshaw").
            name: Base name for the space. Defaults to "sandbox".
                  A unique suffix is always appended.
            template: Source Space to duplicate (default: burtenshaw/sandbox).
            hardware: Hardware tier (cpu-basic, t4-small, etc.).
            private: Whether the Space should be private.
            sleep_time: Auto-sleep after N seconds of inactivity.
            token: HF API token (from user's OAuth session).
            wait_timeout: Max seconds to wait for Space to start (default: 300).

        Returns:
            A Sandbox instance connected to the running Space.
        """
        _log = log or print
        api = HfApi(token=token)

        base = name or "sandbox"
        suffix = uuid.uuid4().hex[:8]
        space_id = f"{owner}/{base}-{suffix}"

        _log(f"Creating sandbox: {space_id} (from {template})...")

        kwargs = {
            "from_id": template,
            "to_id": space_id,
            "private": private,
            "hardware": hardware,
        }
        if sleep_time is not None:
            kwargs["sleep_time"] = sleep_time

        api.duplicate_space(**kwargs)
        _log(f"Space created: https://huggingface.co/spaces/{space_id}")

        # Upload sandbox server and Dockerfile (triggers rebuild)
        cls._setup_server(space_id, api, log=_log)

        # Wait for it to come online (rebuild + start)
        _log(f"Waiting for Space to start (timeout: {wait_timeout}s)...")
        deadline = time.time() + wait_timeout
        while time.time() < deadline:
            runtime = api.get_space_runtime(space_id)
            if runtime.stage == "RUNNING":
                _log(f"Space is running (hardware: {runtime.hardware})")
                break
            if runtime.stage in ("RUNTIME_ERROR", "BUILD_ERROR"):
                raise RuntimeError(
                    f"Space failed to start: {runtime.stage}. "
                    f"Check https://huggingface.co/spaces/{space_id}"
                )
            _log(f"  {runtime.stage}...")
            time.sleep(WAIT_INTERVAL)
        else:
            raise TimeoutError(
                f"Space did not start within {wait_timeout}s. "
                f"Check https://huggingface.co/spaces/{space_id}"
            )

        # Wait for the API server to be responsive (non-fatal)
        sb = cls(space_id=space_id, token=token, _owns_space=True)
        try:
            sb._wait_for_api(timeout=API_WAIT_TIMEOUT, log=_log)
        except TimeoutError as e:
            _log(
                f"Warning: API health check timed out ({e}), but Space is RUNNING. Continuing."
            )
        return sb

    @staticmethod
    def _setup_server(space_id: str, api: HfApi, *, log: Callable[[str], object] = print) -> None:
        """Upload embedded sandbox server + Dockerfile to the Space (single commit)."""
        log(f"Uploading sandbox server to {space_id}...")
        api.create_commit(
            repo_id=space_id,
            repo_type="space",
            operations=[
                CommitOperationAdd(
                    path_in_repo="sandbox_server.py",
                    path_or_fileobj=io.BytesIO(_SANDBOX_SERVER.encode()),
                ),
                CommitOperationAdd(
                    path_in_repo="Dockerfile",
                    path_or_fileobj=io.BytesIO(_DOCKERFILE.encode()),
                ),
            ],
            commit_message="Setup sandbox server",
        )
        log("Server files uploaded, rebuild triggered.")

    @classmethod
    def connect(cls, space_id: str, *, token: str | None = None) -> Sandbox:
        """
        Connect to an existing running Space.

        Does a health check to verify the Space is reachable.
        """
        sb = cls(space_id=space_id, token=token, _owns_space=False)
        sb._wait_for_api(timeout=60)
        return sb

    def _wait_for_api(self, timeout: int = API_WAIT_TIMEOUT, log: Callable[[str], object] = print):
        """Poll the health endpoint until the server responds."""
        deadline = time.time() + timeout
        last_err = None
        last_status = None
        while time.time() < deadline:
            try:
                resp = self._client.get("health", timeout=10)
                last_status = resp.status_code
                if resp.status_code == 200:
                    log(f"API is responsive at {self._base_url}")
                    return
            except Exception as e:
                last_err = e
            time.sleep(3)
        raise TimeoutError(
            f"Sandbox API at {self._base_url} not responding after {timeout}s. "
            f"Last status: {last_status}, last error: {last_err}"
        )

    def delete(self):
        """Delete the Space. Only works if this Sandbox created it."""
        if not self._owns_space:
            raise RuntimeError(
                f"This Sandbox did not create {self.space_id}. "
                f"Use self._hf_api.delete_repo() directly if you're sure."
            )
        print(f"Deleting sandbox: {self.space_id}...")
        self._hf_api.delete_repo(self.space_id, repo_type="space")
        self._client.close()
        print("Deleted.")

    def pause(self):
        """Pause the Space (stops billing, preserves state)."""
        self._hf_api.pause_space(self.space_id)

    def restart(self):
        """Restart the Space."""
        self._hf_api.restart_space(self.space_id)
        self._wait_for_api()

    @property
    def url(self) -> str:
        """Public URL of the Space."""
        return f"https://huggingface.co/spaces/{self.space_id}"

    @property
    def status(self) -> str:
        """Current Space stage (RUNNING, BUILDING, PAUSED, etc.)."""
        return self._hf_api.get_space_runtime(self.space_id).stage

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc):
        if self._owns_space:
            try:
                self.delete()
            except Exception as e:
                print(f"Warning: failed to delete sandbox: {e}", file=sys.stderr)
        self._client.close()

    # ── HTTP plumbing ─────────────────────────────────────────────

    def _call(
        self, endpoint: str, payload: dict, timeout: float | None = None
    ) -> ToolResult:
        # Strip leading slash for correct httpx base_url resolution
        endpoint = endpoint.lstrip("/")
        effective_timeout = timeout or self.timeout
        last_error = ""

        # Retry up to 3 times for transient failures (sandbox waking from
        # sleep returns empty / non-JSON responses while it starts up).
        for attempt in range(3):
            try:
                resp = self._client.post(
                    endpoint,
                    json=payload,
                    timeout=effective_timeout,
                )
                try:
                    data = resp.json()
                except (ValueError, UnicodeDecodeError):
                    # Non-JSON response — sandbox is likely still starting up.
                    body_preview = resp.text[:200] if resp.text else "(empty)"
                    last_error = (
                        f"Sandbox returned non-JSON response (HTTP {resp.status_code}): "
                        f"{body_preview}"
                    )
                    if attempt < 2:
                        time.sleep(3 * (attempt + 1))
                        continue
                    return ToolResult(success=False, error=last_error)

                if resp.status_code == 200:
                    return ToolResult(
                        success=data.get("success", True),
                        output=data.get("output", ""),
                        error=data.get("error", ""),
                    )
                return ToolResult(
                    success=False,
                    error=data.get("error", f"HTTP {resp.status_code}"),
                )
            except httpx.TimeoutException:
                return ToolResult(
                    success=False, error=f"Timeout after {effective_timeout}s"
                )
            except httpx.ConnectError:
                last_error = (
                    f"Cannot connect to sandbox. Is {self.space_id} running? "
                    f"Status: {self.status}"
                )
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                return ToolResult(success=False, error=last_error)
            except Exception as e:
                return ToolResult(success=False, error=str(e))

        return ToolResult(success=False, error=last_error or "Unknown error")

    # ── Tools ─────────────────────────────────────────────────────

    def bash(
        self,
        command: str,
        *,
        work_dir: str | None = None,
        timeout: int | None = None,
        description: str | None = None,
    ) -> ToolResult:
        return self._call(
            "bash",
            {
                "command": command,
                "work_dir": work_dir or self.work_dir,
                "timeout": min(timeout or self.timeout, MAX_TIMEOUT),
            },
            timeout=timeout,
        )

    def read(
        self, path: str, *, offset: int | None = None, limit: int | None = None
    ) -> ToolResult:
        self._files_read.add(path)
        return self._call(
            "read",
            {
                "path": path,
                "offset": offset,
                "limit": limit or (DEFAULT_READ_LIMIT if offset is None else None),
            },
        )

    def write(self, path: str, content: str) -> ToolResult:
        if path not in self._files_read:
            check = self._call("exists", {"path": path})
            if check.success and check.output == "true":
                return ToolResult(
                    success=False,
                    error=(
                        f"File {path} exists but has not been read this session. "
                        f"Read it first, or use sandbox_edit for targeted changes."
                    ),
                )
        result = self._call("write", {"path": path, "content": content})
        if result.success:
            self._files_read.add(path)
        return result

    def edit(
        self, path: str, old_str: str, new_str: str, *, replace_all: bool = False
    ) -> ToolResult:
        if old_str == new_str:
            return ToolResult(success=False, error="old_str and new_str are identical.")
        if path not in self._files_read:
            return ToolResult(
                success=False,
                error=f"File {path} has not been read this session. Read it first.",
            )
        return self._call(
            "edit",
            {
                "path": path,
                "old_str": old_str,
                "new_str": new_str,
                "replace_all": replace_all,
            },
        )

    def kill_all(self) -> ToolResult:
        """Kill all active bash processes on the sandbox. Used on cancellation."""
        return self._call("kill", {})

    # ── Tool schemas & dispatch ───────────────────────────────────

    TOOLS = {
        "bash": {
            "description": (
                "Run a shell command in the remote sandbox and return stdout/stderr.\n"
                "\n"
                "Commands run in a shell at the working directory (default /app). "
                "Each invocation is independent — use files in /app to persist state.\n"
                "\n"
                "AVOID using bash for operations covered by specialized tools:\n"
                "- File reading: use read (not cat/head/tail)\n"
                "- File editing: use edit (not sed/awk)\n"
                "- File writing: use write (not echo/cat <<EOF)\n"
                "\n"
                "For long-running tasks, background them:\n"
                "  nohup uv run train.py > /app/train.log 2>&1 &\n"
                "Then check with read on the log file.\n"
                "\n"
                "Chain dependent commands with &&. Independent commands should be "
                "separate bash calls (they can run in parallel).\n"
                "\n"
                "Timeout default 120s, max 600s."
            ),
            "parameters": {
                "type": "object",
                "required": ["command"],
                "additionalProperties": False,
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description (5-10 words, active voice). E.g. 'Install dependencies', 'Run training script'.",
                    },
                    "work_dir": {
                        "type": "string",
                        "description": "Working directory (default: /app).",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 240, max: 1200).",
                    },
                },
            },
        },
        "read": {
            "description": (
                "Read file contents with line numbers (cat -n format).\n"
                "\n"
                "Returns the first 2000 lines by default. For large files, use offset/limit "
                "to read a specific range. Line numbers always match the original file.\n"
                "\n"
                "Lines longer than 2000 chars are truncated.\n"
                "Cannot read directories — use bash with 'ls' instead."
            ),
            "parameters": {
                "type": "object",
                "required": ["path"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Start from this line (1-based). Only if file is too large.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of lines to read. Only if file is too large.",
                    },
                },
            },
        },
        "write": {
            "description": (
                "Create or overwrite a file. Creates parent directories as needed.\n"
                "\n"
                "For existing files, you MUST read the file first (system enforced). "
                "Prefer edit for modifications."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file content.",
                    },
                },
            },
        },
        "edit": {
            "description": (
                "Targeted edit via exact string replacement.\n"
                "\n"
                "Rules:\n"
                "- old_str must appear EXACTLY once (unless replace_all is true).\n"
                "- Include enough context in old_str for uniqueness.\n"
                "- old_str and new_str must differ.\n"
                "- Preserve indentation exactly.\n"
                "- To delete code, set new_str to empty string.\n"
                "- File MUST have been read this session (system enforced).\n"
                "- Do NOT include line number prefixes in old_str/new_str.\n"
                "\n"
                "Use replace_all=true for batch operations like variable renaming."
            ),
            "parameters": {
                "type": "object",
                "required": ["path", "old_str", "new_str"],
                "additionalProperties": False,
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact text to find (must differ from new_str).",
                    },
                    "new_str": {"type": "string", "description": "Replacement text."},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false).",
                        "default": False,
                    },
                },
            },
        },
    }

    @classmethod
    def tool_definitions(cls) -> list[dict]:
        return [{"name": name, **spec} for name, spec in cls.TOOLS.items()]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        dispatch = {
            "bash": lambda a: self.bash(
                a["command"],
                work_dir=a.get("work_dir"),
                timeout=a.get("timeout"),
                description=a.get("description"),
            ),
            "read": lambda a: self.read(
                a["path"],
                offset=a.get("offset"),
                limit=a.get("limit"),
            ),
            "write": lambda a: self.write(a["path"], a["content"]),
            "edit": lambda a: self.edit(
                a["path"],
                a["old_str"],
                a["new_str"],
                replace_all=a.get("replace_all", False),
            ),
        }
        fn = dispatch.get(name)
        if not fn:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        return fn(arguments)
