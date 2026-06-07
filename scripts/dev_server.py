#!/usr/bin/env python3
"""Run the local web stack with conflict-free ports.

Usage:
    uv run --frozen python scripts/dev_server.py up
    uv run --frozen python scripts/dev_server.py down
    uv run --frozen python scripts/dev_server.py restart
    uv run --frozen python scripts/dev_server.py status
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATE_DIR = PROJECT_ROOT / "scratch" / "dev-server"
STATE_PATH = STATE_DIR / "state.json"
DEFAULT_BACKEND_HOST = "::1"
DEFAULT_FRONTEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = 7860
DEFAULT_FRONTEND_PORT = 5173
DEFAULT_PORT_WINDOW = 100
DEFAULT_TIMEOUT_SECONDS = 30.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def host_url(host: str, port: int) -> str:
    if ":" in host and not host.startswith("["):
        return f"http://[{host}]:{port}"
    return f"http://{host}:{port}"


def state_exists() -> bool:
    return STATE_PATH.exists()


def load_state() -> dict[str, Any] | None:
    if not state_exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return None


def save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def clear_state() -> None:
    try:
        STATE_PATH.unlink()
    except FileNotFoundError:
        pass


def port_is_free(host: str, port: int) -> bool:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False

    for family, socktype, proto, _, sockaddr in infos:
        with socket.socket(family, socktype, proto) as sock:
            sock.settimeout(0.2)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(sockaddr)
            except OSError:
                return False
    return True


def find_free_port(host: str, preferred_port: int, port_window: int) -> int:
    final_port = min(65535, preferred_port + port_window)
    for port in range(preferred_port, final_port + 1):
        if port_is_free(host, port):
            return port
    raise RuntimeError(
        f"No free port found on {host} from {preferred_port} to {final_port}."
    )


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_command(pid: int) -> str:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def process_matches(process: dict[str, Any]) -> bool:
    pid = int(process["pid"])
    if not process_alive(pid):
        return False

    command = process_command(pid)
    markers = process.get("match_markers", [])
    return bool(command) and all(marker in command for marker in markers)


def active_processes(state: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not state:
        return []
    return [
        process
        for process in state.get("processes", [])
        if isinstance(process, dict) and process_matches(process)
    ]


def stop_process(process: dict[str, Any], timeout: float) -> None:
    pid = int(process["pid"])
    pgid = int(process.get("pgid", pid))
    name = process.get("name", str(pid))

    if not process_matches(process):
        print(f"{name}: not running")
        return

    print(f"{name}: stopping PID {pid}")
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not process_alive(pid):
            return
        time.sleep(0.1)

    print(f"{name}: forcing PID {pid}")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        os.kill(pid, signal.SIGKILL)


def stop_state(state: dict[str, Any] | None, timeout: float) -> None:
    if not state:
        print("No managed dev server state found.")
        clear_state()
        return

    processes = state.get("processes", [])
    for process in reversed(processes):
        if isinstance(process, dict):
            stop_process(process, timeout)
    clear_state()


def wait_for_http(name: str, url: str, timeout: float, log_path: Path) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(0.25)

    print(f"{name} did not become ready at {url}. See {log_path}.")
    return False


def start_process(
    name: str,
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    match_markers: list[str],
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("ab")
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_file.close()

    return {
        "name": name,
        "pid": proc.pid,
        "pgid": os.getpgid(proc.pid),
        "command": command,
        "cwd": str(cwd),
        "log": str(log_path),
        "match_markers": match_markers,
    }


def ensure_frontend_dependencies(args: argparse.Namespace) -> int:
    node_modules = FRONTEND_DIR / "node_modules"
    if node_modules.exists():
        return 0
    if not args.install:
        print(
            "frontend/node_modules is missing. Run `cd frontend && npm ci`, "
            "or rerun this command with `--install`."
        )
        return 1

    result = subprocess.run(["npm", "ci"], cwd=FRONTEND_DIR, check=False)
    return result.returncode


def build_backend_env(frontend_url: str) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("ML_INTERN_CORS_ORIGINS", "")
    origins = [origin for origin in existing.split(",") if origin]
    origins.extend(
        [
            frontend_url,
            frontend_url.replace("127.0.0.1", "localhost"),
        ]
    )
    env["ML_INTERN_CORS_ORIGINS"] = ",".join(dict.fromkeys(origins))
    return env


def build_frontend_env(backend_url: str, frontend_port: int) -> dict[str, str]:
    env = os.environ.copy()
    env["VITE_BACKEND_PROXY_TARGET"] = backend_url
    env["VITE_DEV_SERVER_PORT"] = str(frontend_port)
    return env


def command_up(args: argparse.Namespace) -> int:
    existing_state = load_state()
    active = active_processes(existing_state)
    if active and not args.replace:
        print("Managed dev server is already running.")
        print_status(existing_state)
        return 0
    if active:
        stop_state(existing_state, args.stop_timeout)
    elif state_exists():
        clear_state()

    if ensure_frontend_dependencies(args) != 0:
        return 1

    try:
        backend_port = find_free_port(
            args.backend_host, args.backend_port, args.port_window
        )
        frontend_port = find_free_port(
            args.frontend_host, args.frontend_port, args.port_window
        )
    except RuntimeError as error:
        print(error)
        return 1

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    backend_url = host_url(args.backend_host, backend_port)
    frontend_url = host_url(args.frontend_host, frontend_port)
    backend_log = STATE_DIR / "backend.log"
    frontend_log = STATE_DIR / "frontend.log"

    backend_command = [
        "uv",
        "run",
        "--frozen",
        "uvicorn",
        "main:app",
        "--host",
        args.backend_host,
        "--port",
        str(backend_port),
    ]
    frontend_command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        args.frontend_host,
        "--port",
        str(frontend_port),
        "--strictPort",
        "--clearScreen",
        "false",
    ]

    processes: list[dict[str, Any]] = []
    state = {
        "started_at": utc_now(),
        "backend_url": backend_url,
        "frontend_url": frontend_url,
        "frontend_proxy_health_url": f"{frontend_url}/api",
        "state_path": str(STATE_PATH),
        "processes": processes,
    }

    try:
        processes.append(
            start_process(
                "backend",
                backend_command,
                BACKEND_DIR,
                build_backend_env(frontend_url),
                backend_log,
                ["uvicorn", "main:app"],
            )
        )
        if not wait_for_http(
            "backend", f"{backend_url}/api", args.timeout, backend_log
        ):
            raise RuntimeError("backend failed to start")

        processes.append(
            start_process(
                "frontend",
                frontend_command,
                FRONTEND_DIR,
                build_frontend_env(backend_url, frontend_port),
                frontend_log,
                ["npm", "run", "dev"],
            )
        )
        if not wait_for_http(
            "frontend proxy", f"{frontend_url}/api", args.timeout, frontend_log
        ):
            raise RuntimeError("frontend failed to start")
    except RuntimeError:
        stop_state(state, args.stop_timeout)
        return 1

    save_state(state)
    print("Started managed dev server.")
    print_status(state)
    return 0


def command_down(args: argparse.Namespace) -> int:
    stop_state(load_state(), args.stop_timeout)
    return 0


def command_restart(args: argparse.Namespace) -> int:
    stop_state(load_state(), args.stop_timeout)
    args.replace = False
    return command_up(args)


def print_status(state: dict[str, Any] | None) -> None:
    if not state:
        print("No managed dev server state found.")
        return

    print(f"Frontend: {state.get('frontend_url')}/")
    print(f"Backend:  {state.get('backend_url')}/api")
    print(f"Proxy:    {state.get('frontend_proxy_health_url')}")
    print(f"State:    {state.get('state_path', STATE_PATH)}")

    for process in state.get("processes", []):
        if not isinstance(process, dict):
            continue
        status = "running" if process_matches(process) else "stopped"
        print(
            f"{process.get('name')}: {status}, "
            f"pid={process.get('pid')}, log={process.get('log')}"
        )


def command_status(args: argparse.Namespace) -> int:
    print_status(load_state())
    return 0


def add_common_up_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend-host", default=DEFAULT_BACKEND_HOST)
    parser.add_argument("--frontend-host", default=DEFAULT_FRONTEND_HOST)
    parser.add_argument("--backend-port", type=int, default=DEFAULT_BACKEND_PORT)
    parser.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT)
    parser.add_argument("--port-window", type=int, default=DEFAULT_PORT_WINDOW)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--stop-timeout", type=float, default=5.0)
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run npm ci if frontend/node_modules is missing.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    up = subparsers.add_parser("up", help="Start backend and frontend dev servers.")
    add_common_up_options(up)
    up.add_argument(
        "--replace",
        action="store_true",
        help="Stop the managed server first if one is already running.",
    )
    up.set_defaults(func=command_up)

    down = subparsers.add_parser("down", help="Stop the managed dev servers.")
    down.add_argument("--stop-timeout", type=float, default=5.0)
    down.set_defaults(func=command_down)

    restart = subparsers.add_parser("restart", help="Stop and start dev servers.")
    add_common_up_options(restart)
    restart.set_defaults(func=command_restart)

    status = subparsers.add_parser("status", help="Show managed dev server status.")
    status.set_defaults(func=command_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
