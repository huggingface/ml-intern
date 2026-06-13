import importlib.util
import socket
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dev_server.py"


def bind_first_available_port(host: str) -> socket.socket:
    for port in range(45000, 45100):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, port))
            return sock
        except OSError:
            sock.close()
    raise AssertionError("No available test port found")


def load_dev_server_module():
    spec = importlib.util.spec_from_file_location("dev_server", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_host_url_wraps_ipv6_hosts():
    dev_server = load_dev_server_module()

    assert dev_server.host_url("::1", 7860) == "http://[::1]:7860"


def test_find_free_port_skips_occupied_port():
    dev_server = load_dev_server_module()

    with bind_first_available_port("127.0.0.1") as sock:
        sock.listen()
        occupied_port = sock.getsockname()[1]

        free_port = dev_server.find_free_port("127.0.0.1", occupied_port, 5)

    assert free_port > occupied_port


def test_classifies_repo_backend_process():
    dev_server = load_dev_server_module()
    process = dev_server.ProcessInfo(
        pid=123,
        pgid=123,
        command="uv run --frozen uvicorn main:app --host ::1 --port 7860",
        cwd=dev_server.BACKEND_DIR,
    )

    assert dev_server.classify_dev_server_process(process) == "backend"


def test_classifies_repo_frontend_vite_process():
    dev_server = load_dev_server_module()
    process = dev_server.ProcessInfo(
        pid=123,
        pgid=123,
        command="node ./node_modules/.bin/vite --host 127.0.0.1 --port 5173",
        cwd=dev_server.FRONTEND_DIR,
    )

    assert dev_server.classify_dev_server_process(process) == "frontend"


def test_ignores_matching_command_from_other_directory(tmp_path):
    dev_server = load_dev_server_module()
    process = dev_server.ProcessInfo(
        pid=123,
        pgid=123,
        command="uv run --frozen uvicorn main:app --host ::1 --port 7860",
        cwd=tmp_path,
    )

    assert dev_server.classify_dev_server_process(process) is None


def test_managed_stack_active_requires_all_processes(monkeypatch):
    dev_server = load_dev_server_module()
    state = {
        "processes": [
            {"pid": 1, "pgid": 1, "match_markers": ["uvicorn"]},
            {"pid": 2, "pgid": 2, "match_markers": ["npm"]},
        ]
    }

    monkeypatch.setattr(
        dev_server, "process_matches", lambda process: process["pid"] == 1
    )

    assert not dev_server.managed_stack_active(state)
