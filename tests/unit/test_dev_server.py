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
