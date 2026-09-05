"""JSON-line TCP client used by GodotBackend and spike runners."""

from __future__ import annotations

import json
import socket
import time
from typing import Any


class JsonLineClient:
    def __init__(self, host: str, port: int, timeout: float = 120.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout)
        self._buf = b""

    def send(self, obj: dict[str, Any]) -> None:
        self.sock.sendall((json.dumps(obj) + "\n").encode("utf-8"))

    def recv(self) -> dict[str, Any]:
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line, self._buf = self._buf[:nl], self._buf[nl + 1 :]
                if not line.strip():
                    continue
                return json.loads(line.decode("utf-8"))
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Godot closed the TCP connection")
            self._buf += chunk

    def call(self, obj: dict[str, Any]) -> dict[str, Any]:
        self.send(obj)
        return self.recv()

    def call_expect(self, obj: dict[str, Any], cmd: str) -> dict[str, Any]:
        """Like call(), but drops async messages (step_result etc.) whose
        'cmd' differs, so probe commands work while the sim is running."""
        self.send(obj)
        while True:
            msg = self.recv()
            if msg.get("cmd") == cmd:
                return msg

    def close(self) -> None:
        try:
            self.send({"cmd": "close"})
            try:
                self.recv()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


def wait_connect(
    host: str, port: int, timeout: float = 20.0, recv_timeout: float = 120.0
) -> JsonLineClient:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            return JsonLineClient(host, port, timeout=recv_timeout)
        except OSError as e:
            last = e
            time.sleep(0.05)
    raise ConnectionError(f"could not connect to {host}:{port}: {last}")
