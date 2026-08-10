#!/usr/bin/env python3
"""Local bridge: browser <-> Sable UCI engine.

Serves web/index.html and exposes POST /api/move which forwards the game's
move list to a persistent engine subprocess and returns its bestmove plus
the search info lines.
"""
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(ROOT, "..", "target", "release", "sable")
PORT = 8375

lock = threading.Lock()
proc = None
ident = []


def engine():
    global proc, ident
    if proc is None or proc.poll() is not None:
        proc = subprocess.Popen(
            [ENGINE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        proc.stdin.write("uci\n")
        proc.stdin.flush()
        ident = []
        while True:
            line = proc.stdout.readline().strip()
            if not line or "uciok" in line:
                break
            if line.startswith("id "):
                ident.append(line[3:])
    return proc


def _search(moves, movetime):
    p = engine()
    pos = "position startpos"
    if moves:
        pos += " moves " + " ".join(moves)
    p.stdin.write(f"{pos}\ngo movetime {movetime}\n")
    p.stdin.flush()
    info = []
    while True:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("engine died")
        line = line.strip()
        if line.startswith("info"):
            info.append(line)
        elif line.startswith("bestmove"):
            return line.split()[1], info


def bestmove(moves, movetime):
    with lock:
        try:
            return _search(moves, movetime)
        except (RuntimeError, BrokenPipeError):
            # Engine crashed mid-search: restart it once and retry the request.
            global proc
            proc = None
            return _search(moves, movetime)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            with lock:
                engine()
            self._send(200, json.dumps({"engine": ident}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        if self.path != "/api/move":
            return self._send(404, b"{}")
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            moves = req.get("moves", [])
            movetime = max(50, min(int(req.get("movetime", 1000)), 10000))
            if not isinstance(moves, list) or not all(
                isinstance(m, str) and 4 <= len(m) <= 5 and m.isalnum() for m in moves
            ):
                return self._send(400, b'{"error": "bad move list"}')
        except (ValueError, TypeError):
            return self._send(400, b'{"error": "bad request"}')
        try:
            move, info = bestmove(moves, movetime)
            self._send(200, json.dumps({"bestmove": move, "info": info}).encode())
        except Exception as e:
            self._send(500, json.dumps({"error": str(e)}).encode())


if __name__ == "__main__":
    print(f"Sable bridge on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
