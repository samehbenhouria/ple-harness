#!/usr/bin/env python3
"""End-to-end agent-loop test at zero cost.

A fake OpenAI-compatible model server (stdlib only) plays a scripted model:
turn 1 -> exec(ls), turn 2 -> write_file(report.md), turn 3 -> final text.
A fake evaluator (same protocol handling as mock_evaluator.py) executes the
harness's requests in a temp workdir. Asserts: report.md written exactly,
`done` received with the model's final text, and >= 3 usage frames reported.

Run:  python tests/agent_loop_test.py
Exit 0 and "AGENT LOOP PASS" on success.
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import websockets

MODEL_PORT = 18742
SERVE_PORT = 18743
REPORT = "# Report\nFake model deliverable.\n"


class FakeModel(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            msg = {
                "role": "assistant", "content": "Listing the workdir first.",
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "exec", "arguments": json.dumps({"command": "ls -la"})},
                }],
            }
            finish = "tool_calls"
        elif len(tool_msgs) == 1:
            msg = {
                "role": "assistant", "content": "Writing the deliverable.",
                "tool_calls": [{
                    "id": "call_2", "type": "function",
                    "function": {"name": "write_file",
                                 "arguments": json.dumps({"path": "report.md", "content": REPORT})},
                }],
            }
            finish = "tool_calls"
        elif len(tool_msgs) == 2:
            msg = {
                "role": "assistant", "content": "Packaging and verifying the deliverable.",
                "tool_calls": [{
                    "id": "call_3", "type": "function",
                    "function": {"name": "verify_package",
                                 "arguments": json.dumps({"paths": ["report.md"], "zip_path": "bundle.zip"})},
                }],
            }
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": "Final summary from fake model."}
            finish = "stop"
        payload = {
            "id": "chatcmpl-fake", "object": "chat.completion", "model": "fake-model",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        }
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


async def run():
    server = HTTPServer(("127.0.0.1", MODEL_PORT), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    workdir = tempfile.mkdtemp(prefix="ple-agent-")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        **os.environ,
        "PORT": str(SERVE_PORT),
        "OPENAI_BASE_URL": f"http://127.0.0.1:{MODEL_PORT}/v1",
        "OPENAI_API_KEY": "test-key",
        "MODEL_NAME": "fake-model",
    }
    env.pop("HARNESS_MOCK", None)
    proc = subprocess.Popen(
        [sys.executable, os.path.join(root, "serve.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        ws = None
        for _ in range(50):
            try:
                ws = await websockets.connect(f"ws://127.0.0.1:{SERVE_PORT}", max_size=16 * 2**20, ping_interval=None)
                break
            except Exception:
                await asyncio.sleep(0.2)
        assert ws is not None, "server did not start"

        usage_frames = 0
        async with ws:
            await ws.send(json.dumps({
                "type": "task", "protocol": 1, "session": "agenttest1",
                "instruction": "Write report.md with a short report and summarize.",
                "workdir": workdir,
                "limits": {"agent_timeout_sec": 300, "exec_timeout_sec": 60,
                           "max_output_bytes": 100000, "max_write_bytes": 10000000},
                "mcp_servers": [],
            }))
            done_msg = None
            while True:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=60))
                t = frame.get("type")
                if t == "log":
                    continue
                if t == "usage":
                    usage_frames += 1
                    continue
                if t == "done":
                    done_msg = frame
                    break
                if t == "error":
                    raise RuntimeError(f"harness sent error: {frame}")
                if t == "exec":
                    p = subprocess.run(["bash", "-c", frame["command"]], cwd=workdir,
                                       capture_output=True, text=True, timeout=60)
                    await ws.send(json.dumps({
                        "type": "result", "id": frame["id"], "exit_code": p.returncode,
                        "output": p.stdout + p.stderr, "truncated": False, "timed_out": False,
                    }))
                elif t == "write_file":
                    path = os.path.join(workdir, frame["path"])
                    os.makedirs(os.path.dirname(path) or workdir, exist_ok=True)
                    data = base64.b64decode(frame.get("content_b64", ""))
                    with open(path, "ab" if frame.get("append") else "wb") as f:
                        f.write(data)
                    await ws.send(json.dumps({"type": "result", "id": frame["id"], "ok": True, "size": len(data)}))
                elif t == "read_file":
                    await ws.send(json.dumps({"type": "result", "id": frame["id"], "ok": False, "error": "not in test"}))
                elif t == "http":
                    await ws.send(json.dumps({"type": "result", "id": frame["id"], "ok": False, "error": "no MCP in test"}))
                else:
                    raise RuntimeError(f"unexpected frame: {frame}")

        report = os.path.join(workdir, "report.md")
        assert done_msg is not None, "no done frame"
        assert os.path.exists(report), "report.md not written"
        assert open(report).read() == REPORT, "report.md content mismatch"
        bundle = os.path.join(workdir, "bundle.zip")
        assert os.path.exists(bundle), "bundle.zip not created by verify_package"
        import zipfile
        with zipfile.ZipFile(bundle) as z:
            assert z.testzip() is None, "bundle.zip corrupt"
            assert z.namelist() == ["report.md"], f"bundle.zip entries: {z.namelist()}"
            assert z.read("report.md").decode() == REPORT, "bundle.zip content mismatch"
        assert "Final summary from fake model" in done_msg.get("final_message", ""), "final text missing"
        assert "report.md" in done_msg.get("final_message", ""), "deliverable not listed in summary"
        assert "bundle.zip" in done_msg.get("final_message", ""), "package not listed in summary"
        assert usage_frames >= 4, f"expected >=4 usage frames, got {usage_frames}"
        print(f"AGENT LOOP PASS - usage_frames={usage_frames} workdir={workdir}")
    finally:
        proc.terminate()
        server.shutdown()
        try:
            out, _ = proc.communicate(timeout=5)
            if proc.returncode not in (0, -15):
                print("server output:", out)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(run())
