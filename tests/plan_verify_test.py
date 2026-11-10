#!/usr/bin/env python3
"""End-to-end plan-verify mode test at zero cost (harness v0.3).

Sets PLE_MODE=planverify and drives serve.py with a fake evaluator (real
exec / write_file / read_file in a temp workdir) and a fake
OpenAI-compatible model scripted by PHASE, detected from a marker in the
system prompt of each separate conversation:

- planner ("You are the planner..."): exec(ls), then emits the plan text;
- worker (the default agent prompt): writes report.md v1 - MISSING the
  plan's key required value - then declares the draft complete;
- verifier ("independent verifier"): runs in a FRESH conversation, re-reads
  report.md itself, REJECTS it with a named correction (the missing
  allowable load 182.4 kN); after the worker corrects it, a second fresh
  verifier pass ACCEPTS.

Asserts: every verifier request is a fresh conversation (it starts with
zero tool messages and contains NO worker tool results or worker text),
the final report.md is the corrected v2 content, `done` reports the
verifier's ACCEPT verdict and 1 correction round, and usage frames were
reported for every model call (2 planner + 2 worker + 2 verifier +
2 correction + 2 verifier = 10).

Run:  python tests/plan_verify_test.py
Exit 0 and "PLAN VERIFY PASS" on success.
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

MODEL_PORT = 18748
SERVE_PORT = 18749

PLAN = (
    "# Plan\n\n"
    "Deliverable: report.md in the workdir - beam check memo.\n"
    "Steps:\n"
    "1. Inventory the workdir.\n"
    "2. Write report.md for section W310x60, span 7.5 m, per CSA S16-19.\n"
    "Verification criteria: report.md must state the allowable load value 182.4 kN.\n"
)
REPORT_V1 = "# Beam check\n\nSection W310x60 reviewed against CSA S16-19.\nSpan: 7.5 m.\n"
REPORT_V2 = ("# Beam check\n\nSection W310x60 reviewed against CSA S16-19.\n"
             "Span: 7.5 m.\nAllowable load: 182.4 kN.\n")
REJECT_TEXT = (
    "VERDICT: REJECT\n\nCORRECTIONS:\n"
    "1. report.md: state the allowable load value 182.4 kN - the plan's "
    "verification criteria require it and it is absent from the deliverable."
)
ACCEPT_TEXT = (
    "VERDICT: ACCEPT\n\nAll checks pass: report.md states the allowable "
    "load 182.4 kN required by the plan's verification criteria."
)
WORKER_SECRET = "WORKER-ONLY draft narration 7f3a"   # must never leak into the verifier
CORRECTION_MARK = "REJECTED them"                     # harness wording to the worker

VERIFIER_REQUESTS = []   # every `messages` payload the verifier phase received


def tool_call_msg(call_id, name, args, content=""):
    return {
        "role": "assistant", "content": content,
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


class FakeModel(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        system = ""
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content") or ""
                break
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        n = len(tool_msgs)

        if "You are the planner" in system:
            if n == 0:
                msg = tool_call_msg("p_call_1", "exec", {"command": "ls -la"},
                                    "Inventorying the workdir for the plan.")
                finish = "tool_calls"
            else:
                msg = {"role": "assistant", "content": PLAN}
                finish = "stop"
        elif "independent verifier" in system:
            VERIFIER_REQUESTS.append(messages)
            if n == 0:
                msg = tool_call_msg("v_call_1", "read_file", {"path": "report.md"},
                                    "Re-reading the deliverable myself.")
                finish = "tool_calls"
            else:
                seen = tool_msgs[-1].get("content") or ""
                if "182.4" in seen:
                    msg = {"role": "assistant", "content": ACCEPT_TEXT}
                else:
                    msg = {"role": "assistant", "content": REJECT_TEXT}
                finish = "stop"
        else:  # worker conversation
            corrected = any(
                m.get("role") == "user" and CORRECTION_MARK in (m.get("content") or "")
                for m in messages
            )
            if corrected and n == 1:
                msg = tool_call_msg("w_call_2", "write_file",
                                    {"path": "report.md", "content": REPORT_V2},
                                    "Applying the verifier's correction.")
                finish = "tool_calls"
            elif corrected:
                msg = {"role": "assistant",
                       "content": "Corrections applied: report.md now states the allowable load 182.4 kN."}
                finish = "stop"
            elif n == 0:
                msg = tool_call_msg("w_call_1", "write_file",
                                    {"path": "report.md", "content": REPORT_V1},
                                    WORKER_SECRET)
                finish = "tool_calls"
            else:
                msg = {"role": "assistant",
                       "content": "Worker draft complete: report.md written per plan."}
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

    workdir = tempfile.mkdtemp(prefix="ple-planverify-")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {
        **os.environ,
        "PORT": str(SERVE_PORT),
        "OPENAI_BASE_URL": f"http://127.0.0.1:{MODEL_PORT}/v1",
        "OPENAI_API_KEY": "test-key",
        "MODEL_NAME": "fake-model",
        "PLE_MODE": "planverify",
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
                "type": "task", "protocol": 1, "session": "planverify1",
                "instruction": "Write report.md: a beam check memo for section W310x60.",
                "workdir": workdir,
                "limits": {"agent_timeout_sec": 600, "exec_timeout_sec": 60,
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
                    path = os.path.join(workdir, frame["path"])
                    if os.path.isfile(path):
                        data = open(path, "rb").read()
                        await ws.send(json.dumps({
                            "type": "result", "id": frame["id"], "ok": True,
                            "content_b64": base64.b64encode(data).decode(),
                        }))
                    else:
                        await ws.send(json.dumps({"type": "result", "id": frame["id"],
                                                  "ok": False, "error": "no such file"}))
                elif t == "http":
                    await ws.send(json.dumps({"type": "result", "id": frame["id"], "ok": False, "error": "no MCP in test"}))
                else:
                    raise RuntimeError(f"unexpected frame: {frame}")

        assert done_msg is not None, "no done frame"
        final_message = done_msg.get("final_message", "")

        # The worker really corrected the deliverable after the REJECT.
        report = os.path.join(workdir, "report.md")
        assert os.path.exists(report), "report.md not written"
        content = open(report).read()
        assert content == REPORT_V2, f"report.md is not the corrected v2: {content!r}"
        assert REPORT_V1 != REPORT_V2, "test bug: v1 and v2 identical"

        # The verifier ran exactly two FRESH passes (reject, then accept).
        fresh_starts = [req for req in VERIFIER_REQUESTS
                        if not any(m.get("role") == "tool" for m in req)]
        assert len(fresh_starts) == 2, \
            f"expected 2 fresh verifier conversations, got {len(fresh_starts)}"
        assert VERIFIER_REQUESTS, "verifier was never called"
        for req in VERIFIER_REQUESTS:
            joined = "\n".join(str(m.get("content") or "") for m in req)
            assert "bytes to report.md" not in joined, \
                "a worker write_file result leaked into the verifier conversation"
            assert WORKER_SECRET not in joined, \
                "worker assistant text leaked into the verifier conversation"
            assert "Worker draft complete" not in joined, \
                "the worker's closing text leaked into the verifier conversation"
            first_user = next(m.get("content") or "" for m in req if m.get("role") == "user")
            assert "# Plan" in first_user, "verifier did not receive the plan"
            assert "W310x60" in first_user, "verifier did not receive the instruction"

        # The `done` summary reports the verdict and the correction round.
        assert "final verifier verdict: ACCEPT" in final_message, \
            f"verdict missing from summary: {final_message!r}"
        assert "correction rounds used: 1" in final_message, \
            f"round count missing from summary: {final_message!r}"
        assert "report.md" in final_message, "deliverable not listed in summary"
        assert usage_frames >= 10, f"expected >=10 usage frames, got {usage_frames}"
        print(f"PLAN VERIFY PASS - usage_frames={usage_frames} "
              f"verifier_passes={len(fresh_starts)} workdir={workdir}")
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
