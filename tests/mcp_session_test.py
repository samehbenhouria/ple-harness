#!/usr/bin/env python3
"""End-to-end MCP *session* test at zero cost (harness v0.2.2).

The world test (mcp_world_test.py) proved the http chain against a lenient
fake MCP. Real task servers speak Streamable HTTP and enforce session
discipline, which the harness - not the model - must own. This test proves
exactly that, against a strict fake MCP served by the fake evaluator:

- `initialize` returns an `Mcp-Session-Id` header (sess-123) plus a
  negotiated protocolVersion (2025-03-26);
- `tools/list` and `tools/call` return HTTP 400 unless the request carries
  that exact session header AND `notifications/initialized` was received
  first;
- `tools/call` answers in SSE (text/event-stream) with a progress event
  before the real result event;
- the scripted fake model sends NO session headers and NO Accept header:
  every bit of discipline must come from the harness.

Asserts: the server saw initialize -> notifications/initialized ->
tools/list -> tools/call with no 400, session + protocol-version headers
replayed on tools/list and tools/call, the harness's session note line in
the model's context, and beam_check.md citing the exact MCP value
(182.4 kN / B-1042) despite the SSE transport.

Run:  python tests/mcp_session_test.py
Exit 0 and "MCP SESSION PASS" on success.
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

MODEL_PORT = 18746
SERVE_PORT = 18747
MCP_URL = "http://mcp.invalid/records"  # never resolved: the fake evaluator intercepts http frames

SESSION_ID = "sess-123"
PROTOCOL_VERSION = "2025-03-26"
SESSION_NOTE = "[harness: MCP session established; notifications/initialized sent]"

RECORD = {
    "record_id": "B-1042",
    "section": "W310x60",
    "material": "CSA G40.21 Grade 350W",
    "span_m": 7.5,
    "allowable_load_kN": 182.4,
    "governing_clause": "CSA S16-19 Cl. 13.5",
}

MCP_LOG = []        # (method, lowered request headers, status returned)
NOTIFIED = {"done": False, "session_header_ok": False}
FAKE_STATE = {}     # what the fake model observed / extracted


# ---------------------------------------------------------------- fake MCP
def _hget(headers, name):
    lname = name.lower()
    for k, v in (headers or {}).items():
        if str(k).lower() == lname:
            return v
    return None


def mcp_handle(frame):
    """Strict Streamable-HTTP MCP. Returns (status, resp_headers, body)."""
    raw = base64.b64decode(frame.get("body_b64", "")).decode("utf-8")
    req_headers = {str(k).lower(): v for k, v in (frame.get("headers") or {}).items()}
    try:
        request = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        request = {}
    method = request.get("method", "")
    rid = request.get("id")

    def log(status):
        MCP_LOG.append((method, req_headers, status))

    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "fake-records-mcp", "version": "0.2.0"},
        }
        log(200)
        return 200, {"Mcp-Session-Id": SESSION_ID, "Content-Type": "application/json"}, \
            json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})
    if method == "notifications/initialized":
        NOTIFIED["session_header_ok"] = _hget(req_headers, "mcp-session-id") == SESSION_ID
        NOTIFIED["done"] = True
        log(202)
        return 202, {"Content-Type": "application/json"}, ""
    # Everything else requires the session header AND the notification.
    if _hget(req_headers, "mcp-session-id") != SESSION_ID or not NOTIFIED["done"]:
        log(400)
        err = {"jsonrpc": "2.0", "id": rid,
               "error": {"code": -32000,
                         "message": "missing/invalid Mcp-Session-Id or notifications/initialized not received"}}
        return 400, {"Content-Type": "application/json"}, json.dumps(err)
    if method == "tools/list":
        result = {"tools": [{
            "name": "get_record",
            "description": "Return one engineering record by id.",
            "inputSchema": {
                "type": "object",
                "properties": {"record_id": {"type": "string"}},
                "required": ["record_id"],
            },
        }]}
        log(200)
        return 200, {"Content-Type": "application/json"}, \
            json.dumps({"jsonrpc": "2.0", "id": rid, "result": result})
    if method == "tools/call":
        params = request.get("params") or {}
        if params.get("name") != "get_record":
            log(400)
            err = {"jsonrpc": "2.0", "id": rid,
                   "error": {"code": -32602, "message": f"unknown tool: {params.get('name')}"}}
            return 400, {"Content-Type": "application/json"}, json.dumps(err)
        progress = {"jsonrpc": "2.0", "method": "notifications/progress",
                    "params": {"progressToken": "tok-1", "progress": 1, "total": 2}}
        result_rpc = {"jsonrpc": "2.0", "id": rid,
                      "result": {"content": [{"type": "text", "text": json.dumps(RECORD)}],
                                 "isError": False}}
        body = (f"event: message\ndata: {json.dumps(progress)}\n\n"
                f"event: message\ndata: {json.dumps(result_rpc)}\n\n")
        log(200)
        return 200, {"Content-Type": "text/event-stream",
                     "Mcp-Session-Id": SESSION_ID}, body
    log(404)
    err = {"jsonrpc": "2.0", "id": rid,
           "error": {"code": -32601, "message": f"method not found: {method}"}}
    return 404, {"Content-Type": "application/json"}, json.dumps(err)


# ------------------------------------------------------------- fake model
def rpc_body(method, rpc_id, params=None):
    return json.dumps({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params or {}})


def http_call(call_id, body):
    # Deliberately NO Accept and NO session headers: the harness must add
    # the Accept header and own the whole session discipline.
    return {
        "role": "assistant", "content": "",
        "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "http", "arguments": json.dumps({
                "method": "POST", "url": MCP_URL,
                "headers": {"Content-Type": "application/json"},
                "body": body,
            })},
        }],
    }


def extract_record(messages):
    """Pull the beam record out of the tools/call tool result, as a real
    model would read it from its context (harness prepends status/headers)."""
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content") or ""
        start = content.find("{")
        if start < 0:
            continue
        try:
            rpc, _ = json.JSONDecoder().raw_decode(content[start:])
        except json.JSONDecodeError:
            continue
        result = rpc.get("result") or {}
        for item in result.get("content") or []:
            if item.get("type") == "text" and "allowable_load_kN" in item.get("text", ""):
                return json.loads(item["text"])
    return None


class FakeModel(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages", [])
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        n = len(tool_msgs)
        if n == 0:
            msg = http_call("call_1", rpc_body("initialize", 1, {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                "clientInfo": {"name": "fake-model", "version": "0.1"},
            }))
            msg["content"] = "Opening an MCP session with the records server."
            finish = "tool_calls"
        elif n == 1:
            # The initialize result must carry the harness's session note.
            init_result = tool_msgs[-1].get("content") or ""
            FAKE_STATE["session_note_seen"] = SESSION_NOTE in init_result
            FAKE_STATE["session_header_shown"] = SESSION_ID in init_result
            msg = http_call("call_2", rpc_body("tools/list", 2))
            msg["content"] = "Session open. Listing the server's tools."
            finish = "tool_calls"
        elif n == 2:
            msg = http_call("call_3", rpc_body("tools/call", 3, {
                "name": "get_record", "arguments": {"record_id": RECORD["record_id"]},
            }))
            msg["content"] = "Fetching beam record B-1042."
            finish = "tool_calls"
        elif n == 3:
            record = extract_record(messages)
            assert record is not None, "fake model could not read the MCP record"
            FAKE_STATE["record"] = record
            load = record["allowable_load_kN"]
            deliverable = (
                f"# Beam record check - {record['record_id']}\n\n"
                f"Source: records MCP server (tool get_record).\n\n"
                f"- Section: {record['section']} ({record['material']})\n"
                f"- Span: {record['span_m']} m\n"
                f"- Allowable load: {load} kN\n"
                f"- Governing clause: {record['governing_clause']}\n"
            )
            msg = {
                "role": "assistant", "content": "Writing the deliverable with the MCP value.",
                "tool_calls": [{
                    "id": "call_4", "type": "function",
                    "function": {"name": "write_file",
                                 "arguments": json.dumps({"path": "beam_check.md", "content": deliverable})},
                }],
            }
            finish = "tool_calls"
        else:
            record = FAKE_STATE["record"]
            msg = {"role": "assistant",
                   "content": (f"Final summary from fake model: beam_check.md written; "
                               f"record {record['record_id']} allowable load "
                               f"{record['allowable_load_kN']} kN cited from the MCP world server.")}
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


# --------------------------------------------------------- fake evaluator
async def run():
    server = HTTPServer(("127.0.0.1", MODEL_PORT), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    workdir = tempfile.mkdtemp(prefix="ple-mcp-session-")
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
                "type": "task", "protocol": 1, "session": "mcpsession1",
                "instruction": ("Consult the records MCP server, fetch beam record B-1042, "
                                "and write beam_check.md citing its allowable load."),
                "workdir": workdir,
                "limits": {"agent_timeout_sec": 600, "exec_timeout_sec": 60,
                           "max_output_bytes": 100000, "max_write_bytes": 10000000},
                "mcp_servers": [{"name": "records", "url": MCP_URL, "transport": "streamable-http"}],
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
                if t == "http":
                    status, resp_headers, resp_body = mcp_handle(frame)
                    await ws.send(json.dumps({
                        "type": "result", "id": frame["id"], "ok": True, "status": status,
                        "headers": resp_headers,
                        "body_b64": base64.b64encode(resp_body.encode()).decode(),
                    }))
                elif t == "exec":
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
                else:
                    raise RuntimeError(f"unexpected frame: {frame}")

        assert done_msg is not None, "no done frame"
        methods = [m for (m, _h, _s) in MCP_LOG]
        statuses = [s for (_m, _h, s) in MCP_LOG]
        assert methods == ["initialize", "notifications/initialized", "tools/list", "tools/call"], \
            f"MCP sequence: {methods}"
        assert statuses == [200, 202, 200, 200], f"unexpected statuses (a 400 means broken discipline): {statuses}"
        assert NOTIFIED["done"] and NOTIFIED["session_header_ok"], "notification missing or without session header"
        # The harness must have injected the streamable Accept on initialize...
        init_headers = MCP_LOG[0][1]
        assert "text/event-stream" in (_hget(init_headers, "accept") or ""), \
            f"harness did not add the streamable Accept header: {init_headers}"
        # ...and replayed session + protocol version on tools/list and tools/call.
        for method, hdrs, _s in MCP_LOG[2:]:
            assert _hget(hdrs, "mcp-session-id") == SESSION_ID, \
                f"{method} missing/wrong session header: {hdrs}"
            assert _hget(hdrs, "mcp-protocol-version") == PROTOCOL_VERSION, \
                f"{method} missing/wrong protocol version header: {hdrs}"
        assert FAKE_STATE.get("session_note_seen"), "model never saw the harness session note"
        assert FAKE_STATE.get("session_header_shown"), "model never saw the session id header line"
        deliverable = os.path.join(workdir, "beam_check.md")
        assert os.path.exists(deliverable), "beam_check.md not written"
        content = open(deliverable).read()
        assert str(RECORD["allowable_load_kN"]) in content, \
            f"exact MCP value {RECORD['allowable_load_kN']} missing from deliverable: {content!r}"
        assert RECORD["record_id"] in content, "record id missing from deliverable"
        assert "beam_check.md" in done_msg.get("final_message", ""), "deliverable not listed in summary"
        assert usage_frames >= 1, f"expected >=1 usage frame, got {usage_frames}"
        print(f"MCP SESSION PASS - usage_frames={usage_frames} mcp_sequence={methods} "
              f"statuses={statuses} workdir={workdir}")
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
