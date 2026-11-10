#!/usr/bin/env python3
"""Idris PLE harness - minimal agent server for Physical-Engineering's Last Exam.

Speaks the PLE serve protocol v1 (docs/protocol.md of google-research/ple):
a WebSocket server; per connection the evaluator sends one task message and
executes our exec/read_file/write_file/http requests inside the task
container until we send `done` or `error`.

The agent loop calls the sponsored model through the OpenAI Chat Completions
API using only environment variables (OPENAI_BASE_URL / OPENAI_API_KEY /
MODEL_NAME), so the same code runs on any of the 8 google-cloud models.

Zero-cost test mode: HARNESS_MOCK=1 runs a scripted 3-step session
(list -> write deliverable -> done) without any model call.

v0.2.1: `verify_package` tool - deterministic package assembly (sorted
entries, fixed 1980 timestamps) with read-back verification (entry list +
integrity test) before `done`; born from the project inventory: 24 of the
29 dev tasks deliver a package (submission.zip or a world-app folder).

v0.2.2: MCP session management in the `http` tool - the harness (not the
model) owns the Streamable-HTTP session per task MCP server: it captures
the `Mcp-Session-Id` and negotiated `MCP-Protocol-Version` returned by
`initialize`, replays them on later calls, sends
`notifications/initialized` itself, decodes SSE responses down to the last
JSON-RPC payload, surfaces useful response headers, and gives http
results a dedicated 120k-char cap (tool catalogs can reach ~500 KB).

Competences mobilised in this part (system registry): COMP-D4-04 (agent
design patterns: plan -> tools -> verify -> finish), COMP-D4-01 (MCP over
the protocol's http bridge), COMP-D4-02 (state/checkpoint discipline for
long runs: explicit time budget, resumable summary).
"""

import asyncio
import base64
import json
import os
import time

import websockets

MAX_FRAME = 16 * 2**20
TOOL_RESULT_CHAR_CAP = 30000
HTTP_RESULT_CHAR_CAP = 120000  # MCP tool catalogs can reach ~500 KB

SYSTEM_PROMPT = """You are an autonomous engineering agent solving a real-world \
physical-engineering task (structural, civil, electrical or mechanical) inside \
an isolated project container. You act through tools: exec (bash), read_file, \
write_file, and http (to reach the task's MCP servers when present).

Method (follow it in order):
1. INVENTORY first: list the workdir recursively, identify every project file,
   and note superseded/revised versions - the newest governing revision wins.
2. Read the task instruction again and extract the exact deliverables, paths,
   formats, codes/standards and constraints it names.
3. PLAN briefly in your head, then compute with real tools: run Python in the
   container for every calculation (never mental arithmetic), check units at
   each step, and verify results against the governing code tables or limits.
4. Draft the deliverable files, then RE-READ the instruction and verify each
   requirement against what you wrote. Fix gaps before finishing.
5. If the deliverable is a package (a submission.zip or a set of files in a
   folder), call verify_package to assemble and verify it BEFORE finishing:
   never finish on an unverified package.
6. Finish with a clear final summary: what was produced, where each file is,
   the key numbers, and any assumption you had to make.

Rules: prefer plain deliverable formats (.md/.txt/.csv/.json/.xlsx/.pdf). \
Never invent a code clause, a table value or a file you did not read. If data \
is missing, say so in the deliverable instead of fabricating it. Keep each \
step small enough to verify. You have a hard time budget; do not start a \
computation you cannot finish and check.
Engineering discipline (from the project's capability file): no number in a \
deliverable may come from prose - every figure is produced by a tool and the \
deliverable states its inputs and formula. For any acceptance or review \
task, write the governing criterion and its source BEFORE your verdict. If \
the task lives in MCP world applications, read the whole dossier or declare \
in the deliverable exactly which parts you did not read. Code you deliver \
(a script, an IFC model) must be re-parsed or auto-tested before finishing, \
not merely re-read."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Run a bash command inside the task container (working directory = task workdir). Files persist between calls; cwd and env vars do not.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 120},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the task container (max 10 MB). Returns base64 content decoded as text when possible.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a text file into the task container (creates parent directories). Use append=true for chunked writes over 10 MB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                    "append": {"type": "boolean", "default": False},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http",
            "description": "Send an HTTP request from inside the task container. Only the task's MCP server URLs are reachable. Used for MCP JSON-RPC calls (initialize, tools/list, tools/call).",
            "parameters": {
                "type": "object",
                "properties": {
                    "method": {"type": "string", "default": "POST"},
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                    "body": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_package",
            "description": "Assemble and verify a deliverable package inside the task container. Walks the given file/folder paths, optionally writes them into a deterministic zip (sorted entries, fixed timestamps) at zip_path, then reads the package back: full file/entry list with sizes, missing paths, and a zip integrity test. Call this before finishing any package-shaped deliverable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "zip_path": {"type": "string"},
                },
                "required": ["paths"],
            },
        },
    },
]


class Cancelled(Exception):
    pass


class Bridge:
    """One-request-at-a-time protocol bridge (simple and safe)."""

    def __init__(self, ws):
        self.ws = ws
        self.n = 0

    async def call(self, request):
        self.n += 1
        rid = str(self.n)
        await self.ws.send(json.dumps({**request, "id": rid}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("type") == "cancel":
                raise Cancelled(msg.get("reason", "cancelled"))
            if msg.get("type") == "result" and msg.get("id") == rid:
                return msg

    async def log(self, text):
        await self.ws.send(json.dumps({"type": "log", "text": str(text)[:4000]}))

    async def usage(self, model, usage_obj):
        if usage_obj is None:
            return
        await self.ws.send(json.dumps({
            "type": "usage",
            "model": model,
            "input_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
            "cache_read_tokens": 0,
        }))


async def run_tool(bridge, name, args, state):
    """Execute one model tool call through the bridge; return text for the model."""
    if name == "exec":
        r = await bridge.call({
            "type": "exec",
            "command": args["command"],
            "timeout_sec": int(args.get("timeout_sec", 120)),
        })
        out = f"exit_code={r.get('exit_code')} timed_out={r.get('timed_out')}\n{r.get('output', '')}"
        return out[:TOOL_RESULT_CHAR_CAP]
    if name == "read_file":
        r = await bridge.call({"type": "read_file", "path": args["path"]})
        if not r.get("ok"):
            return f"read_file failed: {r.get('error', 'unknown error')}"
        raw = base64.b64decode(r.get("content_b64", ""))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return f"[binary file, {len(raw)} bytes - not inlined]"
        return text[:TOOL_RESULT_CHAR_CAP]
    if name == "write_file":
        content = args["content"]
        r = await bridge.call({
            "type": "write_file",
            "path": args["path"],
            "content_b64": base64.b64encode(content.encode("utf-8")).decode(),
            "append": bool(args.get("append", False)),
        })
        if not r.get("ok"):
            return f"write_file failed: {r.get('error', 'unknown error')}"
        state["deliverables"].append(args["path"])
        return f"wrote {r.get('size')} bytes to {args['path']}"
    if name == "http":
        return await handle_http(bridge, args, state)
    if name == "verify_package":
        return await verify_package(bridge, args, state)
    return f"unknown tool: {name}"


def _header_get(headers, name):
    """Case-insensitive header lookup on a plain dict."""
    lname = name.lower()
    for k, v in (headers or {}).items():
        if str(k).lower() == lname:
            return v
    return None


def _mcp_server_key(url, state):
    """Longest task mcp_servers URL that prefixes this request URL."""
    best = None
    for srv in state.get("mcp_servers") or []:
        surl = srv.get("url") if isinstance(srv, dict) else None
        if surl and url.startswith(surl) and (best is None or len(surl) > len(best)):
            best = surl
    return best


def _rpc_method(body_text):
    """JSON-RPC method of a request body, or None if not a JSON-RPC call."""
    try:
        req = json.loads(body_text) if body_text else None
    except json.JSONDecodeError:
        return None
    if isinstance(req, dict):
        return req.get("method")
    return None


def _decode_http_body(text, content_type):
    """Return (rendered_text, was_sse).

    Plain bodies pass through untouched. SSE bodies (Content-Type
    text/event-stream, or any `data:` lines) are reduced to the last
    complete JSON payload - the JSON-RPC result - so the model never has
    to parse the event stream itself.
    """
    is_sse = "text/event-stream" in (content_type or "").lower()
    has_data = any(ln.lstrip().lower().startswith("data:") for ln in text.splitlines())
    if not is_sse and not has_data:
        return text, False
    payloads, current = [], []
    for ln in text.splitlines() + [""]:
        s = ln.strip()
        if s.lower().startswith("data:"):
            current.append(s[len("data:"):].strip())
        elif current:
            payloads.append("\n".join(current))
            current = []
    last = None
    for payload in payloads:
        try:
            json.loads(payload)
            last = payload
        except json.JSONDecodeError:
            continue
    return (last if last is not None else text), True


async def handle_http(bridge, args, state):
    """One http bridge call with MCP Streamable-HTTP session discipline.

    The harness (not the model) owns the session: it captures the
    Mcp-Session-Id returned by `initialize`, replays it together with the
    negotiated MCP-Protocol-Version on later calls to the same server,
    sends `notifications/initialized` itself, decodes SSE replies, and
    surfaces the useful response headers in the tool result.
    """
    body = args.get("body", "")
    url = args["url"]
    headers = dict(args.get("headers") or {})
    server_key = _mcp_server_key(url, state)
    if server_key:
        if _header_get(headers, "accept") is None:
            headers["Accept"] = "application/json, text/event-stream"
        session = (state.get("mcp_sessions") or {}).get(server_key)
        if session:
            if _header_get(headers, "mcp-session-id") is None and session.get("session_id"):
                headers["Mcp-Session-Id"] = session["session_id"]
            if _header_get(headers, "mcp-protocol-version") is None and session.get("protocol_version"):
                headers["MCP-Protocol-Version"] = session["protocol_version"]
    r = await bridge.call({
        "type": "http",
        "method": args.get("method", "POST"),
        "url": url,
        "headers": headers,
        "body_b64": base64.b64encode(body.encode("utf-8")).decode() if body else "",
    })
    if not r.get("ok"):
        return f"http failed: {r.get('error', 'unknown error')}"
    resp_headers = r.get("headers") or {}
    text = base64.b64decode(r.get("body_b64", "")).decode("utf-8", "replace")
    content_type = _header_get(resp_headers, "content-type") or ""
    decoded, _was_sse = _decode_http_body(text, content_type)
    notes = []
    if server_key and _rpc_method(body) == "initialize":
        sid = _header_get(resp_headers, "mcp-session-id")
        if sid:
            version = None
            try:
                parsed = json.loads(decoded)
                if isinstance(parsed, dict):
                    version = (parsed.get("result") or {}).get("protocolVersion")
            except json.JSONDecodeError:
                pass
            state.setdefault("mcp_sessions", {})[server_key] = {
                "session_id": sid, "protocol_version": version,
            }
            notif_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Mcp-Session-Id": sid,
            }
            if version:
                notif_headers["MCP-Protocol-Version"] = version
            notif_body = json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"})
            await bridge.call({
                "type": "http", "method": "POST", "url": url,
                "headers": notif_headers,
                "body_b64": base64.b64encode(notif_body.encode("utf-8")).decode(),
            })
            notes.append("[harness: MCP session established; notifications/initialized sent]")
    lines = [f"status={r.get('status')}"]
    for hname in ("Mcp-Session-Id", "Content-Type"):
        val = _header_get(resp_headers, hname)
        if val is not None:
            lines.append(f"header {hname}: {val}")
    lines.extend(notes)
    if len(decoded) > HTTP_RESULT_CHAR_CAP:
        total = len(decoded)
        decoded = decoded[:HTTP_RESULT_CHAR_CAP] + (
            f"\n[truncated by harness: http result was {total} chars; "
            f"showing first {HTTP_RESULT_CHAR_CAP}]"
        )
    lines.append(decoded)
    return "\n".join(lines)


PACKAGE_SCRIPT = r'''
import json, os, zipfile
with open(".ple_package_cfg.json") as fh:
    cfg = json.load(fh)
report = {"files": [], "missing": [], "zip": None}
roots = []
for p in cfg["paths"]:
    if os.path.isfile(p):
        roots.append(p)
    elif os.path.isdir(p):
        for dp, dn, fn in os.walk(p):
            dn.sort()
            for f in sorted(fn):
                roots.append(os.path.join(dp, f))
    else:
        report["missing"].append(p)
roots = sorted(set(roots))
for p in roots:
    report["files"].append({"path": p, "size": os.path.getsize(p)})
if cfg.get("zip_path"):
    zp = cfg["zip_path"]
    parent = os.path.dirname(zp)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in roots:
            zi = zipfile.ZipInfo(p.lstrip("/"), date_time=(1980, 1, 1, 0, 0, 0))
            zi.external_attr = 0o644 << 16
            with open(p, "rb") as fh:
                z.writestr(zi, fh.read())
    with zipfile.ZipFile(zp) as z:
        bad = z.testzip()
        names = z.namelist()
    report["zip"] = {"path": zp, "size": os.path.getsize(zp),
                     "entries": names, "corrupt_entry": bad}
print(json.dumps(report))
'''


async def verify_package(bridge, args, state):
    """Deterministic package assembly + read-back verification (v0.2.1)."""
    cfg = {"paths": args.get("paths", []), "zip_path": args.get("zip_path")}
    for fname, content in ((".ple_package.py", PACKAGE_SCRIPT),
                           (".ple_package_cfg.json", json.dumps(cfg))):
        r = await bridge.call({
            "type": "write_file", "path": fname,
            "content_b64": base64.b64encode(content.encode("utf-8")).decode(),
            "append": False,
        })
        if not r.get("ok"):
            return f"verify_package failed to stage {fname}: {r.get('error', 'unknown error')}"
    r = await bridge.call({
        "type": "exec",
        "command": "python3 .ple_package.py",
        "timeout_sec": 300,
    })
    if r.get("exit_code") != 0:
        return f"verify_package execution failed:\n{r.get('output', '')}"[:4000]
    try:
        report = json.loads((r.get("output") or "").strip().splitlines()[-1])
    except Exception:
        return f"verify_package returned an unreadable report:\n{r.get('output', '')}"[:4000]
    files = report.get("files", [])
    total = sum(f["size"] for f in files)
    lines = [f"package contents: {len(files)} file(s), {total} bytes total"]
    for f in files[:60]:
        lines.append(f"  {f['path']} ({f['size']} B)")
    if len(files) > 60:
        lines.append(f"  ... and {len(files) - 60} more")
    if report.get("missing"):
        lines.append("MISSING PATHS: " + ", ".join(report["missing"]))
    z = report.get("zip")
    if z:
        lines.append(f"zip written: {z['path']} ({z['size']} B), {len(z['entries'])} entries, "
                     f"integrity: {'CORRUPT at ' + str(z['corrupt_entry']) if z['corrupt_entry'] else 'OK'}")
        state["deliverables"].append(z["path"])
    ok = bool(files) and not report.get("missing") and (not z or not z["corrupt_entry"])
    lines.append("PACKAGE OK" if ok else "PACKAGE INCOMPLETE - fix before finishing")
    return "\n".join(lines)[:TOOL_RESULT_CHAR_CAP]


def mark_cache(messages, model_name):
    """Claude prompt caching needs explicit breakpoints (docs/models.md)."""
    if "claude" not in (model_name or "").lower():
        return messages
    cache = {"type": "ephemeral"}
    out = list(messages)
    for i in {0, len(out) - 1}:
        content = out[i].get("content")
        if isinstance(content, str) and content:
            out[i] = {**out[i], "content": [{"type": "text", "text": content, "cache_control": cache}]}
    return out


async def mock_run(bridge, task):
    """Scripted zero-cost session: proves the protocol loop end to end."""
    await bridge.log("mock run: inventory")
    listing = await bridge.call({"type": "exec", "command": "ls -la", "timeout_sec": 30})
    await bridge.call({
        "type": "write_file",
        "path": "hello.txt",
        "content_b64": base64.b64encode(b"hello ple\n").decode(),
        "append": False,
    })
    summary = (
        "Mock run complete. Instruction received: "
        + task.get("instruction", "")[:200]
        + f" | workdir listing exit_code={listing.get('exit_code')} | wrote hello.txt"
    )
    await bridge.ws.send(json.dumps({"type": "done", "final_message": summary}))


RETRY_WAITS = [5, 15, 45, 120]
WRAP_UP_SEC = 240


def _trim_old_tool_outputs(messages, keep=1000):
    """Shrink older long tool results in place (context-length recovery)."""
    tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
    for i in tool_idx[:-4]:
        content = messages[i].get("content") or ""
        if len(content) > keep + 200:
            messages[i] = {**messages[i], "content": content[:keep] + "\n[trimmed to save context]"}


async def _create_with_retries(bridge, client, model, messages, deadline, tool_choice="auto"):
    """One model call with backoff on transient failures (docs/models.md).

    Partial responses are discarded: nothing is appended to the history here.
    A 400 that smells of context length trims old tool outputs and retries.
    """
    waits = list(RETRY_WAITS)
    trimmed = False
    while True:
        try:
            return await client.chat.completions.create(
                model=model,
                messages=mark_cache(messages, model),
                tools=TOOLS,
                tool_choice=tool_choice,
                max_tokens=32000,
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            text = str(exc).lower()
            if status == 400 and not trimmed and ("context" in text or "too long" in text or "token" in text):
                _trim_old_tool_outputs(messages)
                trimmed = True
                await bridge.log("context-length 400: trimmed old tool outputs, retrying")
                continue
            transient = status is None or status == 429 or (isinstance(status, int) and status >= 500)
            if not transient or not waits:
                raise
            wait = waits.pop(0)
            if time.monotonic() + wait > deadline:
                raise
            await bridge.log(f"transient model error ({status}); retry in {wait}s")
            await asyncio.sleep(wait)


async def agent_run(bridge, task):
    from openai import AsyncOpenAI

    model = os.environ.get("MODEL_NAME", "")
    client = AsyncOpenAI()  # reads OPENAI_BASE_URL / OPENAI_API_KEY
    limits = task.get("limits") or {}
    timeout = limits.get("agent_timeout_sec") or 6 * 3600
    started = time.monotonic()
    deadline = started + timeout * 0.92
    state = {
        "deliverables": [],
        "mcp_servers": task.get("mcp_servers") or [],
        "mcp_sessions": {},  # server URL -> {"session_id", "protocol_version"}
    }

    def remaining_min():
        return max(0, int((started + timeout - time.monotonic()) // 60))

    mcp_note = ""
    if task.get("mcp_servers"):
        mcp_note = "\nMCP servers available via the http tool: " + json.dumps(task["mcp_servers"])

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Task instruction:\n{task.get('instruction', '')}\n\n"
                f"Workdir: {task.get('workdir', '/app')}{mcp_note}\n"
                "Start with the inventory step."
            ),
        },
    ]

    final_text = "No deliverable summary produced."
    nudges = 0
    wrapped = False
    try:
        for _turn in range(300):
            if time.monotonic() > deadline:
                await bridge.log("time budget reached; finishing")
                break
            # Wrap-up phase: near the deadline, force a summary with no tools.
            tool_choice = "auto"
            if not wrapped and (started + timeout - time.monotonic()) < WRAP_UP_SEC:
                wrapped = True
                tool_choice = "none"
                messages.append({
                    "role": "user",
                    "content": "Time is almost up. Do not call any more tools. "
                               "Write your final summary now: what you produced, where the files are, key numbers, assumptions.",
                })
            resp = await _create_with_retries(bridge, client, model, messages, deadline, tool_choice)
            await bridge.usage(model, getattr(resp, "usage", None))
            choice = resp.choices[0]
            msg = choice.message
            # model_dump preserves provider fields (e.g. Gemini thought signatures).
            messages.append(msg.model_dump(exclude_none=True))
            if msg.content:
                final_text = msg.content
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    result = await run_tool(bridge, tc.function.name, args, state)
                    result += f"\n[time remaining: ~{remaining_min()} min]"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                continue
            # No tool calls: finished - unless the reply was cut off or empty.
            if (choice.finish_reason == "length" or not (msg.content or "").strip()) and nudges < 3 and not wrapped:
                nudges += 1
                messages.append({
                    "role": "user",
                    "content": "Your last reply was empty or cut off. Continue exactly where you stopped.",
                })
                continue
            break
    except Cancelled:
        pass

    written = ", ".join(state["deliverables"]) or "none recorded"
    summary = f"{final_text}\n\nDeliverable files written via write_file: {written}"
    await bridge.ws.send(json.dumps({"type": "done", "final_message": summary[:15000]}))


async def session(ws):
    bridge = Bridge(ws)
    try:
        task = json.loads(await ws.recv())
        if os.environ.get("HARNESS_MOCK") == "1":
            await mock_run(bridge, task)
        else:
            await agent_run(bridge, task)
    except Cancelled:
        try:
            await ws.send(json.dumps({"type": "done", "final_message": "Cancelled by evaluator; finishing with partial state."}))
        except Exception:
            pass
    except Exception as exc:  # never die silently: the grader needs a frame
        try:
            await ws.send(json.dumps({"type": "error", "message": f"harness error: {exc}"[:2000]}))
        except Exception:
            pass


async def main():
    port = int(os.environ["PORT"])
    async with websockets.serve(session, "0.0.0.0", port, max_size=MAX_FRAME, ping_interval=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
