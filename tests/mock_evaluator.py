#!/usr/bin/env python3
"""Fake evaluator: proves the harness protocol loop end to end at zero cost.

Starts serve.py with HARNESS_MOCK=1, connects as the evaluator would, sends
one task, executes the harness's exec/read_file/write_file requests inside a
temp workdir, and asserts the session ends with `done` and hello.txt written.

Run:  python tests/mock_evaluator.py
Exit 0 and "MOCK SMOKE PASS" on success.
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import tempfile

import websockets

PORT = 18741


async def run():
    workdir = tempfile.mkdtemp(prefix="ple-mock-")
    env = {**os.environ, "PORT": str(PORT), "HARNESS_MOCK": "1"}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, os.path.join(root, "serve.py")],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        # wait for the port
        for _ in range(50):
            try:
                ws = await websockets.connect(f"ws://127.0.0.1:{PORT}", max_size=16 * 2**20, ping_interval=None)
                break
            except Exception:
                await asyncio.sleep(0.2)
        else:
            raise RuntimeError("server did not start")

        async with ws:
            await ws.send(json.dumps({
                "type": "task", "protocol": 1, "session": "mock1",
                "instruction": "Write hello.txt containing 'hello ple' and summarize.",
                "workdir": workdir,
                "limits": {"agent_timeout_sec": 120, "exec_timeout_sec": 60,
                           "max_output_bytes": 100000, "max_write_bytes": 10000000},
                "mcp_servers": [],
            }))
            done_msg = None
            while True:
                frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
                t = frame.get("type")
                if t in ("log", "usage"):
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
                    mode = "ab" if frame.get("append") else "wb"
                    with open(path, mode) as f:
                        f.write(data)
                    await ws.send(json.dumps({"type": "result", "id": frame["id"], "ok": True, "size": len(data)}))
                elif t == "read_file":
                    path = os.path.join(workdir, frame["path"])
                    try:
                        with open(path, "rb") as f:
                            data = f.read()
                        await ws.send(json.dumps({
                            "type": "result", "id": frame["id"], "ok": True,
                            "content_b64": base64.b64encode(data).decode(),
                            "size": len(data), "truncated": False,
                        }))
                    except Exception as exc:
                        await ws.send(json.dumps({"type": "result", "id": frame["id"], "ok": False, "error": str(exc)}))
                elif t == "http":
                    await ws.send(json.dumps({"type": "result", "id": frame["id"], "ok": False, "error": "no MCP in mock"}))
                else:
                    raise RuntimeError(f"unexpected frame: {frame}")

            hello = os.path.join(workdir, "hello.txt")
            assert done_msg is not None, "no done frame"
            assert os.path.exists(hello), "hello.txt not written"
            content = open(hello).read()
            assert content == "hello ple\n", f"unexpected hello.txt content: {content!r}"
            assert "Mock run complete" in done_msg.get("final_message", ""), "final_message missing mock summary"
            print(f"MOCK SMOKE PASS - workdir={workdir}")
            print(f"final_message: {done_msg['final_message'][:300]}")
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
            if proc.returncode not in (0, -15):
                print("server output:", out)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    asyncio.run(run())
