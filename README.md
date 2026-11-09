# Idris PLE Harness

Autonomous agent harness for **Google - Agent Harness on Physical-Engineering's
Last Exam** (Kaggle). It solves real structural / civil / electrical /
mechanical engineering tasks inside the PLE evaluation containers.

## Architecture (v0 prototype)

```
evaluator --WebSocket (protocol v1)--> serve.py
                                          |  Bridge: exec / read_file /
                                          |  write_file / http (MCP)
                                          v
                                   task container
                                          ^
                                          |  OpenAI Chat Completions
                                   sponsored model (env-injected)
```

- `serve.py` - the harness server. One WebSocket connection per task; the
  agent loop is: inventory -> extract deliverables -> plan -> compute with
  tools -> verify against the instruction -> `done` with a grader summary.
- The model is **never** hard-coded: `MODEL_NAME`, `OPENAI_BASE_URL` and
  `OPENAI_API_KEY` come from the evaluator environment, so the same code runs
  on any of the 8 `google-cloud/*` models. `submission.yaml` currently commits
  `google-cloud/claude-opus-5-5` (see the project's MODELES.md for the scored
  rationale); Claude cache breakpoints are applied automatically.
- Token usage is reported after every model call (`usage` frames) - cost is a
  scored dimension of the competition, so it is measured from the first run.
- Time budget: the loop stops at 92% of `limits.agent_timeout_sec` and always
  finishes with `done` (or answers a `cancel` with the best summary so far).

## Local checks (no Docker, no model cost)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python tests/mock_evaluator.py   # fake evaluator, HARNESS_MOCK=1
.venv/bin/python tests/agent_loop_test.py  # fake OpenAI model + fake evaluator
```

Expected: `MOCK SMOKE PASS` and `AGENT LOOP PASS` (the second test drives the
real agent loop - tool calls, deliverable write, usage reporting, final
summary - against a scripted model server, still at zero cost).

## Official validation (needs Docker + the PLE kit)

```bash
uv run python -m kit check ../harness      # from a google-research/ple clone
uv run python eval/run.py ../harness       # dev set -> artifacts/
uv run python -m kit artifacts ../harness
```

Local runs on sponsored models go through `kit/model_proxy` billed to our own
Google Cloud project; local runs on our own provider keys use the kit's
`anthropic/` / `openai/` / `gemini/` model names. Both are real money - see the
project cadrage before spending.

## License

MIT (see LICENSE). The winning-entry license required by the competition is
Apache-2.0 or MIT for harness code, CC BY 4.0 for the writeup.
