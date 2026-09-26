# LLM Council

An orchestrator that puts one question to several independent local LLMs, then
has a "head" model synthesise their answers. Talks to `llama-server` instances
over HTTP.

## Design

- **The orchestrator does not manage processes.** Servers are started and
  stopped outside it; the orchestrator is a pure HTTP client pointed at
  endpoints. This keeps model lifecycle (systemd, a swap proxy, a container,
  or a terminal window) independent of council logic, and lets members live on
  different machines.
- **Independent members.** Members never see each other's answers. Correlated
  answers would defeat the point of aggregating them.
- **One model per lab.** Council quality comes from independent failure modes,
  not member count. Members are drawn from different labs with different
  training lineages.
- **Head aggregates.** A single, more capable model reads all member answers and
  produces the final response.

## Layout

```
config.json          endpoints and settings
main.py              command-line front end
council/
  health.py          endpoint reachability and readiness checks
  client.py          OpenAI-compatible /v1/chat/completions client
  head.py            aggregation prompt and synthesis step
  cache.py           per-member answer cache
  sampling.py        per-endpoint temperature, max_tokens and seed
  chat.py            interactive terminal interface for debugging
  prompts.py         system prompts, roles, and context assembly
  answer.py          extract the machine-readable result from the head
  jobs.py            single-slot job queue for the web API
api.py               HTTP API in front of run_council()
web/
  council.js         Express router: chat page + proxy to api.py
  INTEGRATION.md     how to wire it into an existing server.js
```

No third-party dependencies; standard library only.

## Running the servers

Each member needs a reachable OpenAI-compatible endpoint. Two topologies work:

**One server per member, each on its own port.** Simplest, and every model
stays resident. Needs enough RAM for all of them at once.

```powershell
llama.exe serve -hf Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M `
  --host 0.0.0.0 --port 8090 -c 8192 --parallel 1
llama.exe serve -hf <other-model> --host 0.0.0.0 --port 8091 -c 8192 --parallel 1
```

`--host 0.0.0.0` is required when the orchestrator runs on a different machine
from the server; the default binds to loopback only. `--parallel 1` gives a
single request the whole KV cache instead of splitting it across the four slots
the server allocates by default.

Note that binding to `0.0.0.0` exposes the endpoint to the whole network, and
the server runs with CORS open and no API key. Restrict it at the firewall.

### OpenVINO: prompts over 512 tokens

On the OpenVINO backend, any prompt longer than the default micro-batch of 512
tokens fails with:

```
GGML OpenVINO backend ov::Exception: ...
While validating node 'opset1::Add ...': Argument shapes are inconsistent.
```

returned to the client as HTTP 500 "Compute error". Prompts under 512 succeed;
the boundary is exact. The backend mis-shapes a node when a prompt is split
across micro-batches. Raise the micro-batch above the longest prompt you
expect:

```
-b 2048 -ub 2048
```

Note the server logs `n_batch = 2048` by default while `n_ubatch` is still 512,
so the log does not make the cause obvious. This bites once prompts grow —
a longer system prompt or any real context will cross 512 — so apply it to
every server, not only the head.

**One swap proxy for all members.** A proxy such as `llama-swap` exposes a
single endpoint and loads models on demand, so only one is resident at a time.
Point every member at the same `base_url` and set each member's `model` field
to the name the proxy routes by.

## Use as a library

Orchestration lives in `council/orchestrator.py`, not in `main.py`, so a parent
process imports and calls it directly rather than shelling out and parsing a
file:

```python
from council import load_config, run_council, succeeded

config = load_config("config.json")
result = run_council(
    "Solve for x: 3x + 7 = 25",
    config,
    context="Earlier in this series: 2x + 3 = 11 -> x = 4",
)

if succeeded(result):
    print(result["head"]["answer"])
```

`run_council` returns the same structure the CLI writes to `answers.json`:

```python
{
  "question": str,
  "members": [{"name", "lab", "seconds", "ok", "answer" | "error", "cached"?}],
  "head":    {"name", "lab", "seconds", "ok", "answer" | "error"}   # optional
}
```

It does not raise when a model fails — check the `ok` flags, or call
`succeeded(result)`. It raises `ConfigError` for an unusable config and
`EndpointsUnavailable` when preflight finds a server down.

Keyword arguments: `context`, `use_head`, `use_cache`, `retries`, `preflight`.

See `example_parent.py` for a complete caller.

## Command-line usage

```bash
python main.py --chat                       # interactive debugging session
python main.py "question" --context "..."   # context sent to every member
python main.py "question" --context-file notes.txt
python main.py --check                      # verify endpoints, ask nothing
python main.py "your question here"
python main.py "your question" --no-head    # member answers only
python main.py "your question" --no-cache   # ignore and do not write cache
python main.py --clear-cache                # drop every cached answer
python main.py "your question" --out answers.json --verbose
```

`--check` is the fast preflight: it reports which endpoints are up without
spending inference time. A full run checks every endpoint first and aborts
before asking anything if one is down, so a missing model is caught in seconds
rather than after several members have already answered.

Member answers are written to `answers.json`. A member whose request fails is
recorded with its error rather than aborting the run; the exit code is non-zero
if any member failed.

`main.py` is a thin wrapper over the same function, so both paths behave
identically.

## Chat mode

`--chat` opens a terminal loop for poking at the council by hand. Each question
is an independent council run — members are stateless and never see prior
turns. That is deliberate: it mirrors the nightly job exactly, so what you see
in chat is what the batch job would produce. It is a debugging tool, not a
conversation.

Member answers are truncated in the transcript; commands show the detail:

| Command | Effect |
| --- | --- |
| `/members` | Last run's member answers in full |
| `/head` | Last run's final answer in full |
| `/json` | Last run as raw JSON |
| `/config` | Current endpoints and sampling settings |
| `/nohead` | Toggle the aggregation step |
| `/nocache` | Toggle use of the answer cache |
| `/context` | Show the current context |
| `/setcontext <text>` | Set the context sent with every question |
| `/nocontext` | Clear the context |
| `/clear` | Drop every cached answer |
| `/exit` | Leave (Ctrl-D and Ctrl-C also work) |

Endpoints are checked once on the first question rather than before every turn.
A failed check re-arms it, so recovering a downed server does not need a
restart.

Toggling the cache off and asking the same question again is the quickest way
to see how much a member's answer varies between runs.

## Prompts, roles and context

Three layers, each optional:

**System prompt.** `head_preset` selects a built-in head prompt (`default` or
`math`). `member_system_prompt` and `head_system_prompt` supply text directly
and win over the preset. An individual endpoint's `system_prompt` overrides
even that.

**Role.** An endpoint's `role` is appended to its system prompt. This is the
only per-member framing, and it is static config, never assigned at runtime.

**Context.** Passed per question to `run_council(question, config,
context=...)`, prepended to the question for every member and for the head. Use
it for prior answers, constraints, or working notes.

### Why the head does not assign roles or context

It was considered and rejected. Members answering independently is the property
the whole design rests on: if the head framed each member's task, one framing
choice would propagate into every answer, and correlated errors are exactly
what a council exists to avoid. It would also serialise the run, since no
member could start until the head had spoken.

Static roles in config are a different thing and are supported — but see the
caution below before using them for mathematics.

### Why the default member prompt does not mention the panel

Earlier versions told each member it was one of several experts. That tends to
make a model hedge, defer, or note that others may catch its errors — useful in
a human panel, useless here, since members cannot see each other. The default
now simply asks for the model's best independent answer and explicitly
discourages padding. Set `member_system_prompt` to test the alternative.

### Context is shared, which cuts both ways

Context is data rather than framing, so it does not correlate members the way a
per-member role would. But every member does see it: a wrong premise in the
context misleads all of them at once, and the head cannot detect it because no
member dissents. When context carries prior answers, a mistake that got through
once can be reinforced rather than caught. Prefer stating prior answers as
claims to check rather than as established facts.

## Web API

`api.py` puts an HTTP API in front of `run_council()` so a web page can use the
council. Start it alongside the model servers:

```bash
python api.py --config config.json --host 127.0.0.1 --port 8099 --check
```

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/ask` | `{"question": str, "context": str?}` -> 202 with a job |
| GET | `/job/<id>` | Job state, and the result once finished |
| GET | `/jobs` | The last few jobs |
| GET | `/health` | Liveness, and whether a job is running |
| GET | `/endpoints` | Member and head names, for the page header |

While a job runs, `progress` carries a per-seat state map — each member is
`waiting`, `asking`, `answered`, `cached` or `failed`, plus the head's own
state and the current phase. `run_council` takes an `on_event` callback that
drives this; anything the callback raises is swallowed, so a progress display
can never fail a run.

### Polling, not a held request

A council run takes minutes with real models — longer than a browser, a proxy
or a load balancer will hold a request open. `/ask` therefore returns a job id
immediately and the client polls `/job/<id>` until `state` is `done` or
`failed`. A dropped connection loses nothing; the job keeps running and the id
still resolves.

### One job at a time

A second `/ask` while a job is running returns **409** with `busy: true`. The
model servers run `--parallel 1` and the council asks sequentially, so
concurrent runs would queue inside llama-server regardless — and each costs
minutes of CPU. Refusing plainly beats silently doubling everyone's wait.

A run that fails records the error on the job and frees the slot, so one bad
run cannot wedge the queue.

### Bind to localhost

The API has no authentication of its own. It binds to `127.0.0.1` by default
and is intended to sit behind a front end that already authenticates — the
Express app reverse-proxies to it, so login and CSRF stay in one place. Binding
to `0.0.0.0` without auth in front would expose an endpoint where one request
costs minutes of CPU; the server logs a warning if you do.

### The round table

The page is two columns: the conversation on the left (two thirds) and the
council table on the right (one third). The table panel is sticky, so it stays
in view while a long answer scrolls — the point of a live diagram is lost if
you have to scroll back to it.

There is one table for the page rather than one per turn: it shows the current
run, or the last one once finished. Per-turn tables would push the
conversation down and leave a row of stale diagrams behind.

A seat per member sits around the head, coloured by state — waiting, answering
(pulsing), answered, cached, failed — with a legend under it. Members report
inward along animated lines only while the head is actually aggregating.

Every seat's colour comes from the run's real progress, reported by the API.
Nothing is on a timer: an animation that did not track the run would be
decoration pretending to be information.

The seats also say what their members said. A `member_done` progress event
carries the member's answer, so the moment a member finishes its seat grows a
speech bubble with the opening of its answer (the full text on hover), the
"voices" list under the table gets the line, and the conversation on the left
gets that member's full answer - all while the other members are still
thinking and before the head has said a word. With `ask_members: "parallel"`
several seats pulse at once and the bubbles appear in whatever order the
models finish.

### Streaming, and a compact conversation

Replies are requested with `stream: true` and assembled from the server-sent
events (`client._post_stream`), and the text-so-far is reported through the
same progress channel the seats are coloured from - at most four times a
second per reply, since every report copies the whole progress state - so the
page shows each member's answer being typed, with a cursor, in the
conversation, in the voices list and in its seat's bubble (which shows the
tail of the text while it grows). The head's verdict streams the same way. No
new transport was needed: the page already polls the job, and it simply polls
every second instead of every two while replies are in flight. The timeout in
streaming mode is the longest a server may go without sending anything, which
is the right shape for slow CPU inference: a long reply is fine, a stall is
not.

Answers are markdown, and the page renders them as such - headings, lists
(nested by indentation), bold and italics, inline and fenced code, block
quotes, rules, pipe tables and http(s) links - by building DOM nodes and never
through `innerHTML`, because the text is model output and therefore
untrusted. The renderer re-runs on every streamed chunk, so a construct that
is still half-written renders as far as it goes and settles when its closing
marks arrive. Excerpts (the member rows, the voices list, the bubbles) have
the markdown syntax stripped so a summary does not read as `### **Result**`.

When streaming looks as if it is not working for one endpoint, the council log
says which of two very different things is happening. `<name>: first token
after 42.0s` is a server that is merely slow to start - a big model reading a
long prompt on a CPU can take minutes before its first token, and streams
normally after that; the page shows the head's box with "reading the members'
answers" during that wait. `<name>: a 1800-character reply arrived in 1
piece(s)` is a server that generated everything first and sent it at once - a
buffering proxy in front of it, or a backend that does not really stream -
and no client-side change can help that.

The conversation is kept compact: a member's answer is one row - who, and the
opening words - that opens on click, and the rows fold away by themselves the
moment the head has reported, so what remains in view per turn is the
question and the verdict. A reply that is still streaming stays open so it
can be watched; sessions loaded from history show their members folded.

### Sessions

Conversations are kept per account, on the server, in the gitignored
`config/council-sessions/` folder (one directory per account, one file per
session): reloading the page, or coming back tomorrow, shows them again, and
one account's sessions are unreachable from another's by construction. A
session exists from its first question onward - there is no empty session to
create, and a session with no messages is never written nor listed. The turn
is recorded when the question is submitted and completed by the Node server's
own poller when api.py finishes, so closing the tab does not lose the answer,
and a restarted server resumes polling for whatever was still pending.

Below 900px the layout stacks and the table moves above the conversation, so
it is still visible without scrolling past everything. Animations respect
`prefers-reduced-motion`.

## Aggregation

The head receives every successful member answer in one prompt and produces the
final response. Two deliberate choices:

- **Members are anonymised.** The head sees "Member 1", "Member 2" and so on,
  never model names, so it cannot defer to a brand rather than to an argument.
- **The head is told not to average.** Its prompt instructs it to judge claims
  individually, to treat agreement as weak evidence rather than proof, to
  surface contradictions explicitly, and to drop claims it judges wrong even
  when several members make them. Length is explicitly called out as not being
  a proxy for reliability — in prototype testing the shortest member answer was
  the only one with no false claims in it.

The head runs at a lower temperature than the members. Members benefit from
some variety; the head should be as steady as possible.

If no member answers, the head is skipped. If the head fails, the member
answers are still written out.

## Resilience

A nightly run can take hours, and a member that fails late is expensive to
redo. Two mechanisms cover that.

**Retry.** Transient failures — a dropped connection, a timeout, an HTTP 5xx or
429 — are retried with linear backoff (`retry_delay_s` x attempt number). A 4xx
response is treated as permanent and raised immediately, since repeating a
malformed request only wastes time.

**Cache.** Every successful member answer is written to `cache_dir`, keyed by a
hash of the question, the member's name, its endpoint, its model, and the
system prompt version. Rerunning the same question reuses those answers and
asks only the members that failed. Failures are never cached, so a rerun always
retries them.

The key includes the resolved system prompt and the endpoint's sampling
settings, so changing a role, a prompt, a seed or a temperature invalidates the
affected entries automatically and leaves the rest alone. `PROMPT_VERSION` in
`orchestrator.py` covers changes to the built-in defaults.

The head is not cached: it is one call, and its whole purpose is to see the
current set of member answers.

When a member fails, the run exits non-zero and logs a reminder that a rerun
will retry only what failed. That makes the recovery path a plain rerun of the
same command rather than anything special.

## Sampling and reproducibility

`temperature`, `max_tokens` and `seed` are set per endpoint, falling back to
role defaults (members 0.7 / 2048, head 0.3 / 4096, seed unset for both).

`seed` is omitted from the request when null, leaving the server to pick one.
Set it to make a run reproducible. This matters when tuning the head prompt:
two head runs over *identical cached member answers* produced noticeably
different output in testing — one clean, one repeating a member's fabrication —
so without a fixed seed you cannot tell whether a prompt change helped or you
simply drew a different sample.

Reproducibility holds only while the prompt, model, quant, server flags and
backend are unchanged. Moving a model between CPU and GPU, or changing
`--parallel`, can change the output for the same seed.

A seed does not reduce variance, it only fixes which sample you get. If the
head is unsteady, lowering its temperature addresses the cause; a seed only
makes the symptom repeatable. The prototype head is set to `temperature: 0.0`
(greedy decoding) for that reason.

Useful combinations:

| Members | Head | Use |
| --- | --- | --- |
| seed null | seed set | Natural member variety, reproducible aggregation — best for tuning the head prompt |
| seed set | seed set | Fully reproducible pipeline — regression testing |
| seed null | seed null | Fresh sample every run — normal nightly operation |

## Configuration

| Key | Meaning |
| --- | --- |
| `request_timeout_s` | Per-question timeout; generous, CPU inference is slow |
| `health_timeout_s` | Per-endpoint preflight timeout |
| `retries` | Extra attempts per request after the first |
| `retry_delay_s` | Base backoff; attempt N waits N x this |
| `cache_dir` | Where member answers are cached; omit or `null` to disable |
| `stream_answers` | Stream member and head replies token by token so the page shows them being typed (default true). Set false for a server that does not speak server-sent events; the run then reports each reply when it is complete |
| `ask_members` | `sequential` (default): one member after another. `parallel`: every member at the same time, one thread each - each member is its own llama-server, so this shares the model box between them rather than queueing inside one server. Anything else falls back to sequential with a warning |
| `shuffle_members` | Randomise the order answers reach the head (default true) |
| `shuffle_seed` | Fix the shuffle for a reproducible run; null means random |
| `head_preset` | Built-in head prompt: `default` (weigh opinions, prose reply), `math` (verify each result by substitution, `ANSWER: <result>`), `research` (agreed / disputed / evidence / verdict with confidence and the strongest counter-argument, `ANSWER: <verdict>`), `decision` (options, trade-offs, one recommendation with the reason the runner-up lost, `ANSWER: <option>`). `head_system_prompt` replaces any of them with your own text. The head result carries `expects_value` (true when the head's prompt asks for the marker), and the page and CLI only report a missing `ANSWER:` line when it was asked for - the `default` preset answers in prose |
| `member_system_prompt` | Replaces the default member system prompt; null uses the default |
| `head_system_prompt` | Replaces the default head system prompt; null uses the default |
| `members[]` | `name`, `lab`, `base_url`, `model`, `enabled`, plus optional `temperature`, `max_tokens`, `seed`, `role`, `system_prompt` |
| `head` | Same fields as a member; different sampling defaults |

`model` may be `null` for a plain `llama-server`, which serves whatever it was
started with. Set it when the endpoint routes by model name.

## Status

Steps 1-4 of 5 complete; verified end-to-end against `llama-server` instances
on a separate host.

- [x] **1. Skeleton + one working member** — config, endpoint health checks,
      query client, preflight
- [x] **2. Add members one at a time** — three prototype members running
      (Qwen/Alibaba, Llama/Meta, Phi/Microsoft); production members will be
      Qwen3.6, Mistral, GLM, Llama
- [x] **3. Add the head** — aggregation prompt and synthesis step; production
      head will be Gemma 4
- [x] **4. Persistence and resilience** — per-member answer cache, retry with
      backoff, permanent vs transient error handling
- [x] **5a. Chat mode** — interactive terminal interface for debugging
- [x] **5b. Web API** — job queue, polling endpoints, localhost-bound
- [x] **5c. Web page** — Express router, chat UI, polling front end
- [ ] **5d. Scheduling** — nightly trigger, output handling

## Prototype topology

The orchestrator runs in a Linux VM; the `llama-server` instances run on the
Windows host so they can reach the Intel iGPU through the OpenVINO backend.
Set the execution device before starting a server:

```powershell
$env:GGML_OPENVINO_DEVICE = "GPU"   # CPU | GPU | NPU
$env:LLAMA_CACHE = "C:\\llama-cache"
$env:GGML_OPENVINO_CACHE_DIR = "C:\\llama-cache\\ov_cache"
```

`GGML_OPENVINO_CACHE_DIR` cuts model compile time on repeat starts, and is not
supported on NPU. All three are read at process start.

## Prototype members

Small stand-ins used to exercise the plumbing. Chosen for lab diversity, not
quality — a 1B model is not a useful council member.

| Member | Lab | Repo (`-hf`) | Port |
| --- | --- | --- | --- |
| qwen2.5-1.5b | Alibaba | `Qwen/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M` | 8090 |
| llama-3.2-1b | Meta | `unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M` | 8091 |
| phi-4-mini | Microsoft | `bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M` | 8092 |
| head-llama-3.2-3b | Meta | `unsloth/Llama-3.2-3B-Instruct-GGUF:Q4_K_M` | 8093 |

The prototype head shares a lab with one member, which would be a poor choice
in production but is irrelevant for testing the plumbing.

Not every model works on every backend. Phi-4-mini fails on the OpenVINO GPU
path (`cannot determine dynamic dim for CONT node`) and must run with
`GGML_OPENVINO_DEVICE=CPU`. Mixed devices across members are fine; the
orchestrator neither knows nor cares which backend a member uses.

All prototype servers need `-b 2048 -ub 2048` — see the OpenVINO note above.

The OpenVINO backend has been the source of every model-level failure hit so
far: an SSM reshape on Qwen3.5, an unsupported GatedDeltaNet op on the NPU,
Phi-4-mini's attention layout on GPU, and the micro-batch bug above. None of it
will be in the production path, which is CPU-only. The standard CPU build is a
closer match to the target machine and has broader model coverage.

## Mathematical questions

Set `head_preset` to `"math"`. It changes aggregation from weighing opinions to
checking results, and adds a machine-readable answer line.

### Verification instead of voting

The default head prompt says agreement is weak evidence, which correctly stops
the head counting votes but gives it nothing to reason with instead. In testing
on `1 + 1 = x`, two members answered x = 2 and a third declined on a pedantic
reading; the head adopted the refusal and discarded both correct answers.

The maths preset instead tells the head to solve the problem itself first, then
substitute each candidate result back into the original problem and keep only
what holds. One member with a result that verifies beats three without one. It
also tells the head to ignore members that declined rather than treat them as
dissent, and to answer the reasonable reading of an ill-posed question rather
than objecting to it.

### Position bias, and why members are shuffled

The same test exposed a second problem: the answer the head adopted was the
last one in its prompt. Heads favour recent text.

`shuffle_members` (on by default) randomises the order answers are presented to
the head on each run. This does not remove the bias — nothing in a prompt
reliably does — but it stops the bias landing on the same model every night, so
one member is not systematically privileged. Set `shuffle_seed` for a
reproducible order, or `shuffle_members: false` to disable. The order in
`result["members"]` always follows config, so the caller is unaffected.

### The answer contract

The maths preset ends the head's reply with a line of the form

```
ANSWER: 42
```

which is parsed into `result["head"]["value"]`:

```python
value = result["head"]["value"]      # "42", "UNDETERMINED", or None
```

`None` means the head produced no marker at all — treat that as a failed run
rather than falling back to a regex over the prose, which is the failure mode
the contract exists to prevent. `is_undetermined(value)` covers the case where
the head says so explicitly.

The last marker in the reply wins, so a head that restates the format while
explaining itself does not have its example picked up instead of its answer.

### Roles probably do not help here

Framing like "you are a careful algebraist" reliably changes tone and does not
reliably change accuracy; on arithmetic it sometimes hurts. If you use roles,
A/B them with a fixed seed and compare, rather than assuming.

## Target deployment

36-core CPU server, 148 GB DDR3. Generation there is memory-bandwidth-bound, so
MoE models (few active parameters per token) are strongly preferred over dense
models of the same nominal size. Expect single-digit tokens/second for large
dense models.

Whether all members fit in memory simultaneously determines which topology
above to use. If they do not, a swap proxy is the answer.

The laptop is a prototyping environment only; small models are used there to
exercise the plumbing, not to judge answer quality.