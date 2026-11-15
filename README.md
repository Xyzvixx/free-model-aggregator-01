# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Xyzvixx/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Xyzvixx/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)


One server, two APIs, ten free-tier providers. It serves the OpenAI Chat
Completions API (`POST /v1/chat/completions`) and the Anthropic Messages API
(`POST /v1/messages`), in JSON and in server-sent events, routing every
request to the best free model that can serve it and failing over when a
provider refuses.

## How it works

- **Catalog.** At startup it reads the brief (`$MAKEFUN_PROMPT_FILE`: the
  provider directory, the quality table, the workload manifest), probes each
  provider's catalog route in that provider's own wire (OpenAI, Anthropic or
  Gemini) and auth scheme, and builds one catalog document.
- **Routing.** The manifest is known up front, so the server plans the whole
  workload first: weight-2 and scarce requests claim their best model before
  plain ones, respecting each provider's per-trial budget (soft caps from the
  sealed budget ranges in the kit). At serve time the planned model goes
  first; behind it a dynamic ranking (rating, minus a spread penalty per call
  already made to that provider and a failure penalty) orders the failover
  candidates, tried provider round-robin so no provider is tried twice
  before every provider has had a turn.
- **Wires.** Requests are normalised once, then rebuilt in the provider's
  wire — OpenAI, Anthropic (with `anthropic-version`), or Gemini
  (`generateContent` / `streamGenerateContent`, `responseMimeType` for JSON
  mode) — preserving the system prompt, user text, images, tools, stop
  sequences, `max_tokens` and JSON mode exactly. Answers are parsed back
  into one shape and emitted in the wire the client used, with the
  provider's text whole, its usage counts, and its stop reason mapped.
  A JSON-mode request from an OpenAI client never goes to an
  Anthropic-wire provider (that wire cannot carry JSON mode).
- **Failover.** A 402, 429 or 5xx (or an unreachable/unparsable provider) is
  never the answer: the next candidate serves the request instead.
- **Determinism.** No wall clock, no unseeded randomness. Every choice is a
  function of the brief, the requests and the provider answers so far, so a
  replay reproduces the run byte for byte.

## Run

```sh
node main.ts --list   # print the catalog document and exit 0
node main.ts          # register with $MAKEFUN_ENV_URL, serve, exit 0 when done
```

Node 22, standard library only — nothing to install. The server listens on
an ephemeral port on `0.0.0.0`, registers it with
`POST $MAKEFUN_ENV_URL/register`, and polls `GET $MAKEFUN_ENV_URL/work`
until the workload is done.

## Results (practice trials, `grade-local --driver process`)

| Trial | Score | Quality | Served | Wire | Failover | Catalog | Calls | Replay |
|-------|-------|---------|--------|------|----------|---------|-------|--------|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 25 | matched |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 26 | matched |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 32 | matched |

20/20 requests served validly in every trial. Quality 99 is the practical
ceiling: the best model for 17 of the 20 requests is one 1475-rated model
whose per-trial budget caps at 12 calls.

## Files

- `main.ts` — the server: brief, catalog, routing, failover, register/poll
- `lib/providers.ts` — brief parsing, auth headers, catalog probe
- `lib/wire.ts` — client↔provider translations and SSE emitters
- `LICENSE` — MIT

## License

MIT — see `LICENSE`.
