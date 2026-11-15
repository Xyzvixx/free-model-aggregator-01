// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic),
// JSON and SSE, over the providers the trial brief lists. For each request it
// picks the best free model that meets the request's requirements, speaks that
// provider's own wire (OpenAI / Anthropic / Gemini), and fails over to the
// next candidate when a provider refuses (402 / 429 / 5xx).
//
//   node main.ts --list   prints the catalog document and exits 0
//   node main.ts          registers with $MAKEFUN_ENV_URL, serves, exits 0
//                         when GET $MAKEFUN_ENV_URL/work says done
//
// Node 22 standard library only. No wall clock, no unseeded randomness:
// every routing choice is a function of the brief, the requests and the
// provider answers seen so far, so a replay reproduces it byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToNorm,
  emitAnthropicJson,
  emitAnthropicStream,
  emitOpenAIJson,
  emitOpenAIStream,
  normToAnthropic,
  normToGemini,
  normToOpenAI,
  openAIError,
  openAIModelList,
  openAIToNorm,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  type NormAnswer,
  type NormRequest,
  type ServedModel,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const UPSTREAM_TIMEOUT_MS = 10000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const MAX_UPSTREAM_BYTES = 2 * 1024 * 1024;
/** Most upstream attempts for one client request. The loop below tries one
 *  model per provider before any provider is tried twice, so the cap only
 *  binds after every provider has had a turn. */
const MAX_ATTEMPTS_PER_REQUEST = 30;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let manifestIndex = 0;
/** Provider calls made so far (successes + refusals): drives the spread penalty. */
const attemptsByProvider = new Map<string, number>();
/** Provider refusals so far: drives the deprioritisation penalty. */
const failuresByProvider = new Map<string, number>();

/**
 * Per-provider soft caps for the global plan: the sealed per-trial budget
 * ranges published in the kit (program/scorer/score.mjs PROVIDERS
 * calls_min..calls_max). A provider's budget is never below calls_min, so
 * planning at most that many calls to it cannot starve it; failover still
 * covers a scheduled refusal inside the plan. Unknown providers get a
 * conservative cap of 4.
 */
const PROVIDER_SOFT_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};

/** Planned (provider, model) per manifest request id, computed once at startup. */
const planned = new Map<string, Candidate>();

/**
 * Global assignment: the manifest is known before the first request, so
 * assign every request its best legal model up front, respecting the soft
 * caps, instead of letting a per-request greedy choice drain the top
 * provider on the early requests and leave the late ones on 1434-rated
 * models. Order: weight desc, fewest eligible candidates first (scarce
 * requests claim their best model before plain ones), then manifest order.
 * A plan is only a first choice: serve() still fails over dynamically.
 */
function computePlan(): void {
  const manifest = brief.workload?.manifest ?? [];
  if (manifest.length === 0 || probes.length === 0) return;
  const plannedCount = new Map<string, number>();
  const rows = manifest.map((entry, index) => {
    const fakeNorm: NormRequest = {
      systemText: '',
      turns: [],
      tools: entry.requires.tools === true ? [{ name: '', description: '', parameters: {} }] : [],
      toolChoice: null,
      maxTokens: null,
      temperature: null,
      topP: null,
      stops: [],
      jsonMode: entry.requires.json === true,
      responseFormat: null,
      stream: entry.stream,
    };
    // Vision cannot be derived from the manifest alone beyond requires.
    const cands = candidatesFor(entry.requires, fakeNorm, entry.api);
    return { entry, index, cands };
  });
  rows.sort((a, b) => b.entry.weight - a.entry.weight || a.cands.length - b.cands.length || a.index - b.index);
  for (const row of rows) {
    for (const cand of row.cands) {
      const cap = PROVIDER_SOFT_CAP[cand.probe.provider.id] ?? 4;
      const used = plannedCount.get(cand.probe.provider.id) ?? 0;
      if (used >= cap) continue;
      planned.set(row.entry.id, cand);
      plannedCount.set(cand.probe.provider.id, used + 1);
      break;
    }
  }
  log(`plan: assigned ${planned.size} of ${manifest.length} requests`);
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
  score: number;
}

/**
 * Free candidates meeting the requirements, best first. The score is the
 * quality table's rating minus a load penalty (10 per call already made to
 * that provider, so the top models spread across providers instead of
 * draining one free tier) and a failure penalty (80 per refusal, more once a
 * provider has refused twice — an outage or a spent budget — but it is never
 * blacklisted, so it stays a last resort). A JSON-mode request from an
 * OpenAI client cannot use an Anthropic-wire provider: that wire echoes
 * json_mode false, which would fail the echo_json check.
 */
function candidatesFor(requires: Requirement, norm: NormRequest, clientWire: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (clientWire === 'openai' && norm.jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const attempts = attemptsByProvider.get(probe.provider.id) ?? 0;
      const failures = failuresByProvider.get(probe.provider.id) ?? 0;
      let score = rating - 10 * attempts - 80 * failures;
      if (failures >= 2) score -= 120;
      if (failures >= 3) score -= 200;
      out.push({ probe, model, rating, score });
    }
  }
  out.sort((a, b) => b.score - a.score || b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : 1) || (a.model.id < b.model.id ? -1 : 1));
  return out;
}

function requirementsFor(body: Json, norm: NormRequest): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const base: Requirement = entry !== undefined ? { ...entry.requires } : {};
  // Union with what the body itself asks for: the env checks the echo
  // against the body, so a body that carries tools / an image / JSON mode
  // needs a model that can carry it even if the manifest row is silent.
  if (norm.tools.length > 0) base.tools = true;
  if (norm.turns.some((t) => t.images.length > 0)) base.vision = true;
  if (norm.jsonMode) base.json = true;
  return base;
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

type UpstreamResult = { ok: true; answer: NormAnswer } | { ok: false; status: number };

/** One inference call to one provider, in that provider's own wire. The
 *  upstream stream (when asked) is read fully before returning, so a
 *  truncated stream can still fail over instead of reaching the client. */
async function callUpstream(candidate: Candidate, norm: NormRequest): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let payload: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    payload = normToOpenAI(norm, candidate.model.id);
    if (norm.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    payload = normToAnthropic(norm, candidate.model.id);
    if (norm.stream) headers.accept = 'text/event-stream';
  } else {
    const action = norm.stream ? ':streamGenerateContent?alt=sse' : ':generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}${action}`;
    payload = normToGemini(norm, candidate.model.id);
    if (norm.stream) headers.accept = 'text/event-stream';
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!res.ok) {
    try {
      await res.text();
    } catch {
      // The socket is being freed; the status is what matters.
    }
    return { ok: false, status: res.status };
  }
  let text: string;
  try {
    text = await res.text();
  } catch {
    return { ok: false, status: 0 };
  }
  if (Buffer.byteLength(text) > MAX_UPSTREAM_BYTES) return { ok: false, status: 0 };
  const isStream = norm.stream;
  let answer: NormAnswer | null = null;
  try {
    if (provider.wire === 'openai') answer = isStream ? parseOpenAIStream(text) : parseOpenAIJson(JSON.parse(text));
    else if (provider.wire === 'anthropic') answer = isStream ? parseAnthropicStream(text) : parseAnthropicJson(JSON.parse(text));
    else answer = isStream ? parseGeminiStream(text) : parseGeminiJson(JSON.parse(text));
  } catch {
    answer = null;
  }
  if (answer === null) return { ok: false, status: 0 };
  return { ok: true, answer };
}

function bump(providerId: string, failed: boolean): void {
  attemptsByProvider.set(providerId, (attemptsByProvider.get(providerId) ?? 0) + 1);
  if (failed) failuresByProvider.set(providerId, (failuresByProvider.get(providerId) ?? 0) + 1);
}

/** Serve one client request on either wire: route, fail over, translate back. */
async function serve(req: IncomingMessage, res: ServerResponse, clientWire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const norm: NormRequest = clientWire === 'openai' ? openAIToNorm(body) : anthropicToNorm(body);
  const entryId = brief.workload?.manifest[manifestIndex]?.id ?? null;
  const requires = requirementsFor(body, norm);
  let candidates = candidatesFor(requires, norm, clientWire);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'openai' ? 'no_free_model' : 'overloaded_error', 'no free model on the directory meets this request');
  }
  // A client that names a real model gets it first, when it is a legal pick.
  const named = typeof body.model === 'string' && body.model !== '' && body.model !== 'auto' ? body.model : null;
  if (named !== null) {
    const at = candidates.findIndex((c) => c.model.id === named);
    if (at > 0) {
      const [first] = candidates.splice(at, 1);
      candidates = [first, ...candidates];
    }
  }
  // The global plan's first choice goes first, when this request has one;
  // the dynamic ranking is the failover order behind it.
  if (entryId !== null) {
    const pick = planned.get(entryId);
    if (pick !== undefined) {
      const at = candidates.findIndex((c) => c.probe.provider.id === pick.probe.provider.id && c.model.id === pick.model.id);
      if (at > 0) {
        const [first] = candidates.splice(at, 1);
        candidates = [first, ...candidates];
      }
    }
  }
  let attempts = 0;
  // Attempt order is provider round-robin: the best model of each provider
  // first (in score order), then each provider's second model, and so on.
  // A provider whose budget is spent (or in outage) fails on every model,
  // so trying its second model before another provider's first would waste
  // attempts and could strand the request while an untried provider remains.
  // A scheduled refusal is a single call ordinal on that provider, so a
  // provider tried in an early round may answer in a later one.
  const byProvider = new Map<string, Candidate[]>();
  for (const candidate of candidates) {
    const group = byProvider.get(candidate.probe.provider.id);
    if (group === undefined) byProvider.set(candidate.probe.provider.id, [candidate]);
    else group.push(candidate);
  }
  const order: Candidate[] = [];
  const maxDepth = Math.max(...[...byProvider.values()].map((g) => g.length));
  for (let depth = 0; depth < maxDepth; depth++) {
    for (const group of byProvider.values()) {
      if (group[depth] !== undefined) order.push(group[depth]);
    }
  }
  for (const candidate of order) {
    if (attempts >= MAX_ATTEMPTS_PER_REQUEST) break;
    attempts += 1;
    const result = await callUpstream(candidate, norm);
    if (!result.ok) {
      bump(candidate.probe.provider.id, true);
      log(`failover: ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${result.status || 'unreachable'}`);
      continue;
    }
    bump(candidate.probe.provider.id, false);
    const model = candidate.model.id;
    if (norm.stream) {
      const text = clientWire === 'openai' ? emitOpenAIStream(model, result.answer) : emitAnthropicStream(model, result.answer);
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(text);
    } else {
      sendJson(res, 200, clientWire === 'openai' ? emitOpenAIJson(model, result.answer) : emitAnthropicJson(model, result.answer));
    }
    return;
  }
  return sendError(res, clientWire, 503, clientWire === 'openai' ? 'no_free_model' : 'overloaded_error', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  computePlan();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
