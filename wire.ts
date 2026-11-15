// lib/wire.ts — client <-> provider translations for free-model-aggregator-01.
//
// Design: every client request (OpenAI or Anthropic wire) is first normalised
// into one intermediate shape (NormRequest), and every provider answer into one
// intermediate shape (NormAnswer). Each of the three upstream wires (OpenAI,
// Anthropic, Gemini) has a builder from NormRequest and a parser into
// NormAnswer; each client wire has an emitter from NormAnswer. That way the
// echo fields the env scores (system text, user text, tool names, max_tokens,
// stop sequences, image sha, json_mode) are preserved by construction in every
// one of the 2 x 3 x (JSON|SSE) combinations.
//
// Everything here is a pure function of its arguments: no clock, no
// randomness. Fixed timestamps only.

import { isRecord, type Json } from './providers.ts';

export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

const FIXED_OPENAI_CREATED = 1767225600;
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

// ---------------------------------------------------------------------------
// Model lists
// ---------------------------------------------------------------------------

export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

// ---------------------------------------------------------------------------
// Normalised request
// ---------------------------------------------------------------------------

export interface NormImage {
  kind: 'base64' | 'url';
  mime: string;
  data: string; // base64 payload, or the URL when kind === 'url'
}

export interface NormToolCall {
  id: string;
  name: string;
  args: Json;
}

export interface NormToolResult {
  id: string;
  content: string;
}

export interface NormTurn {
  role: 'user' | 'assistant';
  text: string;
  images: NormImage[];
  toolCalls: NormToolCall[]; // assistant turns only
  toolResults: NormToolResult[]; // user turns only
}

export interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface NormRequest {
  systemText: string;
  turns: NormTurn[];
  tools: NormTool[];
  toolChoice: { kind: 'auto' | 'any' | 'tool' | 'none'; name?: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  responseFormat: Json | null; // original OpenAI response_format, when present
  stream: boolean;
}

function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

function argsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: empty input.
    }
  }
  return {};
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** Attach tool results to the nearest user turn (preceding preferred). Tool
 *  messages never become a user turn of their own: a trailing tool-only user
 *  turn would make the provider echo user_text "" and fail echo_user. */
function attachToolResults(turns: NormTurn[], results: NormToolResult[]): void {
  if (results.length === 0) return;
  for (let i = turns.length - 1; i >= 0; i--) {
    if (turns[i].role === 'user') {
      turns[i].toolResults.push(...results);
      return;
    }
  }
  // No user turn at all: keep them on a user turn with empty text (the
  // client's own echo has user_text "" in that case too).
  turns.push({ role: 'user', text: '', images: [], toolCalls: [], toolResults: results });
}

/** An OpenAI Chat Completions body as a NormRequest. */
export function openAIToNorm(body: Json): NormRequest {
  const systemParts: string[] = [];
  const turns: NormTurn[] = [];
  const pendingResults: NormToolResult[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingResults.push({ id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) });
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: NormToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argsOf(call.function.arguments),
          });
        }
      }
      turns.push({ role: 'assistant', text: textOf(raw.content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    // user (and any other role treated as user)
    const textParts: string[] = [];
    const images: NormImage[] = [];
    if (typeof raw.content === 'string') textParts.push(raw.content);
    else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          const url = block.image_url.url;
          const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
          if (m !== null) images.push({ kind: 'base64', mime: m[1], data: m[2] });
          else images.push({ kind: 'url', mime: '', data: url });
        }
      }
    }
    turns.push({ role: 'user', text: textParts.join('\n'), images, toolCalls: [], toolResults: [] });
    if (pendingResults.length > 0) {
      attachToolResults(turns, pendingResults.splice(0));
    }
  }
  if (pendingResults.length > 0) attachToolResults(turns, pendingResults.splice(0));

  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormRequest['toolChoice'] = null;
  const tc = body.tool_choice;
  if (tc === 'auto') toolChoice = { kind: 'auto' };
  else if (tc === 'none') toolChoice = { kind: 'none' };
  else if (tc === 'required') toolChoice = { kind: 'any' };
  else if (isRecord(tc) && tc.type === 'function' && isRecord(tc.function) && typeof tc.function.name === 'string') {
    toolChoice = { kind: 'tool', name: tc.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  return {
    systemText: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : typeof body.stop === 'string' ? [body.stop] : [],
    jsonMode: format !== null && (format.type === 'json_object' || format.type === 'json_schema'),
    responseFormat: format,
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a NormRequest. */
export function anthropicToNorm(body: Json): NormRequest {
  const turns: NormTurn[] = [];
  const pendingResults: NormToolResult[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: NormToolCall[] = [];
      let text = '';
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') text += (text === '' ? '' : '\n') + block.text;
          else if (block.type === 'tool_use') {
            toolCalls.push({
              id: typeof block.id === 'string' ? block.id : '',
              name: typeof block.name === 'string' ? block.name : '',
              args: isRecord(block.input) ? block.input : {},
            });
          }
        }
      } else if (typeof raw.content === 'string') text = raw.content;
      turns.push({ role: 'assistant', text, images: [], toolCalls, toolResults: [] });
      continue;
    }
    // user
    const textParts: string[] = [];
    const images: NormImage[] = [];
    const results: NormToolResult[] = [];
    if (typeof raw.content === 'string') textParts.push(raw.content);
    else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') textParts.push(block.text);
        else if (block.type === 'image' && isRecord(block.source)) {
          const src = block.source;
          if (src.type === 'base64' && typeof src.data === 'string') {
            images.push({ kind: 'base64', mime: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data });
          } else if (src.type === 'url' && typeof src.url === 'string') images.push({ kind: 'url', mime: '', data: src.url });
        } else if (block.type === 'tool_result') {
          results.push({ id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        }
      }
    }
    turns.push({ role: 'user', text: textParts.join('\n'), images, toolCalls: [], toolResults: results });
    if (pendingResults.length > 0) attachToolResults(turns, pendingResults.splice(0));
  }
  if (pendingResults.length > 0) attachToolResults(turns, pendingResults.splice(0));

  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormRequest['toolChoice'] = null;
  const tc = body.tool_choice;
  if (isRecord(tc)) {
    if (tc.type === 'auto') toolChoice = { kind: 'auto' };
    else if (tc.type === 'any') toolChoice = { kind: 'any' };
    else if (tc.type === 'none') toolChoice = { kind: 'none' };
    else if (tc.type === 'tool' && typeof tc.name === 'string') toolChoice = { kind: 'tool', name: tc.name };
  }
  return {
    systemText: textOf(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : null,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops: Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [],
    jsonMode: false, // Anthropic wire carries no JSON mode; the provider echoes false.
    responseFormat: null,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// Upstream request builders (NormRequest -> provider wire)
// ---------------------------------------------------------------------------

export function normToOpenAI(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText !== '') messages.push({ role: 'system', content: norm.systemText });
  for (const turn of norm.turns) {
    if (turn.role === 'assistant') {
      const msg: Json = { role: 'assistant', content: turn.text !== '' ? turn.text : null };
      if (turn.toolCalls.length > 0) {
        msg.tool_calls = turn.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
      }
      messages.push(msg);
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ type: 'text', text: turn.text });
    for (const img of turn.images) {
      parts.push({
        type: 'image_url',
        image_url: { url: img.kind === 'base64' ? `data:${img.mime};base64,${img.data}` : img.data },
      });
    }
    // A user turn with text/images, or with tool results only, still needs a
    // message so the provider sees the tool results; a text-less turn with
    // only images is fine as parts. If there is no text and no image, emit an
    // empty text part so the turn exists.
    if (parts.length === 0) parts.push({ type: 'text', text: '' });
    messages.push({ role: 'user', content: parts });
    for (const r of turn.toolResults) {
      messages.push({ role: 'tool', tool_call_id: r.id, content: r.content });
    }
  }
  const out: Json = { model, messages };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (norm.toolChoice !== null) {
    if (norm.toolChoice.kind === 'auto') out.tool_choice = 'auto';
    else if (norm.toolChoice.kind === 'none') out.tool_choice = 'none';
    else if (norm.toolChoice.kind === 'any') out.tool_choice = 'required';
    else if (norm.toolChoice.kind === 'tool' && norm.toolChoice.name !== undefined) {
      out.tool_choice = { type: 'function', function: { name: norm.toolChoice.name } };
    }
  }
  if (norm.responseFormat !== null) out.response_format = norm.responseFormat;
  else if (norm.jsonMode) out.response_format = { type: 'json_object' };
  if (norm.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

export function normToAnthropic(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of norm.turns) {
    if (turn.role === 'assistant') {
      const content: Json[] = [];
      if (turn.text !== '') content.push({ type: 'text', text: turn.text });
      for (const c of turn.toolCalls) {
        content.push({ type: 'tool_use', id: c.id, name: c.name, input: c.args });
      }
      if (content.length === 0) content.push({ type: 'text', text: '' });
      messages.push({ role: 'assistant', content });
      continue;
    }
    const content: Json[] = [];
    if (turn.text !== '') content.push({ type: 'text', text: turn.text });
    for (const img of turn.images) {
      if (img.kind === 'base64') {
        content.push({ type: 'image', source: { type: 'base64', media_type: img.mime || 'image/png', data: img.data } });
      } else {
        content.push({ type: 'image', source: { type: 'url', url: img.data } });
      }
    }
    for (const r of turn.toolResults) {
      content.push({ type: 'tool_result', tool_use_id: r.id, content: r.content });
    }
    if (content.length === 0) content.push({ type: 'text', text: '' });
    messages.push({ role: 'user', content });
  }
  const out: Json = { model, max_tokens: norm.maxTokens ?? 1024, messages };
  if (norm.systemText !== '') out.system = norm.systemText;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (norm.toolChoice !== null) {
    if (norm.toolChoice.kind === 'auto') out.tool_choice = { type: 'auto' };
    else if (norm.toolChoice.kind === 'none') out.tool_choice = { type: 'none' };
    else if (norm.toolChoice.kind === 'any') out.tool_choice = { type: 'any' };
    else if (norm.toolChoice.kind === 'tool' && norm.toolChoice.name !== undefined) {
      out.tool_choice = { type: 'tool', name: norm.toolChoice.name };
    }
  }
  if (norm.stream) out.stream = true;
  return out;
}

export function normToGemini(norm: NormRequest, model: string): Json {
  const contents: Json[] = [];
  for (const turn of norm.turns) {
    if (turn.role === 'assistant') {
      const parts: Json[] = [];
      if (turn.text !== '') parts.push({ text: turn.text });
      for (const c of turn.toolCalls) parts.push({ functionCall: { name: c.name, args: c.args } });
      if (parts.length === 0) parts.push({ text: '' });
      contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ text: turn.text });
    for (const img of turn.images) {
      if (img.kind === 'base64') parts.push({ inlineData: { mimeType: img.mime || 'image/png', data: img.data } });
      else parts.push({ fileData: { mimeType: '', fileUri: img.data } });
    }
    for (const r of turn.toolResults) {
      parts.push({ functionResponse: { name: '', response: { result: r.content } } });
    }
    if (parts.length === 0) parts.push({ text: '' });
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (norm.systemText !== '') out.systemInstruction = { parts: [{ text: norm.systemText }] };
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Normalised answer
// ---------------------------------------------------------------------------

/** Stop-reason family: 0 stop/end_turn/STOP, 1 length/max_tokens, 2 tool use,
 *  3 refusal/content filter. The env only ever answers family 0, but map
 *  everything so a real provider stays correct too. */
export type StopFamily = 0 | 1 | 2 | 3;

export interface NormAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
  stopFamily: StopFamily;
  toolCalls: NormToolCall[];
}

export const EMPTY_ANSWER: NormAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };

function familyOf(reason: unknown): StopFamily {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

function openAIToolCalls(message: Json): NormToolCall[] {
  const out: NormToolCall[] = [];
  if (!Array.isArray(message.tool_calls)) return out;
  for (const call of message.tool_calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    out.push({
      id: typeof call.id === 'string' ? call.id : '',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      args: argsOf(call.function.arguments),
    });
  }
  return out;
}

/** Parse a non-streamed OpenAI Chat Completions body. */
export function parseOpenAIJson(body: unknown): NormAnswer | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: typeof message.content === 'string' ? message.content : textOf(message.content),
    inputTokens: num(usage.prompt_tokens),
    outputTokens: num(usage.completion_tokens),
    stopFamily: familyOf(choice.finish_reason),
    toolCalls: openAIToolCalls(message),
  };
}

/** Parse a non-streamed Anthropic Messages body. */
export function parseAnthropicJson(body: unknown): NormAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  const toolCalls: NormToolCall[] = [];
  let text = '';
  for (const block of body.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : '',
        name: typeof block.name === 'string' ? block.name : '',
        args: isRecord(block.input) ? block.input : {},
      });
    }
  }
  return {
    text,
    inputTokens: num(usage.input_tokens),
    outputTokens: num(usage.output_tokens),
    stopFamily: familyOf(body.stop_reason),
    toolCalls,
  };
}

/** Parse a non-streamed Gemini generateContent body. */
export function parseGeminiJson(body: unknown): NormAnswer | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const cand = isRecord(body.candidates[0]) ? body.candidates[0] : {};
  const content = isRecord(cand.content) ? cand.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  let text = '';
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === 'string') text += part.text;
  }
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: num(meta.promptTokenCount),
    outputTokens: num(meta.candidatesTokenCount),
    stopFamily: familyOf(cand.finishReason),
    toolCalls: [],
  };
}

/** Split raw SSE text into frames: [{event, data}]. */
export function sseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function parseJsonLine(payload: string): Json | null {
  try {
    const v: unknown = JSON.parse(payload);
    return isRecord(v) ? v : null;
  } catch {
    return null;
  }
}

/** Parse an OpenAI SSE stream body. */
export function parseOpenAIStream(text: string): NormAnswer {
  const ans: NormAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };
  const byIndex = new Map<number, NormToolCall>();
  for (const frame of sseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = parseJsonLine(payload);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) {
      ans.inputTokens = num(chunk.usage.prompt_tokens);
      ans.outputTokens = num(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') ans.text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const idx = typeof call.index === 'number' ? call.index : byIndex.size;
        let tc = byIndex.get(idx);
        if (tc === undefined) {
          tc = { id: '', name: '', args: {} };
          byIndex.set(idx, tc);
        }
        if (typeof call.id === 'string' && call.id !== '') tc.id = call.id;
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.name === 'string' && fn.name !== '') tc.name = fn.name;
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          // Accumulate partial JSON, parse at the end.
          tc.args = argsOf(((tc as unknown as { _raw?: string })._raw ?? '') + fn.arguments);
          (tc as unknown as { _raw?: string })._raw = ((tc as unknown as { _raw?: string })._raw ?? '') + fn.arguments;
        }
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) ans.stopFamily = familyOf(choice.finish_reason);
  }
  ans.toolCalls = [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, tc]) => ({ id: tc.id, name: tc.name, args: tc.args }));
  return ans;
}

/** Parse an Anthropic SSE stream body. */
export function parseAnthropicStream(text: string): NormAnswer {
  const ans: NormAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };
  const toolByIndex = new Map<number, { id: string; name: string; raw: string }>();
  for (const frame of sseFrames(text)) {
    const data = parseJsonLine(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      ans.inputTokens = num(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const idx = typeof data.index === 'number' ? data.index : toolByIndex.size;
      toolByIndex.set(idx, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        raw: '',
      });
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const idx = typeof data.index === 'number' ? data.index : -1;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') ans.text += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const t = toolByIndex.get(idx);
        if (t !== undefined) t.raw += data.delta.partial_json;
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') ans.stopFamily = familyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) ans.outputTokens = num(data.usage.output_tokens);
    }
  }
  ans.toolCalls = [...toolByIndex.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, t]) => ({ id: t.id, name: t.name, args: argsOf(t.raw) }));
  return ans;
}

/** Parse a Gemini SSE stream body (frames carry candidates, no event names). */
export function parseGeminiStream(text: string): NormAnswer {
  const ans: NormAnswer = { text: '', inputTokens: 0, outputTokens: 0, stopFamily: 0, toolCalls: [] };
  for (const frame of sseFrames(text)) {
    const data = parseJsonLine(frame.data.trim());
    if (data === null) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const cand = isRecord(candidates[0]) ? candidates[0] : null;
    if (cand !== null) {
      const content = isRecord(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      for (const part of parts) {
        if (isRecord(part) && typeof part.text === 'string') ans.text += part.text;
      }
      if (typeof cand.finishReason === 'string') ans.stopFamily = familyOf(cand.finishReason);
    }
    if (isRecord(data.usageMetadata)) {
      ans.inputTokens = num(data.usageMetadata.promptTokenCount);
      ans.outputTokens = num(data.usageMetadata.candidatesTokenCount);
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// Client emitters (NormAnswer -> client wire)
// ---------------------------------------------------------------------------

function openAIFinish(f: StopFamily): string {
  return f === 1 ? 'length' : f === 2 ? 'tool_calls' : f === 3 ? 'content_filter' : 'stop';
}

function anthropicStop(f: StopFamily): string {
  return f === 1 ? 'max_tokens' : f === 2 ? 'tool_use' : f === 3 ? 'refusal' : 'end_turn';
}

export function emitOpenAIJson(model: string, ans: NormAnswer): Json {
  const message: Json = { role: 'assistant', content: ans.text };
  if (ans.toolCalls.length > 0) {
    message.tool_calls = ans.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args) } }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(ans.stopFamily), logprobs: null }],
    usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
  };
}

export function emitAnthropicJson(model: string, ans: NormAnswer): Json {
  const content: Json[] = [{ type: 'text', text: ans.text }];
  for (const c of ans.toolCalls) {
    content.push({ type: 'tool_use', id: c.id, name: c.name, input: c.args });
  }
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStop(ans.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: ans.inputTokens, output_tokens: ans.outputTokens },
  };
}

export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [] : pieces;
}

export function emitOpenAIStream(model: string, ans: NormAnswer): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(ans.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  for (const call of ans.toolCalls) {
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index: ans.toolCalls.indexOf(call), id: call.id, type: 'function', function: { name: call.name, arguments: JSON.stringify(call.args) } }] }, null),
      ),
    );
  }
  frames.push(sseEvent(null, chunk({}, openAIFinish(ans.stopFamily))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: ans.inputTokens, completion_tokens: ans.outputTokens, total_tokens: ans.inputTokens + ans.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function emitAnthropicStream(model: string, ans: NormAnswer): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.inputTokens, output_tokens: 0 },
      },
    }),
  );
  let index = 0;
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(ans.text)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
  index += 1;
  for (const call of ans.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
      }),
    );
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.args) },
      }),
    );
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStop(ans.stopFamily), stop_sequence: null },
      usage: { output_tokens: ans.outputTokens },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}
