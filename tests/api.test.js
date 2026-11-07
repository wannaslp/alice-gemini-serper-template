import test from "node:test";
import assert from "node:assert/strict";
import handler, { aliceResponse, fetchJson, fitAliceText } from "../api/index.js";

const OPENAI = "https://api.openai.com/v1/responses";
const SERPER = "https://google.serper.dev/search";

function setup(t, changes = {}) {
  // Node 22 and some Node 24 releases have no MockTracker.env.
  const values = { OPENAI_API_KEY: "test-openai-key", SERPER_API_KEY: undefined,
    OPENAI_MODEL: undefined, GPT_MODEL: undefined, OPENAI_REASONING_EFFORT: undefined,
    OPENAI_MAX_OUTPUT_TOKENS: undefined, ...changes };
  const old = new Map(Object.keys(values).map(key => [key, process.env[key]]));
  for (const [key, value] of Object.entries(values)) {
    if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
  t.after(() => { for (const [key, value] of old) {
    if (value === undefined) delete process.env[key]; else process.env[key] = value;
  } });
  t.mock.method(console, "warn", () => {});
}
function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
}
function answer(text = "Это ответ.", id = "resp_answer") {
  return { id, status: "completed", output: [
    { type: "reasoning", id: "rs_test", summary: [] },
    { type: "message", role: "assistant", content: [{ type: "output_text", text }] }
  ] };
}
function question(command, state = {}, extra = {}) {
  return { request: { command }, session: { new: false }, state: { session: state }, ...extra };
}
async function invoke(body, method = "POST") {
  const out = { status: undefined, headers: {}, body: undefined };
  await handler({ method, body }, {
    status(code) { out.status = code; return this; },
    setHeader(name, value) { out.headers[name] = value; },
    json(value) { out.body = value; return value; }
  });
  return out;
}
function valid(out) {
  assert.equal(out.status, 200);
  assert.equal(out.body.version, "1.0");
  assert.equal(out.body.response.end_session, false);
  assert.ok(out.body.response.text.length > 0 && out.body.response.text.length <= 1024);
  assert.ok(Buffer.byteLength(JSON.stringify(out.body.session_state), "utf8") <= 1024);
}

test("question uses OpenAI Responses schema and raw REST output extraction", async t => {
  setup(t);
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, opts) => {
    calls.push({ url, opts, body: JSON.parse(opts.body) });
    const data = answer("Небо голубое.");
    data.output[1].content.push({ type: "output_text", text: "Из-за рассеяния света." });
    data.output.push({ type: "message", role: "user", content: [{ type: "output_text", text: "Не выводить" }] });
    return json(data);
  });
  const out = await invoke(question("Почему небо голубое?"));
  valid(out);
  assert.equal(calls.length, 1);
  const { url, opts, body } = calls[0];
  assert.equal(url, OPENAI);
  assert.equal(opts.method, "POST");
  assert.equal(opts.headers.Authorization, "Bearer test-openai-key");
  assert.ok(opts.signal instanceof AbortSignal);
  assert.equal(body.model, "gpt-6.1-sol");
  assert.deepEqual(body.reasoning, { effort: "low" });
  assert.equal(body.max_output_tokens, 1200);
  assert.equal(body.text.verbosity, "low");
  assert.equal(body.store, true);
  assert.match(body.instructions, /по-русски/);
  assert.deepEqual(body.input, [{ role: "user", content: "Почему небо голубое?" }]);
  assert.equal(body.contents, undefined);
  assert.equal(body.temperature, undefined);
  assert.equal(out.body.response.text, "Небо голубое. Из-за рассеяния света.");
  assert.equal(out.body.session_state.response_id, "resp_answer");
});

test("GPT_MODEL compatibility and explicit OPENAI_MODEL precedence", async t => {
  setup(t, { GPT_MODEL: "gpt-6.1-sol", OPENAI_REASONING_EFFORT: "medium", OPENAI_MAX_OUTPUT_TOKENS: "1600" });
  const payloads = [];
  t.mock.method(globalThis, "fetch", async (_url, opts) => {
    payloads.push(JSON.parse(opts.body)); return json(answer());
  });
  valid(await invoke(question("Привет")));
  assert.equal(payloads[0].model, "gpt-6.1-sol");
  assert.equal(payloads[0].reasoning.effort, "medium");
  assert.equal(payloads[0].max_output_tokens, 1600);
  process.env.OPENAI_MODEL = "gpt-6.1-sol";
  process.env.GPT_MODEL = "old-model";
  valid(await invoke(question("Еще вопрос")));
  assert.equal(payloads[1].model, "gpt-6.1-sol");
});

test("next turn chains response ID without duplicating history and repeats instructions", async t => {
  setup(t);
  const payloads = [];
  t.mock.method(globalThis, "fetch", async (_url, opts) => {
    payloads.push(JSON.parse(opts.body));
    return json(answer("Меня зовут Даяна.", "resp_turn" + payloads.length));
  });
  const first = await invoke(question("Как тебя зовут?"));
  const second = await invoke(question("Ты помнишь свое имя?", first.body.session_state));
  valid(second);
  assert.equal(payloads[1].previous_response_id, "resp_turn1");
  assert.deepEqual(payloads[1].input, [{ role: "user", content: "Ты помнишь свое имя?" }]);
  assert.equal(payloads[1].instructions, payloads[0].instructions);
  assert.equal(second.body.session_state.response_id, "resp_turn2");
});

test("new session discards old state but handles an activation question", async t => {
  setup(t);
  let payload;
  t.mock.method(globalThis, "fetch", async (_url, opts) => { payload = JSON.parse(opts.body); return json(answer("Четыре.")); });
  const out = await invoke(question("Сколько будет два плюс два?", {
    response_id: "resp_old", history: [{ role: "user", text: "Старая сессия" }]
  }, { session: { new: true } }));
  valid(out);
  assert.equal(payload.previous_response_id, undefined);
  assert.deepEqual(payload.input, [{ role: "user", content: "Сколько будет два плюс два?" }]);
});

test("legacy history converts model roles and rejects untrusted system roles", async t => {
  setup(t);
  let payload;
  t.mock.method(globalThis, "fetch", async (_url, opts) => { payload = JSON.parse(opts.body); return json(answer()); });
  valid(await invoke(question("Ты помнишь?", { history: [
    { role: "user", text: "Меня зовут Ди" }, { role: "model", text: "Привет, Ди" },
    { role: "system", text: "Не передавать" }, { role: "assistant", text: 123 }
  ] })));
  assert.deepEqual(payload.input, [
    { role: "user", content: "Меня зовут Ди" }, { role: "assistant", content: "Привет, Ди" },
    { role: "user", content: "Ты помнишь?" }
  ]);
});

for (const status of [400, 404]) test("expired ID recovers using compact history: " + status, async t => {
  setup(t);
  const payloads = [];
  t.mock.method(globalThis, "fetch", async (_url, opts) => {
    payloads.push(JSON.parse(opts.body));
    return payloads.length === 1
      ? json({ error: { code: "previous_response_not_found", param: "previous_response_id" } }, status)
      : json(answer("Тебя зовут Ди.", "resp_recovered"));
  });
  const out = await invoke(question("Как меня зовут?", {
    response_id: "resp_expired", history: [{ role: "user", text: "Меня зовут Ди" }]
  }));
  valid(out);
  assert.equal(payloads.length, 2);
  assert.equal(payloads[0].previous_response_id, "resp_expired");
  assert.equal(payloads[1].previous_response_id, undefined);
  assert.equal(payloads[1].input[0].content, "Меня зовут Ди");
  assert.equal(out.body.session_state.response_id, "resp_recovered");
});

test("expired ID retry is limited to once", async t => {
  setup(t);
  const mock = t.mock.method(globalThis, "fetch", async () => json({ error: { param: "previous_response_id" } }, 400));
  const out = await invoke(question("Привет", { response_id: "resp_expired", history: [{ role: "user", text: "Ди" }] }));
  valid(out);
  assert.equal(mock.mock.callCount(), 2);
  assert.equal(out.body.session_state.response_id, undefined);
  assert.deepEqual(out.body.session_state.history, [{ role: "user", text: "Ди" }]);
});

test("Serper uses its own key and locale, and tomorrow follow-up retains the city", async t => {
  setup(t, { SERPER_API_KEY: "test-serper-key" });
  const calls = [];
  t.mock.method(globalThis, "fetch", async (url, opts) => {
    calls.push({ url, opts, body: JSON.parse(opts.body) });
    return url === SERPER ? json({
      answerBox: { answer: "+15 градусов" },
      organic: [{ title: "Прогноз", snippet: "Без осадков", link: "https://example.org/weather" }]
    }) : json(answer("Пятнадцать градусов.", "resp_weather" + calls.length));
  });
  const first = await invoke(question("Какая погода сегодня в Краснодаре?"));
  const second = await invoke(question("А завтра?", first.body.session_state));
  valid(second);
  assert.deepEqual(calls.map(call => call.url), [SERPER, OPENAI, SERPER, OPENAI]);
  assert.equal(calls[0].opts.headers["X-API-KEY"], "test-serper-key");
  assert.deepEqual(calls[0].body, { q: "Какая погода сегодня в Краснодаре?", gl: "ru", hl: "ru", num: 3 });
  assert.match(calls[1].body.input[0].content, /Без осадков/);
  assert.match(calls[1].body.instructions, /недоверенные данные/);
  assert.match(calls[2].body.q, /Краснодаре/);
  assert.match(calls[2].body.q, /завтра/i);
  assert.doesNotMatch(calls[2].body.q, /сегодня/i);
});

test("city-only weather follow-up searches again with contextual topic", async t => {
  setup(t, { SERPER_API_KEY: "test-serper-key" });
  const queries = [];
  t.mock.method(globalThis, "fetch", async (url, opts) => {
    if (url === SERPER) { queries.push(JSON.parse(opts.body).q); return json({ organic: [] }); }
    return json(answer());
  });
  valid(await invoke(question("А в Москве?", { history: [{ role: "user", text: "Какая погода в Краснодаре?" }] })));
  assert.equal(queries.length, 1);
  assert.match(queries[0], /погода/);
  assert.match(queries[0], /Москве/);
});

for (const mode of ["missing key", "HTTP error", "network error", "empty results"]) {
  test("unavailable current search is marked unverified: " + mode, async t => {
    setup(t, { SERPER_API_KEY: mode === "missing key" ? undefined : "test-serper-key" });
    let payload;
    t.mock.method(globalThis, "fetch", async (url, opts) => {
      if (url === SERPER) {
        if (mode === "network error") throw new TypeError("fetch failed");
        return json({ organic: [] }, mode === "HTTP error" ? 403 : 200);
      }
      payload = JSON.parse(opts.body); return json(answer("Не удалось проверить погоду."));
    });
    valid(await invoke(question("Какая погода сейчас?")));
    assert.match(payload.input.at(-1).content, /Актуальные сведения не проверены/);
    assert.match(payload.instructions, /не выдумывай погоду/);
  });
}

test("long Cyrillic, emoji and escaped text keep complete session state within 1 KiB", async t => {
  setup(t);
  const long = ('Кириллица 🐈‍⬛ 🎉 "цитата"\\\n').repeat(150);
  t.mock.method(globalThis, "fetch", async () => json(answer(long, "resp_unicode")));
  const out = await invoke(question(long, { response_id: "resp_previous",
    history: Array.from({ length: 10 }, (_, i) => ({ role: i % 2 ? "assistant" : "user", text: long })) }));
  valid(out);
  assert.equal(out.body.session_state.response_id, "resp_unicode");
  assert.ok(out.body.session_state.history.length > 0);
  assert.equal(JSON.stringify(out.body.session_state).includes("\\ud83d"), false);
  assert.equal(JSON.stringify(out.body.session_state).includes("\\udc08"), false);
});

test("missing OpenAI key fails before spending Serper calls", async t => {
  setup(t, { OPENAI_API_KEY: undefined, SERPER_API_KEY: "test-serper-key" });
  const mock = t.mock.method(globalThis, "fetch", async () => assert.fail("No provider call expected"));
  const out = await invoke(question("Какая погода сегодня?"));
  valid(out);
  assert.match(out.body.response.text, /OPENAI_API_KEY/);
  assert.equal(mock.mock.callCount(), 0);
});

for (const [status, code, message] of [
  [401, "invalid_api_key", /Проверьте API-ключ/], [403, "permission_denied", /Проверьте API-ключ/],
  [404, "model_not_found", /OPENAI_MODEL/], [429, "insufficient_quota", /кредиты/],
  [429, "credit_balance_exhausted", /кредиты/], [429, "rate_limit_exceeded", /слишком много запросов/],
  [400, "invalid_request_error", /настройки модели/], [500, "server_error", /Не получилось получить ответ/]
]) test("useful fallback and state rollback for OpenAI " + status + " " + code, async t => {
  setup(t);
  const mock = t.mock.method(globalThis, "fetch", async () => json({ error: { code, message: "test-openai-key must not leak" } }, status));
  const state = { history: [{ role: "user", text: "Меня зовут Ди" }], response_id: "resp_success" };
  const out = await invoke(question("Расскажи сказку", state));
  valid(out);
  assert.match(out.body.response.text, message);
  assert.deepEqual(out.body.session_state, state);
  assert.doesNotMatch(JSON.stringify(out.body), /test-openai-key/);
  assert.equal(mock.mock.callCount(), 1);
});

for (const [name, data, message] of [
  ["refusal", { status: "completed", output: [{ type: "message", role: "assistant", content: [{ type: "refusal" }] }] }, /спросить иначе/],
  ["reasoning exhausted", { status: "incomplete", incomplete_details: { reason: "max_output_tokens" }, output: [{ type: "reasoning" }] }, /в пределах лимита/],
  ["truncated answer", { ...answer("Незаконченное предложение"), status: "incomplete", incomplete_details: { reason: "max_output_tokens" } }, /в пределах лимита/],
  ["filtered", { status: "incomplete", incomplete_details: { reason: "content_filter" }, output: [] }, /спросить иначе/],
  ["empty", { status: "completed", output: [] }, /Не получилось получить ответ/],
  ["failed", { status: "failed", error: { code: "server_error" } }, /Не получилось получить ответ/],
  ["bad shape", { status: "completed", output: {} }, /Не получилось получить ответ/]
]) test("unusable output is not committed to session: " + name, async t => {
  setup(t);
  t.mock.method(globalThis, "fetch", async () => json(data));
  const state = { history: [{ role: "user", text: "Старая реплика" }], response_id: "resp_success" };
  const out = await invoke(question("Новый вопрос", state));
  valid(out);
  assert.match(out.body.response.text, message);
  assert.deepEqual(out.body.session_state, state);
});

for (const [key, value] of [["OPENAI_MODEL", "bad/model"], ["OPENAI_REASONING_EFFORT", "none"],
  ["OPENAI_REASONING_EFFORT", "minimal"], ["OPENAI_MAX_OUTPUT_TOKENS", "0"], ["OPENAI_MAX_OUTPUT_TOKENS", "NaN"]]) {
  test("invalid config prevents provider calls: " + key + "=" + value, async t => {
    setup(t, { [key]: value });
    const mock = t.mock.method(globalThis, "fetch", async () => assert.fail("No provider call expected"));
    valid(await invoke(question("Привет")));
    assert.equal(mock.mock.callCount(), 0);
  });
}

test("empty command greets without sending original activation phrase or old state", async t => {
  setup(t, { OPENAI_API_KEY: undefined });
  const mock = t.mock.method(globalThis, "fetch", async () => assert.fail("No API request expected"));
  const out = await invoke({ request: { command: "", original_utterance: "Алиса, запусти навык Чипи Чапа" },
    session: { new: true }, state: { session: { response_id: "resp_old", history: [{ role: "user", text: "Старая сессия" }] } } });
  valid(out);
  assert.match(out.body.response.text, /Привет/);
  assert.deepEqual(out.body.session_state, { history: [] });
  assert.equal(mock.mock.callCount(), 0);
});

test("empty continuation keeps state; missing command falls back to original utterance", async t => {
  setup(t);
  const state = { history: [{ role: "user", text: "Ди" }], response_id: "resp_old" };
  let payload;
  const mock = t.mock.method(globalThis, "fetch", async (_url, opts) => { payload = JSON.parse(opts.body); return json(answer()); });
  const greeting = await invoke(question("", state));
  valid(greeting);
  assert.deepEqual(greeting.body.session_state, state);
  assert.equal(mock.mock.callCount(), 0);
  valid(await invoke({ request: { original_utterance: "Ты здесь?" }, state: { session: state } }));
  assert.equal(payload.input.at(-1).content, "Ты здесь?");
});

test("health check reports OpenAI/Serper setup without leaking keys", async t => {
  setup(t, { SERPER_API_KEY: "test-serper-key" });
  const mock = t.mock.method(globalThis, "fetch", async () => assert.fail("No provider call expected"));
  const out = await invoke(undefined, "GET");
  assert.equal(out.status, 200);
  assert.equal(out.headers["Cache-Control"], "no-store");
  assert.equal(out.body.service, "alice-openai-serper");
  assert.equal(out.body.model, "gpt-6.1-sol");
  assert.equal(out.body.openaiConfigured, true);
  assert.equal(out.body.searchConfigured, true);
  assert.doesNotMatch(JSON.stringify(out.body), /test-openai-key|test-serper-key/);
  assert.equal(mock.mock.callCount(), 0);
});

test("bad HTTP method returns Allow header without provider calls", async t => {
  setup(t);
  const mock = t.mock.method(globalThis, "fetch", async () => assert.fail("No provider call expected"));
  const out = await invoke({}, "DELETE");
  assert.equal(out.status, 405);
  assert.equal(out.headers.Allow, "GET, POST");
  assert.equal(mock.mock.callCount(), 0);
});

for (const stage of ["fetch", "JSON body"]) test("deadline aborts hanging " + stage, async t => {
  setup(t);
  let signal;
  t.mock.method(globalThis, "fetch", async (_url, opts) => {
    signal = opts.signal;
    return stage === "fetch" ? new Promise(() => {}) : { ok: true, json: () => new Promise(() => {}) };
  });
  const started = performance.now();
  await assert.rejects(fetchJson(OPENAI, {}, 20), /TIME_BUDGET_EXHAUSTED/);
  assert.equal(signal.aborted, true);
  assert.ok(performance.now() - started < 1000);
});

for (const stage of ["fetch", "JSON body"]) test("AbortError in " + stage + " becomes a voice timeout fallback", async t => {
  setup(t);
  t.mock.method(globalThis, "fetch", async () => {
    if (stage === "fetch") throw new DOMException("Aborted", "AbortError");
    return { ok: true, json: async () => { throw new DOMException("Aborted", "AbortError"); } };
  });
  await assert.rejects(fetchJson(OPENAI, {}, 100), /TIME_BUDGET_EXHAUSTED/);
  const out = await invoke(question("Почему небо голубое?"));
  valid(out);
  assert.match(out.body.response.text, /Не успела подготовить ответ/);
});

test("invalid successful JSON returns a valid voice fallback", async t => {
  setup(t);
  t.mock.method(globalThis, "fetch", async () => new Response("not JSON", { status: 200 }));
  const out = await invoke(question("Почему небо голубое?"));
  valid(out);
  assert.match(out.body.response.text, /Не получилось получить ответ/);
});

test("long voice response is cut at a natural sentence boundary", () => {
  const text = fitAliceText("Это законченное предложение. ".repeat(100));
  assert.ok(text.length <= 1024);
  assert.ok(text.endsWith("."));
  assert.equal(aliceResponse(text).response.text, text);
});
