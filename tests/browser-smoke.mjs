/**
 * Browser smoke test: serves docs/, drives headless Chrome over CDP, and
 * checks the demo works on the real packed site - stats from train.json, the
 * loss chart draws, checkpoint samples render, live in-tab generation through
 * the JS engine produces text and attention maps, the tokenizer viewer works,
 * no page exceptions. Runs locally and in CI.
 * SMOKE_SHOTS=1 writes smoke-*.png screenshots for visual review.
 */
import { spawn } from 'node:child_process';
import { writeFileSync } from 'node:fs';

const HTTP_PORT = 8947;
const CDP_PORT = 9347;
const SHOTS = process.env.SMOKE_SHOTS === '1';
const root = new URL('..', import.meta.url).pathname;

const server = spawn('python3', ['-m', 'http.server', String(HTTP_PORT), '--bind', '127.0.0.1', '--directory', `${root}docs`], { stdio: 'ignore' });
const chrome = spawn('google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-dev-shm-usage', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding',
  `--remote-debugging-port=${CDP_PORT}`, 'about:blank',
], { stdio: 'ignore' });

const cleanup = (code) => {
  try { chrome.kill('SIGKILL'); } catch {}
  try { server.kill('SIGKILL'); } catch {}
  process.exit(code);
};
setTimeout(() => { console.error('smoke: timeout'); cleanup(1); }, 180000);
process.on('SIGINT', () => cleanup(2));

async function waitFor(fn, label, timeoutMs = 60000) {
  const start = Date.now();
  for (;;) {
    try { const v = await fn(); if (v) return v; } catch {}
    if (Date.now() - start > timeoutMs) throw new Error(`timed out waiting for ${label}`);
    await new Promise((r) => setTimeout(r, 250));
  }
}

const t0 = Date.now(); const el = () => ((Date.now() - t0) / 1000).toFixed(1) + 's';
const checks = [];
const check = (name, ok, detail = '') => {
  checks.push({ name, ok, detail });
  console.log(`${el()} ${ok ? 'ok' : 'NOT OK'} - ${name}${detail ? ' - ' + detail : ''}`);
};

await waitFor(async () => (await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`)).ok, 'chrome devtools');
const target = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/new`, { method: 'PUT' })).json();
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => { ws.onopen = resolve; ws.onerror = reject; });

let mid = 0;
const rpcPending = new Map();
const consoleErrors = [];
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.id && rpcPending.has(msg.id)) { rpcPending.get(msg.id)(msg); rpcPending.delete(msg.id); }
  else if (msg.method === 'Runtime.exceptionThrown') {
    consoleErrors.push(msg.params.exceptionDetails?.exception?.description ?? 'page exception');
  }
};
const cdp = (method, params = {}) => new Promise((resolve) => {
  const id = ++mid;
  rpcPending.set(id, resolve);
  ws.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => {
  const res = await cdp('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (res.result?.exceptionDetails) throw new Error(JSON.stringify(res.result.exceptionDetails));
  return res.result?.result?.value;
};
const shot = async (name) => {
  if (!SHOTS) return;
  await new Promise((r) => setTimeout(r, 900));
  const r = await cdp('Page.captureScreenshot', { format: 'png' });
  writeFileSync(`${root}smoke-${name}.png`, Buffer.from(r.result.data, 'base64'));
};

await cdp('Runtime.enable');
await cdp('Page.enable');
if (SHOTS) await cdp('Emulation.setDeviceMetricsOverride', { width: 1440, height: 960, deviceScaleFactor: 1, mobile: false });
await cdp('Page.bringToFront');
await cdp('Page.navigate', { url: `http://127.0.0.1:${HTTP_PORT}/` });

// Boot: stats render from train.json, auto-generation completes.
await waitFor(() => evaluate(`document.querySelectorAll('#stats .stat').length === 8`), 'stat cards');
const paramStat = await evaluate(`document.querySelector('#stats .stat .v').textContent`);
check('boot: param count populated', paramStat.includes('197'), paramStat);
await waitFor(() => evaluate(`document.querySelectorAll('#samples .sample').length >= 5`), 'checkpoint samples');
const step0 = await evaluate(`document.querySelector('#samples .sample .step').textContent`);
check('boot: progression starts at step 0', step0 === 'step 0', step0);

// Loss chart drew pixels.
await waitFor(() => evaluate(`(() => {
  const c = document.getElementById('chart-loss');
  if (!c.width) return false;
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let lit = 0;
  for (let i = 3; i < d.length; i += 400) if (d[i] > 0) lit++;
  return lit;
})()`) , 'loss chart drawn');
check('boot: loss chart drew pixels', true);

// Auto-generation on load: gen-out has seed + new spans with real text.
await waitFor(() => evaluate(`document.querySelectorAll('#gen-out span.new').length === 1 && document.querySelector('#gen-out span.new').textContent.length > 40`), 'auto generation', 120000);
const genLen = await evaluate(`document.querySelector('#gen-out span.new').textContent.length`);
check('generate: auto-run produced text', genLen > 40, `${genLen} chars`);

// Attention maps: one canvas per layer x head.
const heads = await evaluate(`document.querySelectorAll('#heads canvas').length`);
check('attention: per-head maps rendered', heads === 8, `${heads} canvases`);
await shot('home');

// User generation with a new prompt.
await evaluate(`(() => {
  const p = document.getElementById('prompt');
  p.value = 'Water is';
  document.getElementById('generate').click();
})()`);
await waitFor(() => evaluate(`document.querySelector('#gen-out span.seed')?.textContent.startsWith('Water is')`), 'custom generation', 120000);
check('generate: custom prompt honored', true);
await shot('generate');

// Tokenizer viewer.
await evaluate(`(() => {
  const t = document.getElementById('tok-in');
  t.value = 'tokenization';
  t.dispatchEvent(new Event('input'));
})()`);
const toks = await evaluate(`document.querySelectorAll('#tok-out .tok').length`);
check('tokenizer: chips rendered', toks >= 1 && toks <= 12, `${toks} chips`);

check('no page exceptions', consoleErrors.length === 0, consoleErrors[0] ?? '');

const failed = checks.filter(c => !c.ok);
console.log(`\n${checks.length - failed.length} passed, ${failed.length} failed`);
cleanup(failed.length ? 1 : 0);
