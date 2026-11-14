import { loadWeights, forward, makeTokenizer, sampleNext } from './engine.js';

const $ = (id) => document.getElementById(id);
const fmt = (n) => n.toLocaleString('en-US');

async function main() {
  const [meta, tokJson, wbuf] = await Promise.all([
    fetch('data/train.json').then((r) => r.json()),
    fetch('data/tokenizer.json').then((r) => r.json()),
    fetch('data/weights.bin').then((r) => r.arrayBuffer()),
  ]);
  const cfg = meta.config;
  const weights = loadWeights(new Float32Array(wbuf), cfg);
  const bpe = makeTokenizer(tokJson.merges, tokJson.vocabSize);

  renderStats(meta, cfg);
  drawLossChart(meta.curve);
  renderSamples(meta.samples);
  wireTokenizer(bpe);

  let lastRun = null;
  async function generate() {
    const prompt = $('prompt').value || 'The ';
    const temp = parseFloat($('sl-temp').value);
    $('sl-temp-v').textContent = temp.toFixed(2);
    $('generate').disabled = true;
    const out = $('gen-out');
    out.textContent = 'Thinking…';
    await new Promise((r) => setTimeout(r, 20)); // let the UI paint
    let toks = bpe.encode(prompt).slice(-cfg.ctx);
    const seedToks = toks.slice();
    const made = [];
    const maxNew = 96;
    for (let n = 0; n < maxNew; n++) {
      const { logits, attn } = forward(weights, cfg, toks, true);
      const nextTok = sampleNext(logits, toks.length, cfg.vocab, temp, 40, Math.random);
      toks.push(nextTok);
      made.push(nextTok);
      if (toks.length > cfg.ctx) toks = toks.slice(toks.length - cfg.ctx);
      if (n === maxNew - 1) lastRun = { tokens: toks.slice(), attn };
    }
    out.textContent = '';
    const s1 = document.createElement('span'); s1.className = 'seed'; s1.textContent = bpe.decode(seedToks);
    const s2 = document.createElement('span'); s2.className = 'new'; s2.textContent = bpe.decode(made);
    out.append(s1, s2);
    drawAttention(lastRun, bpe);
    $('generate').disabled = false;
  }
  $('generate').addEventListener('click', generate);
  $('sl-temp').addEventListener('input', () => { $('sl-temp-v').textContent = parseFloat($('sl-temp').value).toFixed(2); });
  $('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter') generate(); });
  await generate();
}

function renderStats(meta, cfg) {
  const stats = [
    ['parameters', fmt(cfg.params)],
    ['vocabulary', fmt(cfg.vocab) + ' tokens'],
    ['context window', cfg.ctx + ' tokens'],
    ['layers × heads', cfg.nLayer + ' × ' + cfg.nHead],
    ['training tokens', fmt(meta.trainTokens)],
    ['steps', fmt(meta.steps)],
    ['final train loss', meta.trainLoss.toFixed(3)],
    ['validation loss', meta.valLoss.toFixed(3)],
  ];
  const el = $('stats');
  for (const [k, v] of stats) {
    const d = document.createElement('div');
    d.className = 'stat';
    const kk = document.createElement('div'); kk.className = 'k'; kk.textContent = k;
    const vv = document.createElement('div'); vv.className = 'v'; vv.textContent = v;
    d.append(kk, vv);
    el.appendChild(d);
  }
}

function drawLossChart(curve) {
  const cv = $('chart-loss');
  const ctx = cv.getContext('2d');
  const W = (cv.width = cv.clientWidth * devicePixelRatio || 800);
  const H = (cv.height = 280 * devicePixelRatio);
  ctx.scale(devicePixelRatio, devicePixelRatio);
  const w = W / devicePixelRatio, h = 280, padL = 44, padB = 26, padT = 12, padR = 10;
  const min = Math.min(...curve) * 0.98, max = Math.max(...curve) * 1.02;
  const X = (i) => padL + (i / (curve.length - 1)) * (w - padL - padR);
  const Y = (v) => padT + (1 - (v - min) / (max - min)) * (h - padT - padB);
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.strokeStyle = '#262a32';
  ctx.fillStyle = '#8f8b82';
  for (let g = 0; g <= 4; g++) {
    const v = min + ((max - min) * g) / 4;
    const y = Y(v);
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(w - padR, y); ctx.stroke();
    ctx.fillText(v.toFixed(2), 6, y + 4);
  }
  ctx.fillText('step ' + (curve.length * 5), w - padR - 52, h - 8);
  ctx.fillText('step 0', padL, h - 8);
  ctx.strokeStyle = '#d8ff4a';
  ctx.lineWidth = 2;
  ctx.beginPath();
  curve.forEach((v, i) => { const x = X(i), y = Y(v); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
  // hide the val legend entry until we have a val curve
  const legend = document.querySelector('.legend .sw-val');
  if (legend) legend.closest('span').style.display = 'none';
}

function renderSamples(samples) {
  const el = $('samples');
  for (const s of samples) {
    const idx = s.indexOf(': ');
    const step = idx >= 0 ? s.slice(0, idx) : 'sample';
    const text = idx >= 0 ? s.slice(idx + 2) : s;
    const d = document.createElement('div');
    d.className = 'sample';
    const st = document.createElement('div'); st.className = 'step'; st.textContent = step;
    const tx = document.createElement('div'); tx.className = 'text'; tx.textContent = text;
    d.append(st, tx);
    el.appendChild(d);
  }
}

function drawAttention(run, bpe) {
  const host = $('heads');
  host.textContent = '';
  if (!run) return;
  const { tokens, attn } = run;
  const T = tokens.length;
  const win = Math.min(T, 32);
  const off = T - win;
  attn.forEach((layer, l) => {
    layer.forEach((probs, h) => {
      const box = document.createElement('div');
      box.className = 'head';
      const title = document.createElement('h3');
      title.textContent = 'layer ' + (l + 1) + ' · head ' + (h + 1);
      const cv = document.createElement('canvas');
      const cell = 7;
      cv.width = win * cell; cv.height = win * cell;
      const c = cv.getContext('2d');
      for (let t = 0; t < win; t++) {
        for (let s = 0; s <= t; s++) {
          const p = probs[(t + off) * T + (s + off)];
          const a = Math.min(1, p * win * 0.9);
          c.fillStyle = 'rgba(216,255,74,' + (0.04 + 0.96 * a).toFixed(3) + ')';
          c.fillRect(s * cell, t * cell, cell - 1, cell - 1);
        }
      }
      const first = bpe.decode(tokens.slice(off, off + 4)).replace(/\n/g, '⏎');
      const last = bpe.decode(tokens.slice(T - 4)).replace(/\n/g, '⏎');
      cv.title = 'tokens ' + first + ' … ' + last;
      box.append(title, cv);
      host.appendChild(box);
    });
  });
}

function wireTokenizer(bpe) {
  const render = () => {
    const ids = bpe.encode($('tok-in').value);
    const host = $('tok-out');
    host.textContent = '';
    for (const id of ids) {
      const chip = document.createElement('span');
      chip.className = 'tok';
      const idEl = document.createElement('span'); idEl.className = 'id'; idEl.textContent = String(id);
      const txt = document.createElement('span');
      txt.textContent = bpe.decode([id]).replace(/ /g, '␣').replace(/\n/g, '⏎');
      chip.append(idEl, txt);
      host.appendChild(chip);
    }
  };
  $('tok-in').addEventListener('input', render);
  render();
}

main().catch((e) => {
  const out = $('gen-out');
  if (out) out.textContent = 'Failed to load model data: ' + e.message;
});
