// Forward-only port of the Java GPT for in-browser sampling and attention
// inspection. Same math, same layout as gpt.model.Gpt; weights load from the
// float32 weights.bin the trainer writes, in the Java registration order:
// tokEmb, posEmb, lnF, then per layer [ln1, qkvW, qkvB, projW, projB, ln2,
// fc1W, fc1B, fc2W, fc2B].

export function loadWeights(f32, cfg) {
  const d = cfg.dModel, V = cfg.vocab, L = cfg.nLayer, T = cfg.ctx;
  let o = 0;
  const take = (n) => { const a = f32.slice(o, o + n); o += n; return a; };
  const w = {
    tokEmb: take(V * d),
    posEmb: take(T * d),
    lnF: take(2 * d),
    blocks: [],
  };
  for (let l = 0; l < L; l++) {
    w.blocks.push({
      ln1: take(2 * d), qkvW: take(d * 3 * d), qkvB: take(3 * d),
      projW: take(d * d), projB: take(d),
      ln2: take(2 * d), fc1W: take(d * 4 * d), fc1B: take(4 * d),
      fc2W: take(4 * d * d), fc2B: take(d),
    });
  }
  if (o !== f32.length) throw new Error('weights size mismatch: used ' + o + ' of ' + f32.length);
  return w;
}

function layernorm(x, ln, T, d) {
  const out = new Float64Array(T * d);
  for (let t = 0; t < T; t++) {
    const o = t * d;
    let mean = 0;
    for (let i = 0; i < d; i++) mean += x[o + i];
    mean /= d;
    let varr = 0;
    for (let i = 0; i < d; i++) { const z = x[o + i] - mean; varr += z * z; }
    const rstd = 1 / Math.sqrt(varr / d + 1e-5);
    for (let i = 0; i < d; i++) out[o + i] = (x[o + i] - mean) * rstd * ln[i] + ln[d + i];
  }
  return out;
}

function linear(x, w, b, T, fanIn, fanOut) {
  const out = new Float64Array(T * fanOut);
  for (let t = 0; t < T; t++) {
    const xo = t * fanIn, oo = t * fanOut;
    if (b) for (let j = 0; j < fanOut; j++) out[oo + j] = b[j];
    for (let i = 0; i < fanIn; i++) {
      const xi = x[xo + i];
      if (xi === 0) continue;
      const wo = i * fanOut;
      for (let j = 0; j < fanOut; j++) out[oo + j] += xi * w[wo + j];
    }
  }
  return out;
}

function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));
}

// Forward pass. Returns {logits: Float64Array[T*V], attn: per-layer per-head
// [T*T] probs} - attn is kept so the demo can draw live attention maps.
// With lastOnly=true only the last position's logits are computed (sampling
// needs nothing else); other rows stay zero.
export function forward(w, cfg, tokens, lastOnly = false) {
  const T = tokens.length, d = cfg.dModel, H = cfg.nHead, dh = d / H, V = cfg.vocab;
  let cur = new Float64Array(T * d);
  for (let t = 0; t < T; t++)
    for (let i = 0; i < d; i++)
      cur[t * d + i] = w.tokEmb[tokens[t] * d + i] + w.posEmb[t * d + i];

  const attn = [];
  for (let l = 0; l < cfg.nLayer; l++) {
    const b = w.blocks[l];
    const ln1o = layernorm(cur, b.ln1, T, d);
    const qkv = linear(ln1o, b.qkvW, b.qkvB, T, d, 3 * d);
    const attnAll = new Float64Array(T * d);
    const layerAttn = [];
    for (let h = 0; h < H; h++) {
      const probs = new Float64Array(T * T);
      const scale = 1 / Math.sqrt(dh);
      for (let t = 0; t < T; t++) {
        let max = -Infinity;
        for (let s = 0; s <= t; s++) {
          let dot = 0;
          for (let i = 0; i < dh; i++) dot += qkv[t * 3 * d + h * dh + i] * qkv[s * 3 * d + d + h * dh + i];
          probs[t * T + s] = dot * scale;
          if (probs[t * T + s] > max) max = probs[t * T + s];
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) { probs[t * T + s] = Math.exp(probs[t * T + s] - max); sum += probs[t * T + s]; }
        for (let s = 0; s <= t; s++) probs[t * T + s] /= sum;
      }
      layerAttn.push(probs);
      for (let t = 0; t < T; t++)
        for (let i = 0; i < dh; i++) {
          let s = 0;
          for (let u = 0; u <= t; u++) s += probs[t * T + u] * qkv[u * 3 * d + 2 * d + h * dh + i];
          attnAll[t * d + h * dh + i] = s;
        }
    }
    attn.push(layerAttn);
    const projOut = linear(attnAll, b.projW, b.projB, T, d, d);
    const res1 = new Float64Array(T * d);
    for (let i = 0; i < T * d; i++) res1[i] = cur[i] + projOut[i];
    const ln2o = layernorm(res1, b.ln2, T, d);
    const fc1 = linear(ln2o, b.fc1W, b.fc1B, T, d, 4 * d);
    for (let i = 0; i < fc1.length; i++) fc1[i] = gelu(fc1[i]);
    const fc2 = linear(fc1, b.fc2W, b.fc2B, T, 4 * d, d);
    const nxt = new Float64Array(T * d);
    for (let i = 0; i < T * d; i++) nxt[i] = res1[i] + fc2[i];
    cur = nxt;
  }
  const xf = layernorm(cur, w.lnF, T, d);
  const logits = new Float64Array(T * V);
  const t0 = lastOnly ? T - 1 : 0;
  for (let t = t0; t < T; t++)
    for (let j = 0; j < V; j++) {
      let s = 0;
      for (let i = 0; i < d; i++) s += xf[t * d + i] * w.tokEmb[j * d + i];
      logits[t * V + j] = s;
    }
  return { logits, attn };
}

// Byte-level BPE decode/encode matching the Java tokenizer.
export function makeTokenizer(merges, vocabSize) {
  const pairToNew = new Map();
  const idToStr = new Array(vocabSize);
  for (let i = 0; i < 256; i++) idToStr[i] = String.fromCharCode(i);
  for (const [a, b, id] of merges) {
    pairToNew.set(a * 4294967296 + b, id);
    idToStr[id] = idToStr[a] + idToStr[b];
  }
  const encode = (text) => {
    const bytes = new TextEncoder().encode(text);
    const val = Array.from(bytes);
    const prev = val.map((_, i) => i - 1), next = val.map((_, i) => i + 1);
    const dead = new Array(val.length).fill(false);
    next[val.length - 1] = -1;
    // Simple O(merges x len) is fine for prompt-sized inputs.
    for (const [a, b, id] of merges) {
      for (let i = 0; i < val.length; i++) {
        if (dead[i] || val[i] !== a) continue;
        const n = next[i];
        if (n >= 0 && val[n] === b) {
          val[i] = id; dead[n] = true;
          const after = next[n];
          next[i] = after;
          if (after >= 0) prev[after] = i;
        }
      }
    }
    const out = [];
    for (let i = 0; i < val.length; i++) if (!dead[i]) out.push(val[i]);
    return out;
  };
  const decode = (ids) => {
    const s = ids.map((id) => idToStr[id]).join('');
    const bytes = Uint8Array.from(s, (ch) => ch.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  };
  return { encode, decode, vocabSize };
}

// Top-k sampling from the last position's logits.
export function sampleNext(logits, T, V, temperature, topK, rand) {
  const off = (T - 1) * V;
  if (temperature <= 0) {
    let bi = 0;
    for (let i = 1; i < V; i++) if (logits[off + i] > logits[off + bi]) bi = i;
    return bi;
  }
  const idx = Array.from({ length: V }, (_, i) => i);
  idx.sort((a, b) => logits[off + b] - logits[off + a]);
  const k = Math.min(topK, V);
  const max = logits[off + idx[0]];
  const p = new Float64Array(k);
  let sum = 0;
  for (let i = 0; i < k; i++) { p[i] = Math.exp((logits[off + idx[i]] - max) / temperature); sum += p[i]; }
  let r = rand() * sum, acc = 0;
  for (let i = 0; i < k; i++) { acc += p[i]; if (r <= acc) return idx[i]; }
  return idx[k - 1];
}
