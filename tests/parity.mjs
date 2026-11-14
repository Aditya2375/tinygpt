// Parity check: JS engine vs Java reference logits on a fixed prompt.
// Usage: node tests/parity.mjs   (after: java gpt.train.DumpLogits > /tmp/java-logits.txt)
import { readFileSync } from 'node:fs';
import { loadWeights, forward, makeTokenizer } from '../web/src/engine.js';

const tokJson = JSON.parse(readFileSync('data/out/tokenizer.json', 'utf8'));
const wbuf = readFileSync('data/out/weights.bin');
const f32 = new Float32Array(wbuf.buffer, wbuf.byteOffset, wbuf.length / 4);
const meta = JSON.parse(readFileSync('data/out/train.json', 'utf8'));
const cfg = meta.config;
const w = loadWeights(f32, cfg);
const bpe = makeTokenizer(tokJson.merges, tokJson.vocabSize);
const toks = bpe.encode('The city was');
const { logits } = forward(w, cfg, toks);
const V = cfg.vocab, T = toks.length, off = (T - 1) * V;
let norm = 0, max = 0;
for (let i = 0; i < T * V; i++) { const v = logits[i]; norm += v * v; if (Math.abs(v) > max) max = Math.abs(v); }
console.log('tokens=' + JSON.stringify(toks));
console.log('norm=' + Math.sqrt(norm).toFixed(9) + ' max=' + max.toFixed(9));
const last = [];
for (let j = 0; j < 16; j++) last.push(Number(logits[off + j].toPrecision(9)));
console.log('last16=' + last.join(','));

// Pinned Java reference (gpt.train.DumpLogits on the committed weights.bin).
// The JS engine must reproduce these to within float32 print tolerance.
const EXPECT = {
  tokens: [783, 293, 1025, 119, 331],
  norm: 233.846572495,
  max: 8.787519253,
  last16: [-5.91691879, -5.75413207, -5.84688066, -5.72100985, -5.79057058,
    -6.21312736, -5.98755010, -5.55456887, -6.01828157, -5.94807035,
    -5.51458105, -6.21050412, -6.05555796, -5.75786015, -6.13348528, -5.75523246],
};
const close = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));
let bad = 0;
const eq = (name, a, b) => { if (!close(a, b)) { console.error(`MISMATCH ${name}: js=${a} java=${b}`); bad++; } };
if (JSON.stringify(toks) !== JSON.stringify(EXPECT.tokens)) { console.error('MISMATCH tokens'); bad++; }
eq('norm', Math.sqrt(norm), EXPECT.norm);
eq('max', max, EXPECT.max);
for (let j = 0; j < 16; j++) eq(`last16[${j}]`, Number(logits[off + j].toPrecision(9)), EXPECT.last16[j]);
if (bad) { console.error(`parity: ${bad} mismatches`); process.exit(1); }
console.log('parity: JS matches Java reference (norm ' + EXPECT.norm + ')');
