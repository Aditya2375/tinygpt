// Builds docs/ from web/src + the trained artifacts. Works from a deleted
// docs/: fonts and app code are source assets copied fresh every time.
import { cpSync, mkdirSync, rmSync, existsSync, writeFileSync } from 'node:fs';

const SRC = 'web/src';
const OUT = 'docs';
const DATA = 'data/out';

rmSync(OUT, { recursive: true, force: true });
mkdirSync(OUT, { recursive: true });
cpSync(SRC, OUT, { recursive: true });
mkdirSync(`${OUT}/data`, { recursive: true });
for (const f of ['weights.bin', 'train.json', 'tokenizer.json']) {
  if (!existsSync(`${DATA}/${f}`)) {
    console.error('[web] MISSING ' + DATA + '/' + f + ' (run training first)');
    process.exit(1);
  }
  cpSync(`${DATA}/${f}`, `${OUT}/data/${f}`);
}
writeFileSync(`${OUT}/.nojekyll`, '');
console.log('[web] docs/ built');
