# tinygpt

A small GPT built and trained from scratch - tokenizer, multi-head causal
attention, the backward pass, Adam, the training loop - in plain Java, with
the trained weights running live in the browser through a parity-checked
JavaScript port of the same forward pass.

**Live demo:** https://aditya2375.github.io/tinygpt/

## What's inside

- `src/gpt/token/Bpe.java` - byte-level byte-pair encoding: trainer, an
  O(n log n) encoder with a priority queue over pairs, and decoding.
  1,200 merges on top of 256 byte tokens (vocab 1,456).
- `src/gpt/model/Gpt.java` - the network: token + positional embeddings,
  pre-LN transformer blocks, causal multi-head attention, GELU MLP, tied
  output head, and a hand-derived backward pass. Adam with gradient
  clipping. No libraries anywhere.
- `src/gpt/train/Train.java` - the training loop: warmup + cosine LR
  schedule, checkpointed state so long runs survive restarts, and the
  artifact writer (`weights.bin`, `train.json`, `tokenizer.json`).
- `web/src/engine.js` - a forward-only JavaScript port. `tests/parity.mjs`
  pins it against the Java reference logits on the committed weights.
- `web/src/` - the site: training-loss curve, checkpoint samples, live
  in-browser sampling with a temperature slider, per-head attention maps,
  and a tokenizer viewer.

## The model

| | |
|---|---|
| parameters | 197,376 |
| architecture | 2 layers x 4 heads, d_model 64, context 64 |
| vocabulary | 1,456 byte-level BPE tokens |
| training data | 705,457 tokens of Simple English Wikipedia |
| steps | 900 (batch 16, Adam, cosine LR 3e-3 to 3e-4) |
| final loss | train 5.467, validation 5.446 |

## Honest expectations

A 197k-parameter model trained on 900 steps does not write essays. It
produces word salad with real texture - plausible syllables, frequent words,
local rhythm - and the site says so. The interesting part is that every
piece works and is visible: the loss falls, the checkpoint samples improve
in stages, and the attention maps show what each head learned to look at.

## Run it

```sh
make test    # unit tests (incl. finite-difference gradient check) + JS/Java parity
make web     # pack docs/ from web/src + data/out
node tests/browser-smoke.mjs   # drives the packed site in headless Chrome
```

Requires JDK 21+ and Node 18+. The site is fully static: `docs/` after
`make web`, or the live link above.

## Retraining

The corpus comes from a Simple English Wikipedia dump, cleaned by
`tools/clean_corpus.py` (the raw dump and `data/corpus.txt` are not
committed - 11 MB of text). With `data/corpus.txt` in place:

```sh
make train   # fresh 900-step run; resumable in slices: gpt.train.Train data/corpus.txt data/out <steps> 16
```

`data/out/` holds the committed trained artifacts; `data/out/state/` (not
committed) is the resumable trainer state.

## Design

`DESIGN.md` documents the design system: type scale, palette, spacing,
motion rules, and font licenses.

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache License 2.0](LICENSE-APACHE) at your option.
