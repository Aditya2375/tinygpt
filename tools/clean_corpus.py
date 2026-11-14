import re
src = open('data/corpus.txt', encoding='utf-8').read()
# Parsoid data-mw JSON attribute blobs: from "}]] (or similar) up to the closing }'> marker
src = re.sub(r'"\}\]\][^>]*?\}\'>', ' ', src, flags=re.S)
src = re.sub(r',\{"template":.*?\}\]\}\'>', ' ', src, flags=re.S)
# any remaining data-mw style attribute spill
src = re.sub(r'\{"[a-z]+":\{.*?\}\}\]\}', ' ', src, flags=re.S)
# footers
src = re.sub(r'\s*Retrieved from "https?://[^"]*"\s*', '\n', src)
src = re.sub(r'https?://\S+', ' ', src)
# refs, tags, templates, links (as before)
src = re.sub(r'<ref[^>]*/>', ' ', src, flags=re.I)
src = re.sub(r'<ref[^>]*>.*?</ref>', ' ', src, flags=re.I | re.S)
src = re.sub(r'</?[a-z][^>]*>', ' ', src)
out, depth, i = [], 0, 0
while i < len(src):
    if src.startswith('{{', i): depth += 1; i += 2; continue
    if src.startswith('}}', i) and depth: depth -= 1; i += 2; continue
    if depth == 0: out.append(src[i])
    i += 1
src = ''.join(out)
src = re.sub(r'\[\[(?:File|Image|Category):[^\]]*\]\]', ' ', src, flags=re.I)
src = re.sub(r'\[\[[^\]|]*\|([^\]]*)\]\]', r'\1', src)
src = re.sub(r'\[\[([^\]]*)\]\]', r'\1', src)
src = re.sub(r"'{2,}", '', src)
src = re.sub(r'&[a-z]+;', ' ', src)
src = re.sub(r'[ \t]+', ' ', src)
# drop lines that are mostly symbols or too short to be prose
lines = []
for ln in src.split('\n'):
    ln = ln.strip()
    if len(ln) < 60: continue
    alpha = sum(c.isalpha() or c.isspace() for c in ln)
    if alpha / len(ln) < 0.75: continue
    if '"wt"' in ln or "'>" in ln or 'template' in ln.lower()[:40]: continue
    lines.append(ln)
open('data/corpus.txt', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
bad = sum(1 for ln in lines if '"wt"' in ln or "'>" in ln)
print('lines kept:', len(lines), 'bad:', bad, 'chars:', sum(len(l) for l in lines))
