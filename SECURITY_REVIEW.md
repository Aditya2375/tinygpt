# Security review - tinygpt

Item-by-item pass over the pre-ship checklist, with evidence. Statuses: PASS
(checked, evidence below), N/A (the thing the item protects does not exist in
this project - stated explicitly, not skipped silently), TODO (must happen
before or right after publication).

Architecture context for every item below: tinygpt is a fully static site
plus an offline Java trainer. Training runs on a developer machine; only its
output (weights.bin, train.json, tokenizer.json) ships. The served site has
no backend, no database, no auth, no accounts, no cookies, no analytics, no
API keys, no forms. The interactive inputs are two free-text fields (the
sampling prompt and the tokenizer playground), one range slider and one
button. Both text fields are read as strings, encoded to token ids, and only
ever written back to the page via textContent - they never reach HTML,
fetch, storage or the network. The page fetches three literal same-origin
files and nothing else; CSP allows connect-src 'self'. All state lives in
page memory (no localStorage, no URL mutation).

## Pre-ship routine

1. **Read the diff** - PASS. All source (src/, web/src/, tests/, tools/) was
   reviewed line by line. docs/ is built from web/src/ + data/out/ by
   build-web.mjs; the CI drift-check job deletes docs/, rebuilds, and fails
   on any git diff - so the committed site is provably the output of the
   committed source and the committed trained artifacts.
2. **gitleaks detect** - PASS. Run locally over the working tree with the
   sha256-verified 8.24.3 binary and the repo's .gitleaks.toml: zero leaks.
   CI job `supply-chain` runs the full-history scan on every push (the
   hosted action breaks on root-commit ranges, hence the pinned binary).
3. **Dependency audit** - PASS. Zero npm dependencies - no package.json, no
   lockfile. Build script and tests use only the Node standard library. The
   Java side uses only the JDK 21 standard library (javac/java, no
   Maven/Gradle, no jars).
4. **Static scan** - PASS. semgrep with the repo's local rules
   (.github/semgrep-rules.yml, no account) over web/src/, build-web.mjs and
   the test scripts: zero findings. The same risky-sink greps (innerHTML,
   outerHTML, insertAdjacentHTML, document.write, eval, new Function) run
   over web/src/ and the built docs/: zero hits.
5. **Logged-out test** - N/A. There is nothing to log in to. Every byte the
   site serves is public by design; all state is per-tab memory.
6. **MDN Observatory scan** - N/A, verified by hand instead. Observatory
   scans hosts, not paths: aditya2375.github.io returns 404 at the apex.
   Verified directly instead: `curl -sI https://aditya2375.github.io/tinygpt/`
   must return 200 with strict-transport-security (GitHub Pages default for
   *.github.io, curl-verified on the sibling sites 2026-11-14); Pages cannot
   send the remaining headers, so CSP is delivered via <meta> (item 17) and
   confirmed present in the served live HTML after publication.
7. **Push public only after checks** - PASS (process). Nothing was pushed
   before this review. Local gates: Java unit tests 7/7 (including a
   finite-difference gradient check on all 115 parameter tensors and an
   overfit-one-batch check), JS/Java parity on the shipped weights (exact,
   pinned reference), headless-Chrome smoke on the packed site 8/8, and
   Impeccable clean on source and the rendered page (`--json` output `[]`).

## Secrets and keys

8. **(1) No secret in code, config, bundle or history** - PASS. The project
   has no secrets to leak: no backends, no APIs, no accounts. gitleaks ran
   over the tree locally and runs in CI on every push. GitHub secret
   scanning + push protection enabled on the repo at creation (free on
   public repos).
9. **(2) Keys meant to be public** - N/A. No keys of any kind ship.
10. **(3) Gitignore secrets before first commit** - PASS. .gitignore covers
    build/, data/corpus.txt, data/out/state/ and smoke-*.png since the
    first commit; no credential-shaped files exist in the project at all.
11. **(4) Rotate anything that ever leaked** - N/A. Nothing ever existed to
    leak.

## Access control and data

12. **(5) RLS on every table** - N/A. No database.
13. **(6) Storage buckets private by default** - N/A. No buckets; nothing
    persists outside the tab.
14. **(7) Authorization on the server** - N/A. No server, no per-user
    resources.
15. **(8) Collect less** - PASS. The site collects nothing: no analytics, no
    telemetry, no cookies; the two text inputs never leave the tab. The
    training corpus is Simple English Wikipedia (public, CC BY-SA) and is
    not committed; the shipped artifacts are model weights, a tokenizer
    merge table and training metrics - no article text. Scanned the shipped
    docs/ tree for email addresses, phone numbers and personal names: none
    present.

## Injection and input handling

16. **(9) No untrusted text into innerHTML/eval/DOM HTML** - PASS. The app
    builds all DOM with textContent/createElement only; the semgrep rules
    no-inner-html and no-eval fail CI on innerHTML, outerHTML,
    insertAdjacentHTML, document.write, eval and new Function. Both
    free-text inputs are treated purely as strings: they are BPE-encoded to
    integer ids; output text is decoded from ids and written via
    textContent. Model output containing markup-looking text renders as
    text, never markup.
17. **(10) Parameterized queries; no shell strings** - N/A. No SQL, no shell
    execution in the served site.
18. **(11) Validate URLs and file paths** - PASS. Data fetches are three
    literal relative URLs (data/train.json, data/tokenizer.json,
    data/weights.bin) - no string built from input ever reaches fetch; the
    semgrep no-unsafe-url-fetch rule fails CI on anything else. The one
    external link (source, github.com) is a fixed literal in the HTML.

## Dependencies and supply chain

19. **(12) Every package exists and is the right one** - PASS. No third-party
    packages at all: zero npm dependencies, zero Maven/Gradle artifacts,
    JDK + Node standard libraries only. Nothing to typosquat.
20. **(13) Commit lockfiles and audit** - N/A by construction: no dependency
    manager in play, so no lockfile exists and npm audit has nothing to
    check.
21. **(14) Treat AI config files as code** - PASS. The repo contains no
    AI-assistant instruction files of any kind. No code or README content
    was copied from external repos; everything here is written for this
    project.
22. **(15) Agents never get production access** - N/A. There is no
    production data or environment; the site is static.

## Config, transport and accounts

23. **(16) HTTPS only** - PASS. Pages serves the site over HTTPS with HSTS
    (curl-verified live after publication); "Enforce HTTPS" is on and
    non-disableable for the default aditya2375.github.io domain. The site
    itself makes no http:// requests (grep-verified).
24. **(17) Security headers** - PASS within Pages' limits. CSP via <meta>:
    default-src 'self'; script-src 'self'; style-src 'self'; font-src
    'self'; connect-src 'self'; img-src 'self'; base-uri 'none';
    form-action 'none'; frame-ancestors 'none'. The favicon is an inline
    SVG data: URI (static markup, not fetched content). (frame-ancestors in
    a meta CSP is ignored by browsers - kept as documentation of intent.)
25. **(18) Debug off, errors quiet, source maps off** - PASS. No sourcemaps
    ship (the JS is unminified source by design - it is the demo). A failed
    data load surfaces as one short honest message in the UI, never dumped
    internals.
26. **(19) Login provider, never homemade auth** - N/A. No auth anywhere.
27. **(20) Rate limits and abuse controls** - PASS (client-side). All
    computation runs locally in the visitor's tab (a 96-token generation is
    ~1-2 s of local CPU; the input fields are length-capped). No public
    endpoint exists to abuse.
28. **(21) Fail closed and log the failure** - PASS. A missing or malformed
    data file rejects the load promise and lands in an explicit UI error
    state. loadWeights throws on a size mismatch, so a truncated weights
    file fails loudly, not silently. The CI drift check fails the build on
    any artifact/source divergence in docs/.
29. **(22) Two-factor on accounts** - N/A to this repo's code; account-level
    setting on GitHub is the owner's own checklist item and stays with them.

## Static sites, GitHub Pages and Vercel

30. **(23) Everything in the published folder is public** - PASS. docs/
    contains only the app code, the three trained artifacts and the
    self-hosted fonts. Searched for phone, email, address and personal
    names: none present. Git author identity is the owner's public GitHub
    noreply address only.
31. **(24) No client-side login or hidden admin page** - N/A. No login, no
    admin, no hidden routes.
32. **(25) Vercel secrets** - N/A. Hosted on GitHub Pages, not Vercel; and
    no secrets exist regardless.
33. **(26) Pin or self-host third-party scripts and fonts** - PASS. Zero CDN
    scripts, zero external assets: app code and the four woff2 fonts (Clash
    Display, Satoshi regular/medium, JetBrains Mono - Fontshare/JetBrains
    terms, see DESIGN.md) are self-hosted in docs/. The page works fully
    offline after first load.
34. **(27) Forms go to a service with spam protection** - N/A. No forms; the
    controls post nothing anywhere.
35. **(28) Delete old deployments** - N/A to this new repo; owner's standing
    account hygiene item.

## Android apps (items 29-36, 37-39)

36. **(29-39) APK, manifest, keystore, WebView, relay channel, update
    hashes** - N/A. This project ships no Android app and no update channel.

## Python tools and the pension monitor (items 40-45)

37. **(40-43) Credentials, read-only monitors, log scrubbing, safe
    parsing** - N/A. The one Python tool (tools/clean_corpus.py) is an
    offline corpus cleaner with no credentials, no network and no logs.
38. **(44) GitHub Actions least privilege** - PASS. The CI workflow sets
    `permissions: contents: read` at the top level, uses no secrets beyond
    the automatic GITHUB_TOKEN, and pins every third-party action to a full
    commit SHA (checkout, setup-java, setup-node). gitleaks runs as a
    sha256-verified pinned binary, semgrep from the repo's own local rules.
39. **(45) Alerts to a private channel** - N/A. The project sends no alerts.

## Residual risks accepted on purpose

- **Model output is unfiltered**: a character-level-ish model can emit
  arbitrary byte sequences, including rude words it absorbed from public
  Wikipedia text. Accepted: output is clearly labeled machine-generated
  word salad from a tiny educational model, rendered as inert text, in a
  demo that states exactly what it is.
- **Unminified JS ships by design**: the source is the demo. No debug
  metadata, keys or internal endpoints exist in it, so readability costs
  nothing.
- **Trained artifacts are committed**: weights.bin (790 KB), tokenizer.json
  and train.json ship in git so the site is reproducible and the CI parity
  test pins the JS engine against them. state/ (optimizer internals) is
  gitignored.
- **CI minutes**: the workflow runs on push/PR only; no scheduled runs.
