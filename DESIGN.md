# Design system - tinygpt

Written before UI work, per the design-resources rule. Patterns referenced from
getdesign.md / awesome-design-md / Refero Styles conventions (type-first,
single accent, hairline structure). No component code copied; everything below
is hand-written CSS.

## Type
- Display: system UI stack, 600 weight - clamp(40px, 6vw, 72px), line-height 1.02, letter-spacing -0.03em.
- Body: system UI stack, 400/500 - 17px / 1.55, `text-wrap: pretty`.
- Data & labels: JetBrains Mono 400 (SIL OFL 1.1) - 11-14px; eyebrows uppercase, letter-spacing 0.08em.
- Minimums: body >= 17px, mono labels >= 11px. No text below 11px.

## Palette (dark)
- ink #0c0d10 (page), ink-2 #131519 (panel), ink-3 #1a1d23
- line #262a32 (hairlines only - structure comes from dividers, not boxes)
- paper #e8e4dc, paper-dim #a8a49a, paper-faint #8f8b82 (AA on ink)
- accent #d8ff4a chartreuse - one accent, reserved for data, the primary CTA, and generated text
- warm #e07856 - secondary data series only
- Forbidden: gradients (especially purple-blue), colored glow shadows, stock icon grids.

## Spacing & shape
- 8px base unit; panel padding 18-22px; section rhythm 28px; max content width 1180px.
- Radius 10-12px on panels, 999px on pills/buttons.

## Motion
- Hover transitions only, 200ms var(--ease-out). No scroll-triggered animation,
  no autoplaying motion, no parallax. Canvas charts render statically.
  Generation output appears as plain text. Reduced-motion safe by construction.

## Rules enforced in this build
- No nested cards: panels contain hairline-divided rows, not boxed items.
- One accent color per view; accent never decorates, always marks data or action.
- All DOM text written via textContent; no innerHTML with external data.
- Honest labels: model size, loss numbers and sample quality stated as-is.

## Asset licenses
- JetBrains Mono: SIL Open Font License 1.1. No other webfonts bundled; display and body text use the system UI stack.
- No illustrations or stock assets used.
