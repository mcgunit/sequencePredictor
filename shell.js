// The few pages the server still renders itself after the switch to the app (24 Oct 2026): sign-in (auth.js), "not
// found" and "server error" (switchover.js). One card on the app's background, in the app's look: its colour tokens
// (web/src/styles.css), IBM Plex Sans, and the reader's Light / Dark / Auto choice from the app's colour switch, which
// the app keeps in this origin's localStorage (web/src/theme.ts).

const CSS = `
  :root { --bg: #f4f6f9; --card: #ffffff; --fg: #1b2330; --muted: #6a7482; --line: #e3e8ee; --accent: #2563b8; --act-fg: #ffffff;
    --down: #c53b3b; --down-bg: #fbeaea; --shadow: 0 1px 2px rgba(16, 24, 40, 0.05), 0 1px 3px rgba(16, 24, 40, 0.04);
    --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif; color-scheme: light; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg: #0f141a; --card: #171d25; --fg: #e5e9ef; --muted: #98a2b0;
    --line: #262e39; --accent: #6aa3e6; --act-fg: #0f141a; --down: #ef6b6b; --down-bg: #3a1d1e; --shadow: none; color-scheme: dark; } }
  :root[data-theme="dark"] { --bg: #0f141a; --card: #171d25; --fg: #e5e9ef; --muted: #98a2b0; --line: #262e39; --accent: #6aa3e6;
    --act-fg: #0f141a; --down: #ef6b6b; --down-bg: #3a1d1e; --shadow: none; color-scheme: dark; }
  *, *::before, *::after { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; padding: 24px 16px; background: var(--bg); color: var(--fg);
    font-family: var(--sans); font-size: 14px; line-height: 1.45; }
  main { width: 100%; max-width: 360px; background: var(--card); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); padding: 26px 24px 24px; }
  h1 { margin: 0; font-size: 22px; font-weight: 600; letter-spacing: -0.01em; }
  .sub { margin: 2px 0 8px; color: var(--muted); font-size: 12px; }
  label { display: grid; gap: 4px; margin-top: 14px; font-size: 12px; font-weight: 500; }
  input { font: inherit; font-size: 16px; padding: 8px 10px; min-height: 40px; width: 100%; min-width: 0; color: var(--fg); background: var(--bg);
    border: 1px solid var(--line); border-radius: 8px; }
  input:focus { outline: 2px solid var(--accent); outline-offset: 1px; border-color: var(--accent); }
  button { width: 100%; margin-top: 20px; min-height: 40px; font: inherit; font-weight: 500; color: var(--act-fg); background: var(--accent);
    border: 1px solid var(--accent); border-radius: 8px; cursor: pointer; }
  button:hover { filter: brightness(1.08); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .err { margin: 14px 0 0; padding: 8px 10px; border-radius: 8px; color: var(--down); background: var(--down-bg); font-size: 12px; }
  .foot { margin: 16px 0 0; color: var(--muted); font-size: 11px; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  p { margin: 10px 0 0; }
  code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.92em; }
`;

// title: the tab's first word(s) ("Sign in" -> "Sign in · Predictor"); body: trusted HTML (escape what you put in)
function page(title, body) {
  return `<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="color-scheme" content="light dark">
  <script>try { var t = localStorage.getItem('predictor.theme'); if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t); } catch (e) {}</script>
  <title>${title} · Predictor</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap">
  <style>${CSS}</style></head><body><main>${body}
  </main></body></html>`;
}

module.exports = { page };
