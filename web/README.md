# web/ - the markets UI

A Vite + React 18 + TypeScript single-page app for the Crypto and Shares pages. It is built
to `web/dist` and served by `server.js` under the URL prefix `/app`; nothing in here runs
its own server.

## Build

```sh
cd web
npm install          # once (Node 20)
npm run build        # tsc --noEmit, then vite build -> web/dist
npm run typecheck    # the TypeScript pass alone
```

`dist/` holds `index.html` plus hashed assets under `dist/assets/`. Every asset URL starts
with `/app/` (`base: '/app/'` in `vite.config.ts`), and the router uses the same basename, so
the server has to:

- serve `web/dist` statically at `/app` (e.g. `app.use('/app', express.static('web/dist'))`),
- answer every other `/app/*` path with `web/dist/index.html` so deep links and reloads work.

Note that the repository's root `.gitignore` ignores `web/dist/` and `web/node_modules/`:
the build output is not committed, so run `npm run build` on the machine that serves it.

## Routes (basename `/app`)

| Path      | Page                                        |
| --------- | ------------------------------------------- |
| `/`       | redirects to `/crypto`                      |
| `/crypto` | `MarketPage` for the crypto market          |
| `/shares` | `MarketPage` for the shares market          |
| other     | redirects to `/crypto`                      |

The sidebar also links to the existing server pages (`/lottery`, `/council`, `/markets/...`,
`/database...`, `/jobs`, `/services`, `/users`, `/whats-new`) as plain anchors.

## API

All calls are same-origin and typed in `src/api.ts`:

- `GET /api/me` -> `{ user: { name, admin } | null }` (the sidebar footer)
- `GET /api/markets/:market` -> the whole page model (see the `MarketView` type)
- `GET /api/quotes/:market` -> `{ at, quotes: { [symbol]: number } }`, fetched after the page
  model and refreshed every 60 s while the tab is visible

Every field may be null or empty; `getMarket()` fills the containers and the format helpers in
`src/format.ts` answer "—" for anything missing, so the page always renders.

## Layout

```
src/
  api.ts                  typed fetch helpers and the API types
  format.ts               numbers, percentages, dates (Europe/Brussels)
  styles.css              the one global stylesheet (light/dark tokens)
  hooks/useEChart.ts      ECharts binding (tree-shaken core, resize, theme change)
  hooks/useQuotes.ts      live quotes polling
  components/             sidebar, status strip, open/closed positions, trades log, charts
  pages/MarketPage.tsx    loads the market and lays out the page
```

There is no dev server script on purpose: the API lives in the Express app, so the quickest
loop is `npm run build` and a reload of `/app/crypto`.
