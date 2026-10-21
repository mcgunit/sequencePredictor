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

| Path                | Page                                                                  |
| ------------------- | --------------------------------------------------------------------- |
| `/`                 | redirects to `/crypto`                                                |
| `/crypto`           | `MarketPage` for the crypto market                                    |
| `/shares`           | `MarketPage` for the shares market                                    |
| `/admin`            | redirects to `/admin/controls`                                        |
| `/admin/controls`   | the four Sunday controls (Q0, Q2, feature control, market rows)       |
| `/admin/models`     | Best model per game; a game's drawer holds its ranking                |
| `/admin/phase-shift`| the tracked lag peaks per game                                        |
| `/admin/randomness` | the Randomness watch                                                  |
| other               | redirects to `/crypto` (`/admin/...` to `/admin/controls`)            |

The admin routes render only for the administrator (`AdminGuard`, from `/api/me`); the API
answers a JSON 403 anyway. Their drawers and the All / Top 5 switch live in the URL
(`?game=lotto`, `?row=...`, `?scope=top`), so a reading can link to one game.

The sidebar (`components/Sidebar.tsx`) is data: the users see Lottery, Crypto, Shares and
Council; the administrator also the Controls, Models and System groups. Server pages are plain
anchors, marked ↗ while the app has no page of its own for them.

## API

All calls are same-origin and typed in `src/api.ts`:

- `GET /api/me` -> `{ user: { name, admin, open } | null }` (the sidebar and the admin guard; open = open access, no sign-in)
- `GET /api/markets/:market` -> the whole page model (see the `MarketView` type)
- `GET /api/quotes/:market` -> `{ at, quotes: { [symbol]: number } }`, fetched after the page
  model and refreshed every 60 s while the tab is visible

- `GET /api/admin/...` -> the admin views (`getAdminControls`, `getAdminModels`,
  `getAdminPhaseShift`, `getAdminRandomness` and their per-game getters); `admin.js` documents
  the shapes

Every field may be null or empty; `getMarket()` and the admin getters fill the containers and the format helpers in
`src/format.ts` answer "—" for anything missing, so the page always renders.

## Layout

```
src/
  api.ts                  typed fetch helpers and the API types
  format.ts               numbers, percentages, dates (Europe/Brussels)
  styles.css              the one global stylesheet (light/dark tokens)
  hooks/useEChart.ts      ECharts binding (tree-shaken core, resize, theme change)
  hooks/useQuotes.ts      live quotes polling
  hooks/useMe.ts          who is reading (/api/me, asked once)
  hooks/useLoad.ts        one fetch per change of its keys (the admin pages)
  hooks/useUrlState.ts    the All / Top 5 switch and the drawers, kept in the URL
  components/             sidebar, KPI tile, status strip, open/closed positions, trades log, charts
  components/admin/       card, chip, drawer, DotStrip (inline SVG), the switch, guard and load states
  pages/MarketPage.tsx    loads the market and lays out the page
  pages/admin/            Controls, Models, Phase-shift and Randomness pages and their drawers
```

There is no dev server script on purpose: the API lives in the Express app, so the quickest
loop is `npm run build` and a reload of `/app/crypto`.
