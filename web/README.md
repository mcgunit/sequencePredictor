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
| `/admin`            | This week: the weekend's runs, today's predictor, the tuners, the controls matrix, since Monday |
| `/admin/tuning`     | Tuning gates: the newest weekend's gate decisions by tuner or by game  |
| `/admin/controls`   | the four Sunday controls (Q0, Q2, feature control, market rows)       |
| `/admin/models`     | Best model per game; a game's drawer holds its ranking                |
| `/admin/phase-shift`| the tracked lag peaks per game                                        |
| `/admin/randomness` | the Randomness watch                                                  |
| `/admin/games`      | the Workbench's game picker                                           |
| `/admin/games/:game`| every stored draw of a game by month (`?scope=top`: each draw's top 5 as ranked before it) |
| `/admin/games/:game/:date` | one draw scored: the result, every row's tickets, the vote, the coming draw on the newest day |
| `/admin/markets/:market` | the market workbench (crypto, shares): the plan and week cards in detail, following the plan, paper books, models, accuracy, prices, day by day, the guard, regimes |
| other               | redirects to `/crypto` (`/admin/...` to `/admin`)                     |

The admin routes render only for the administrator (`AdminGuard`, from `/api/me`); the API
answers a JSON 403 anyway. Their drawers and switches live in the URL (`?game=lotto`, `?row=...`,
`?run=<job>@<start>`, `?scope=top`, `?by=game`), so a reading can link to one game or one run.
Colour on the admin pages marks a change only: a control that reads differently from the week
before (the weekly readings in `data/readings/`), a run over its usual time or with a new warning
signature, a failed run or a missing output; a state that holds every week stays grey.

The sidebar (`components/Sidebar.tsx`) is data: the users see Lottery, Crypto, Shares and
Council; the administrator also This week (with a status dot: red when a run failed or an output
is missing, amber for a new warning signature) and the Controls, Models and System groups. Server pages are plain
anchors, marked ↗ while the app has no page of its own for them.

## API

All calls are same-origin and typed in `src/api.ts`:

- `GET /api/me` -> `{ user: { name, admin, open } | null }` (the sidebar and the admin guard; open = open access, no sign-in)
- `GET /api/markets/:market` -> the whole page model (see the `MarketView` type)
- `GET /api/quotes/:market` -> `{ at, quotes: { [symbol]: number } }`, fetched after the page
  model and refreshed every 60 s while the tab is visible

- `GET /api/admin/...` -> the admin views (`getAdminWeek`, `getAdminTuning`, `getAdminControls`,
  `getAdminModels`, `getAdminPhaseShift`, `getAdminRandomness` and their per-game getters);
  `admin.js` documents the shapes
- the Workbench (`src/workbench.ts`, `workbench.js` documents the shapes): `GET /api/admin/history/:game?scope=&month=`,
  `/api/admin/draw/:game/:date?scope=`, `/api/admin/market/:market/workbench` and its `/books/:rule` and
  `/instrument/:symbol` series. Top on a past draw is the five ranked first before it (production's git history of the
  ranking), never today's; before 25 Aug 2026 no ranking was recorded and the pages say so

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
  hooks/useUrlState.ts    the switches and the drawers, kept in the URL
  hooks/useWeekStatus.ts  This week's status for the sidebar dot, shared with the page
  components/             sidebar, KPI tile, status strip, open/closed positions, trades log, charts
  components/admin/       card, chip, drawer, DotStrip (inline SVG), stacked and duration bars, the
                          weeks-in-a-row chip, the switch, guard and load states
  components/draws/       a game's history (month cards, day lines), the drawn row, the scored tickets per
                          family, the multi-pick legend, the vote charts, the scope line; mode="admin" | "user"
                          (the users' Lottery reuses them with the top five only)
  workbench.ts            the Workbench's API types, normalisers and getters
  pages/MarketPage.tsx    loads the market and lays out the page
  pages/admin/            This week, Tuning gates, Controls, Models, Phase-shift and Randomness pages
                          and their drawers; Games, a game's history, one draw, the market workbench
                          (its cards in pages/admin/workbench/)
```

There is no dev server script on purpose: the API lives in the Express app, so the quickest
loop is `npm run build` and a reload of `/app/crypto`.
