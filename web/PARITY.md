> Working checklist for the switch to the new app (batch 4, 22 Oct 2026). Built routes: /app/admin/games, /app/admin/games/:game, /app/admin/games/:game/:date (API /api/admin/history/:game, /api/admin/draw/:game/:date) and /app/admin/markets/:market (API /api/admin/market/:market/workbench); the route names proposed below (/admin/draws) were renamed while building. The final parity sign-off happens before the switch (batch 6). Section D (24 Oct 2026, batch 6) covers the rest of the old pages and holds the owner's sign-off list. This file is now a record: the old pages and their code were removed in cleanup batch 6b (26 Oct 2026), and every old address redirects to its app page (switchover.js).

# Workbench parity: old pages to the new app (batch 4)

**How this was read.** Rendered with a copy of production `data/` (the run of 22 Oct 2026 09:04 UTC, commit `2eadd2bc`), read only, through server.js's own routes with the test/pages.test.js stubs and markets.js `install` (`page()` with `describeMarket`), as the administrator (those routes, that test, `install` and `page()` were removed in cleanup batch 6b). Pages rendered:
- `/database/:folder` for all 11 folders.
- `/database/:folder/:file` for the newest and the oldest file of each folder.
- `/markets/crypto` and `/markets/shares`.

The scratch renderer is `parity/render.js`, and its output is in `parity/out/`.

**Status codes**
- **APP**: already in /app (Crypto, Shares).
- **ADM**: already on an admin page.
- **PART**: partly in /app or an admin page; the rest is built here.
- **BUILD**: built in this batch.
- **DROP**: proposed for dropping. Only exact duplicates are proposed.

**U** says what the users' Lottery (next batch) takes from the row.

## New homes proposed for this batch

A new sidebar group, **Admin › Workbench**:

| Page | Route | API | Replaces |
|---|---|---|---|
| **Draws** | `/admin/draws` (games, grouped Lottery / Markets as in admin.js GROUPS) · `/admin/draws/:game` (history) · `/admin/draws/:game/:date` (one draw) | `/api/admin/draws/:game?scope=all\|top` · `/api/admin/draws/:game/:date?scope=` | `/database/:folder`, `/database/:folder/:file` (all 11 folders) |
| **Crypto**, **Shares** | `/admin/markets/crypto`, `/admin/markets/shares` | `/api/admin/markets/:market` | `/markets/:market` (the sidebar's "Market workbench" links move here) |

**Server side**
- **`draws.js`** (new) takes over the scoring that lived in server.js then: `splitRealResult`, `splitTicket`, `calculateProfit`, the jokerplus runs, sign and payout, multi-pick, zodiac, and the vote charts. The admin API and the next batch's user API both use it. Its tests go in `test/draws.test.js`.
- **`rankingHistory.js`** (new) handles the point-in-time ranking (below). It follows the pattern in gates.js: `execFile` git with fixed arguments, cached per HEAD.

**Components** go in `web/src/components/draws/`, with a `mode: 'admin' | 'user'` prop:
- `MonthCard` and `DayRow`
- `ResultStrip`
- `TicketTable`
- `MultiPickLegend`
- `VotesChart` (ECharts via `useEChart`)
- `ScopeLine`: says whose rows, which ranking version and which metric.

The scope switch reuses `components/admin/ScopeSwitch.tsx`.

## Point-in-time ranking (owner rule: no hindsight)

**Source.** Production's git history of `data/database/modelPerformance.json`. There are 59 versions. The first was committed on 25 Aug 2026 at 05:21 (`generatedAt` 24 Aug 22:13). After that there is usually one version per daily run (`Helpers.git_push`, "saving last predictions"), with these exceptions:
- No versions on 30–31 Aug, 1 Sep, 6 Sep or 9 Sep.
- Several versions on 25 Aug (4) and 7 Sep (3).

**Rule for draw D.** Use the newest saved version that meets both conditions:
- It was generated on or before D's day.
- It was committed before the commit that first added D's day file.

One edge: a version generated on D's day after the draw can still qualify. Examples are the first version (24 Aug 22:13) and the manual evening runs. It does not contain D, but if "before" must be literal, the first condition needs D's draw time.

The second condition matters because each run writes the day file and the report in the same commit, so it keeps D's own scoring out. The first condition keeps out back-filled files, such as the jokerplus August files and the back-built market days. The top 5 are the first 5 names of that version that are present in D's table (`ranking.topNames`).

**Special cases**
- If no version qualifies, the page says "no ranking recorded before this draw" and invents nothing.
- A game missing from that version gets the same treatment.
- The line names the version's `generatedAt` and metric.
- keno, pick3 and jokerplus were ranked by `profit_per_bet` until the 13 Oct version. From the 14 Oct 08:52 version they are ranked by `profit_bound`.
- A game that entered the ranking after 25 Aug (Joker+ 7 Sep; crypto, shares 1 Oct; cryptoweek, sharesweek 5 Oct) says so on the history page ("<game> was first ranked on <date>", `ranking.firstOfGame`); the per-draw "before-first" lines keep the global 25 Aug.
- A version git could not read (a timeout, an overflow) makes every draw whose cut-off falls after it, with no readable version in between, "unavailable" (never the older version before it); it is asked again after 30 s, not on every request. A cache entry of the wrong shape is dropped and read again from git (fixes of 22 Oct).
- jokerplus enters the history on 7 Sep 16:56, with a minimum of 4 draws and then 5.
- crypto and shares enter on 1 Oct, the week games on 5 Oct.
- A file not committed yet (the newest, before its commit) uses HEAD's version.
- The coming draw (the newest day's next table) uses today's report.

**Draws per game with a point-in-time top 5** (production, 22 Oct):

| Game | Draws with a top 5 | Total draws | First draw with a top 5 |
|---|---:|---:|---|
| lotto | 17 | 99 | 26 Aug |
| euromillions | 17 | 100 | 25 Aug |
| eurodreams | 17 | 99 | 24 Aug |
| vikinglotto | 9 | 49 | 26 Aug |
| jokerplus | 45 | 73 | 7 Sep |
| pick3 | 59 | 348 | 24 Aug |
| keno | 59 | 348 | 24 Aug |
| crypto | 21 | 52 | 1 Oct |
| shares | 15 | 46 | 1 Oct |
| cryptoweek | 2 | 33 | 11 Oct |
| sharesweek | 2 | 33 | 9 Oct |

Every other draw comes before 25 Aug, or is back-built or back-filled. Five keno files (1–6 Dec 2025) have no add commit because they were renamed in; they come before the history anyway.

## A. `/database/:folder`: one game's history of draws

| # | Old element | Status, new home | Rebuild note (trust rule) | U |
|---|---|---|---|---|
| A1 | Title, lottery nav ("New predictions · History") | DROP: navigation only. Replaced by the sidebar and a breadcrumb | | sidebar |
| A2 | Market note: what the game is, slots in order, link to the market page, "what the digit means" behind ? | BUILD Draws › game head (market games) | Link to `/admin/markets/:m` | – |
| A3 | Month card header "Best Match N (M)" (lotto, EM, ED, VL), "Best Match N" (markets) | BUILD Draws › history › month card | Say "best over D draws, R rows a draw" (R per draw: 5 with Top, "30–31" with All; the month's sum of rows over its draws is no row count). With scope Top: over each draw's point-in-time top 5, and "x of D draws have a recorded ranking" | yes: over the top 5 as known each draw |
| A4 | Month header "Total: X €" (keno, pick3); "Total · Best Match L/R (Z)" (jokerplus) | BUILD same | Say "net over T tickets of R rows a draw · stake S" | yes |
| A5 | Day row: file link, "Match N (M)", lotto "· of 9: K" (hover-only meaning) | BUILD Draws › history › day row, linking to the draw | "best of 9" written out | yes |
| A6 | Day row: profit € (keno, pick3); "Match L/R (Z) · X €" (jokerplus); colours | BUILD same | Count of rows and tickets on the line | yes |
| A7 | Newest month open, the others closed | BUILD same | Back from a draw (?month=YYYY-MM) opens and scrolls to that draw's month instead | yes |

## B. `/database/:folder/:file`: one draw

| # | Old element | Status, new home | Rebuild note | U |
|---|---|---|---|---|
| B1 | Heading (file name), "Back to History / game view" | BUILD Draws › draw head | Date and weekday; previous and next draw | yes |
| B2 | Market note (= A2) | BUILD (same component) | | – |
| B3 | **Real Result** card: the drawn row (jokerplus sign as its name) | BUILD draw › ResultStrip | Show mains, then specials (stars, dream, viking) or the lotto bonus as their own group (old: one flat row) | yes |
| B4 | "How hits are counted" help, per family (N (M); lotto bonus amber; jokerplus L/R (Z); market same slot, chance 1 in 10) | BUILD draw › one help line | Joker+: the two runs each pay by length and add up (as the profit column). Keno: the sizes played changed over time (one ticket a row on many older draws) | yes (user words) |
| B5 | "Show all N models" switch. Unchecked, it applies **today's** ranking to the past draw (hindsight) | BUILD draw › ScopeSwitch All / Top 5 | Point-in-time (above). The line names the version date and metric. With no version: "no ranking recorded before this draw", switch off | top 5 only, no switch, hidden rows not sent |
| B6 | **Analysis of Prediction** table: Model, #, numbers with hit cells, Hits, Profit | BUILD draw › TicketTable | Family rules below | yes |
| B6a | lotto: bonus amber; multi-pick 7th–9th shaded, Hits "N (M) · K" | BUILD | Size, grids and price per header shown, not hover-only. Old vintages: 7-number rows (Nov 2025 – May 2026), no multi-pick before 7 Oct | yes |
| B6b | jokerplus: L green, R blue, sign amber, sign name, L/R (Z), profit with 2 decimals | BUILD | | yes |
| B6c | keno: per row the 20-number ranking plus one ticket per size (5–10 today; the size set changed over time), profit per ticket | BUILD | Write the size (old: row index #). The 20-number row reads "ranking, not played", not "0 €" | yes |
| B6d | pick3: profit per ticket (straight, box, pairs, consolation, stake); "Hits" counts membership | BUILD | Positional cells and the bet types won next to the profit (the old Hits is not a pick3 notion) | yes |
| B6e | euromillions, eurodreams, vikinglotto: specials lit only in their own group; mains-only rows without special cells (EM since 4 Sep) | BUILD | Headers Star 1-2 / Dream / Viking (old: "Num 6, Num 7") | yes |
| B6f | Market games: hit = right bin in the right slot, instrument headers | BUILD | Same table as C14 (Day by day) for the newest 30 days: returns, arrows, exact, direction, P&L; older days: bins and hits only | – |
| B7 | Multi-pick legend behind ? (chance of 6/7/8/9 numbers, grids, prices, link to History) | BUILD draw › MultiPickLegend, **once** | The second copy under the next-draw table is an exact duplicate: DROP | yes (no admin link) |
| B8 | `chart-analysis`: every row's vote per number, drawn mains green | BUILD draw › VotesChart (ECharts, tokens, to scale) | Positional games (jokerplus, pick3, markets) pool positions, so a bar lights for a value drawn in another slot. Draw them per position (owner, Q3), every slot chart on one scale (top = the rows shown). Keno counts the rows' 20-number ranking lines, said so in its help. Missing `currentNumberFrequency` (35 old vikinglotto files): use the previous file's `numberFrequency`, which is identical everywhere else | over the top 5's rows, or not shown (Q2) |
| B9 | **Next Draw Prediction** card: "for the draw of <date>" + table | BUILD draw › Next draw, **newest day only** | Today's ranking (the coming draw). The admin sees all, with the switch | the coming draw: today's top 5 |
| B10 | `chart-detail` (next-draw vote) | BUILD, newest day only | | Q2 |
| B11 | B9 and B10 on **every past day** | DROP: an exact duplicate | The table equals the next file's Analysis rows in all 1,269 consecutive pairs (11 games), and the vote matches in 1,234 (the other 35 have no Analysis chart: see B8). Replace with "scored in the draw of <next date>" → link | – |
| B12 | First file of each game: "No predictions." | BUILD (empty state) | | yes |

## C. `/markets/crypto`, `/markets/shares`: the market workbench

| # | Old element | Status, new home | Rebuild note |
|---|---|---|---|
| C1 | Intro line + "about this page" (bins, chance, books, close times, week game) | BUILD Workbench › head (line) + Background | |
| C2 | **Today's plan**: meta (day, status, close); "One model's calls" (model, why picked: book or exact rate, its book vs buying everything, back-built and followed counts); "Following this card so far"; "sell at the close"; table: instrument (last close), call, band + bin %, buy, sell, models calling up; footnote; help | PART: APP Open positions (day rows: status, band, buy and sell, model pill with the pick in a hover title). BUILD Workbench › Today's plan (the three labelled lines, call/band/bin, models calling up, help) | Lines as the 13 Oct plan-card rule; counts equal to the table. The book line is over the picked model's own settled days (first – last), with buying everything over the same days; the old page gave the market's longest record (fix of 22 Oct) |
| C3 | **Week ahead / This week**: meta (week end, elapsed, settled weeks); the three labelled lines; table: call, band at the week's close, so far, buy, sell, models up; help (guard from 26 Oct) | PART: APP Open positions (week rows, guard). BUILD Workbench › Week card | Rename to "Week game" so it is not confused with the admin page "This week". The book over the model's own weeks (as C2); weeks dated by their end ("the first ending 11 Oct") |
| C4 | Following the plan, card meta (day and week live totals) | DROP: the same figures as the first lines of C5 and C6 | |
| C5 | **Day to day**: "Followed since … (n days, n positions): total vs buying everything (ahead/behind)"; chart of the two running totals; table (day, row followed / by exact rate, positions, result, running, buying everything, its running) | PART: APP Cumulative profit and Daily profit (daily and weekly combined, by position). BUILD Workbench › Following the plan › Day to day (ECharts, `buildCumulative` pattern) | The row followed each day is the admin's figure |
| C6 | **Week to week**: the same per settled week | BUILD › Week to week | Weeks dated by their end: "Followed for 2 live weeks (the weeks ending 11 Oct – 18 Oct)", "Back-built weeks ending before 11 Oct" |
| C7 | "Late" toggle (tickets too late to follow) and "Back-built before <first live>" toggle, each with its totals | BUILD, closed by default, totals on the summary line | Not in the live total, said on the line |
| C8 | **Paper trading**: meta headline (best book per rule among rows with ≥10 settled days vs the market; holding vs buy-and-hold; open to close; long & short vs buying and selling everything); "how the books trade"; chart of cumulative money per model, rules daily / hold / open, long only / long & short, 1M–All, zoom and pan, models best / all / none | BUILD Workbench › Paper books | ECharts plus DataZoom (`echarts.use` registers Bar, Line, Grid, Tooltip only). The headline names the minimum of 10 days on its line |
| C9 | **Instrument cards** (one per coin or share): last close and date; chart Price lines / Price bars / Moves, 1M–All, zoom, model toggles (best-exact-rate model ticked); "Every model's call for <day>" (bin, position, price, band); "Judged at … Right now …" + help | BUILD Workbench › Instruments (ECharts; band bars as a custom or stacked series) | "Judged at / Right now" is the same text in every card: show it once above the cards (DROP the copies). /app has no price chart. The band is stacked with `stackStrategy: 'all'` (the default same-sign stacking drew a band below zero up from 0 on the Moves scale); the window's first close shows its year |
| C10 | "How to read the charts" help | BUILD › Background | |
| C11 | Background › **How a day becomes a draw** (3 steps with the newest day as the worked example; strips per instrument with the bin edges; why bins; a bin as a price range; log-return note) | BUILD Workbench › Background (closed) | Keep the strips (a figure) and the 3 steps short; the rest behind ? A share day's close: 16:00 New York, 22:00 Belgian time in most weeks, 21:00 in the clock-change mismatch weeks |
| C12 | Background › **Models over N scored days**: days, exact, adjacent, direction (green when above chance), trades, win rate, P&L, per trade, holding, open-close, long & short (hover: trades, win rate, per position; sized row "long only") | BUILD Workbench › Models | ADM Models has the History ranking (exact hits) and the row /app follows, not these columns. Hover-only figures become visible; chance on the header |
| C13 | Background › **Accuracy per day** chart (mean exact rate, best model that day, chance; 0–1) | BUILD Workbench › Accuracy per day (ECharts) | |
| C14 | Background › **Day by day** (newest 30 days): returns and bins per instrument, best, mean exact, game-view link; opens to every model's bins with ▲ / ▼ / ×size, exact, direction, P&L | BUILD Workbench › Day by day. The open day uses the B6f table | |
| C15 | Background › **The guard on the week positions**: totals over live weeks (guard, plain week, buying everything); table (week, row, positions, guard, plain, buying everything, note: what the user pages counted, booked after the fact) | PART: APP (the guard in Open positions, guard weeks counted from 26 Oct). BUILD Workbench › The guard | /app's guard note points the reader to "the Admin market page" for the earlier weeks: this section is that page. The live totals sum the unrounded weeks and round once, as the old page and C3/C6 |
| C16 | Background › **Rows under a proper score**: headline vs GARCH, GARCH vs uniform, table (log-score ± se, vs GARCH, vs uniform, exact, adjacent, direction, trades, P&L), failed rows, generated | ADM Controls › Market rows (drawer and overview) | Missing: failed-row names and reasons (only a count is sent today): BUILD into the drawer. Trades and P&L (price units) were left out on purpose: owner, Q4 |
| C17 | Background › **Regime reading**: row, after, regimes, features, template T#, label + volatility rank, probability, expected next-day return per instrument | PART: ADM Controls › Regimes (label, p, volatility rank, regimes, readings kept). BUILD Workbench › Regimes: features, template, expected return | |
| C18 | Background card meta (list of sections) | DROP: navigation | |
| C19 | Footer: record generated (Belgian time), newest game day, link to the game view | BUILD Workbench › head stamp | Link to Draws |

## What the users' Lottery (next batch) needs from this batch

> **Built in batch 5 (23 Oct 2026).** API: `lottery.js` (`/api/lottery`, `/api/lottery/:game`, `/api/lottery/:game/:date`, `/api/home`), with the owner's answers to Q1 and Q2 below: the users' history starts at each game's first saved ranking (the earlier draws left out, one line says how many), and the vote is drawn over the five rows sent. App: `/` (Home), `/lottery`, `/lottery/:game`, `/lottery/:game/:date` (`components/draws/*` in `mode="user"`, no switch, no rank numbers, no stored vote) and `/council`. The sidebar's Lottery and Council are app routes now (no ↗). The old `/lottery`, `/database` and `/council` server pages stayed until the switch (batch 6: since then every old address redirects to its app page, see section D); the lottery folders of `/database` are the administrator's in the meantime (they send a past draw's every row and its file's vote): a user is redirected to the same list, game or draw under `/app/lottery`. The markets' game views stay as they are.

1. **The `rankingHistory` API**, per game and draw: `{version generatedAt, metric, minDraws, top5 present in the table}` or `{none: 'before 25 Aug 2026' | 'game not ranked yet' | 'back-filled'}`. The user API calls it server side.
2. **`draws.js` scoring** with a scope: figures (A3–A6, B6) over the rows in scope only, with the count of rows on the same line. In user mode the server cuts the rows; hidden rows are never sent.
3. **The components** in user mode:
   - MonthCard and DayRow (A3–A7)
   - ResultStrip (B3)
   - TicketTable with every family rule (B6a–e)
   - MultiPickLegend without the admin link (B7)
   - VotesChart over the shown rows only (Q2)
   - the next-draw table with today's top 5 (B9)
   - ScopeLine: "the 5 models with the best record by <metric> as ranked on <date>"
4. **Wording**: the "how hits are counted" lines (B4) and the empty states (B12, no ranking recorded) in the user's terms.

## Open questions for the owner

1. **Before 25 Aug 2026 no ranking was recorded.** That covers 82 of 99 lotto draws. Users' history there: every row with the line "no ranking recorded before 25 Aug 2026", or start the users' history at 25 Aug?
2. **Vote charts (B8, B10) in user mode:** recompute over the top 5's rows, or leave them out (the old user page leaves them out)?
3. **Positional games (jokerplus, pick3, markets):** draw the vote per position or slot, instead of the pooled chart that lights values drawn in another slot?
4. **Market rows' Trades and P&L columns** (price units, the Sunday backtest): drop them for good, or add them to Controls › Market rows?
5. **pick3 Hits column:** replace membership hits with positional hits plus the bet types won (B6d)? **Decided (owner, 24 Oct 2026): yes** - users see digits in place and the bet types won; the membership count stays on the administrator's view only.

## D. The rest of the old pages (batch 6, 24 Oct 2026)

**How this was read.** Before the switch a read-only sweep compared every element of the old pages with the app, in eight areas:
- Home and the shared chrome (navbar, pipeline banner, footer);
- the users' lottery pages;
- the administrator's ranking cards and control cards on `/database`;
- the markets (the users' view, and rows C1-C19 above);
- the lottery workbench (rows A and B above) and the Council;
- the system pages (account, users, activity, jobs, What's new, the introduction);
- the redirect map and every link to an old address.

An adversarial reader then tried to refute each reported gap. Result: 42 gaps reported, 6 refuted, 36 confirmed (no blocker, 10 should, 26 minor).

**Closed in batch 6**
- **Run in progress** (should): `pipeline.js` (moved out of server.js) gives `/api/home` and `/api/lottery` a `running` field. Home and Lottery show the old banner's line, and refresh every minute while a job holds `process.lock`.
- **Users' lottery**: the Joker+ sentence that only the sign can be chosen; "today," on the coming draw; "the 5 of N models"; the multi-pick order sentence; the bonus number not counted in the chances.
- **Administrator's ranking cards**: a combination's "no verdict" on the hit games (a set scores at least its own rows by construction); "above chance" tags and reading guidance on multi-pick; the Best model and Phase-shift help texts; the count of established rows tried; "not declared" Since.
- **Control cards**: the feature control's chance of a column clearing the band by luck (one week in K); its noise-column and importance definitions; the AUC cell label; the Q2 suite (quantum included or not); the randomness autoencoder z and the "boring causes" caveat; the market rows' ablation reading, row kinds, probability floor and HMM feature set (also on the workbench's Regime card).
- **Markets**: the draw table's position marks (▲, ▼, ×size, sits out) and the bin's price range on hover; a zoom slider under the paper-books and instrument charts; the week-to-week running chart; the "why bins" background. The users' position size (×N) and the model pill's reason (book or exact rate) are latent while GARCH is picked.
- **Council**: the administrator's Ask is disabled with the reason when the Council API is down; live answers in folded cards with their Member N labels.
- **System pages**: each run's commit; the Council API's command and directory; the users' pages named correctly on People; the activity log's reach and address source; the schedule's off switch (`config/scheduler.disabled`).
- **Phone check** (360 / 390 / 430 px): the model column stays in place in the ticket tables; long tile labels wrap; a wrapped meta line never starts with a dot.

**Decided with the owner (24 Oct 2026)**
- Q5 above (pick3 Hits): users see digits in place; the old membership count is shown to the administrator only.
- The introduction (`announcements.js` TOUR) rewritten for the app, and the switch announcement (`2026-10-24-new-interface`, major, every reader), both proofread by the owner.

**Deferred** (not needed for the switch)
- Several models drawn at once on an instrument chart, the old "Price bars" view and its "Reset zoom" button. The slider zooms; zooming by mouse wheel or pinch inside the plot was left out on purpose, so a phone scrolls the page.
- Position size on week rows: needs a decision on whether the guarded week scales by size.
- The Council's administrator-only mode (`COUNCIL_ACCESS=admin`): not set on production, so the app does not reflect it yet.
- The models drawer's Since column shows a dash when no date is declared (the overview says "not declared").

**Sign-off before the switch** (the owner)
- [ ] As a user: Home, Lottery (a game, its history, a draw), Crypto, Shares, Council, account, What's new; on a phone too.
- [ ] As the administrator: This week, Controls, Models (all six pages), Jobs & services, People.
- [ ] A few old bookmarks open their app page, for each role.
- [ ] Sign-in page: light, dark, phone.
- [ ] The introduction rewrite and the switch announcement, proofread and approved.

