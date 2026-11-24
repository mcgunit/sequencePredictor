# Scoring: hits, profit, prize tables and the ranking

This page covers how a stored prediction is scored against its draw: how hits are counted per game, how a ticket's profit in euro is computed (fixed payout tables and each draw's own prize table), the order numbers are stored in and Lotto's extra numbers, which draw a prediction is for, and the performance report that ranks the models (the live record, the forward record, the users' top five, the best combination and the phase-shift analysis). The models themselves are on [models](models.md), the tuners' scores on [tuning](tuning.md), the controls and chance tests on [research](research.md), the market games on [markets](markets.md).

## Games and how they are classified

| Game | Ticket | Kind | Money |
|---|---|---|---|
| Keno | 20-number ranking plus one ticket per size | set | fixed payout table |
| Pick3 | 3 digits in drawn order | positional | fixed payout table |
| Joker+ | 6 digits in drawn order + zodiac sign | positional | fixed payout table |
| Lotto | 6 numbers (the draw adds a bonus number) | set | the draw's prize table |
| EuroMillions | 5 numbers + 2 stars | set | the draw's prize table |
| EuroDreams | 6 numbers + dream number | set | the draw's prize table |
| Vikinglotto | 6 numbers + viking number | set | the draw's prize table |
| Crypto, shares and their week games | one bin per instrument | positional | settled on [markets](markets.md) |

A game is **positional** when its draw is an ordered sequence in which values may repeat: `Helpers.is_positional_game` (`POSITIONAL_GAMES`: pick3, Joker+, crypto, shares; a week game's name contains its market's, so it matches too). Every positional branch in the code asks that predicate instead of testing a game name. Positional games use per-position models and have no set-style rows; the vote rows take a per-slot form there (see [models](models.md)).

## Joker+ (`jokerplus`)

Joker+ is a 6-digit positional game plus a zodiac sign out of 12. The digits 0-9 are drawn with replacement, so a digit can repeat and leading zeros count.

- The sign is one **categorical special column**. It is encoded to a code 0-11 at every CSV parse site (`Helpers.encode_zodiac` / `decode_zodiac`, canonical regulation order), so every model, backtest and report works on integers; the web app decodes the code back to the name.
- The codec accepts the CSV spelling `Tweeling`, the regulation's `Tweelingen` and the English names of the draw API. `DataFetcher` parses the API's `'430109Scorpio'` form and writes the Dutch CSV spelling.
- A stored row has 7 values: six digits and the sign code. A row with only the six digits is scored with the sign unknown (no sign match possible).
- The player cannot choose the digits (combinations are system-generated; only the sign is selectable). Digit predictability is therefore an integrity signal, not a way to play, and the sign's own prize only refunds the stake.
- For that reason `RL Ticket Model` does not run for Joker+, and the per-slot vote rows (`WeightedEnsemble Model`, `SubsetEnsemble Model`) run only when `"useJokerplusEnsemble": true` is set in `bestParams_jokerplus.json`. The other rows are described on [models](models.md).
- The tuners score Joker+ like pick3, on slot hits (for Joker+ the paid runs, L+R) with the capped profit bound as a small tie-breaker; `HyperoptQuantum` scores the digits only, sign unknown. See [tuning](tuning.md).

## Hit counting

Hits are counted **per pool**, never as one flat intersection over the whole result row: a predicted main number equal to a star's numeric value is not a hit, because the pools are separate drawings.

- **EuroMillions, EuroDreams, Vikinglotto**: the main numbers score only against the drawn mains, the special column(s) (stars, dream number, viking number) only against the drawn specials. Shown as `N (M)`: N main hits, M special hits.
- **Lotto**: a play is 6 numbers scored against the 6 drawn mains. The 7th drawn value (the bonus) only supplements a partial match and is matched against the played numbers themselves; a play has no bonus slot. `5 (1)` is 5 mains plus the bonus; `6 (0)` is the jackpot, and a full match rules a bonus match out.
- **Keno**: one pool, set hits.
- **Pick3**: by position (a digit in the right slot). The pages also show the membership count (a right digit in any slot).
- **Joker+**: prizes go to groups of *consecutive* positional matches counted from the left end or the right end of the 6-digit number, plus the sign. Hits are tracked as `L/R (Z)`: leading run / trailing run / sign, for example `3/1 (1)`. A full match is `6/0`; otherwise L + R is at most 5. "Hits" in the report and the lag analysis is L + R (`Helpers.jokerplus_runs`).
- **Market games**: by slot, a bin in the right instrument's slot.

In the ticket tables a hit cell is green, Lotto's bonus and Joker+'s sign amber, Joker+'s trailing run blue.

The same split runs through every reader of a ticket:

- `matchingNumbers` in the day files: `matching_numbers` holds the mains, `special_matching_numbers` the specials or the bonus.
- The draw and history views of the web app (`draws.js`).
- `modelPerformance.json`: `avg_hits`, `best_hits` count main hits only; `avg_special_hits` is the special pool (Lotto: the bonus supplement; Joker+: the sign, 0 or 1).
- The lag analysis and the randomness watch, which use the main columns only.
- `src/Backtester.py`'s per-day hit metrics; its baselines bet main-sized tickets on the games with special columns.

## Profit: the fixed payout tables

Profit is always **net**: the payout minus the stake, so a losing ticket is minus its stake and profit per bet is comparable across games. The tables are `Helpers.PAYOUT_TABLE_KENO`, `PAYOUT_TABLE_PICK3` and `PAYOUT_TABLE_JOKERPLUS`; `draws.js` mirrors them for the pages.

### Keno (`keno_ticket_profit`)

- Stake 1 EUR per ticket. Only tickets of 5 to 10 numbers are bets; the 20-number ranking is never a bet.
- A row bets every 5- to 10-number ticket it stores for the draw. The report keeps the bets and profit per ticket size as well (`profit_by_size`).

### Pick3 (`pick3_ticket_profit`)

- The tracked ticket plays all four bet types at 1 EUR each: straight, box, front pair, back pair (4 EUR). A triple cannot play box (3 EUR).
- Each bet is evaluated on its own, so prizes cumulate.

| Bet | Pays (EUR) |
|---|---|
| Straight: all three digits in order | 500 |
| Straight consolation: only the units digit matches (no straight win) | 1 |
| Box, three distinct digits | 80 |
| Box, with a doubled digit | 160 |
| Front pair: positions 1-2 in order | 50 |
| Back pair: positions 2-3 in order | 50 |

An exact hit therefore nets +676 with distinct digits (500 + 80 + 50 + 50 - 4) and +756 with a double (500 + 160 + 50 + 50 - 4).

### Joker+ (`jokerplus_ticket_profit`)

- Stake 1.50 EUR, one ticket per draw (the row's 7-value main ticket).
- All six digits: 20,000 EUR, or 200,000 EUR (the fixed minimum jackpot) with the sign; the jackpot tier replaces the sign prize.
- Otherwise `T[L] + T[R]` with `T = {1: 2, 2: 5, 3: 20, 4: 200, 5: 2000}` (the left and right groups cumulate), plus 1.50 EUR for a matching sign.

## Profit for Lotto, EuroMillions, EuroDreams and Vikinglotto: each draw's prize table

These four games pay pari-mutuel, so there is no fixed table. The National Lottery's draw API (the same call `src/DataFetcher.py` reads the numbers from) carries every draw's prize tiers: the tier name, the amount per winner and the number of Belgian winners. `src/PrizeTables.py` keeps them per game.

### The prize files

- File: `data/trainingData/<game>/<game>-prizes.json`, `{date: {price, jackpot, tiers}}`. Each tier holds `name`, `mains`, `special`, `amount`, `winners`, `type`. The price is the draw's own base price from the API.
- The daily fetch records the tables of the new draws. `Predictor.py` then fills in the table of every draw the game has scored tickets for (`PrizeTables.backfill`, one request per 120-day window; nothing to ask once they are stored).
- A date the API answered without a draw is stored as "no draw in the API" once it is a week old, so it is not asked for again; a newer one may still be on its way.
- A failed fetch costs the profit of those draws, never the run.
- The raffles (Lucky Lotto, My Bonus) are drawn on a code and are not stored or scored.

### How a ticket is scored (`PrizeTables.ticket_profit`)

A row's profit on a draw is **one grid** of its own ticket at the draw's price: the tier its hits reach, minus the price.

| Game | One grid | Usual price |
|---|---|---|
| Lotto | six numbers; the bonus counts when one of the six is it | 1.50 EUR |
| EuroMillions | five numbers and two stars | 2.50 EUR |
| EuroDreams | six numbers and the dream number | 2.50 EUR |
| Vikinglotto | six numbers played with all five vikings; the viking number on the ticket is the Super Viking, which doubles the prize | 10.00 EUR (five times the API's base price of 2.00, `GRID_PLAYS`) |

- Tier names are read by `tier_key`: Lotto `5` / `5+1` (the +1 is the bonus), EuroMillions `4+2` (stars), EuroDreams `6+1` / `5/` (the slash is any dream number), Vikinglotto `k` (k numbers, the drawn viking is not the Super Viking), `k+Fav` (k numbers and the drawn viking is the Super Viking: twice `k`) and `6/` (all six whatever the viking number: the jackpot). On Vikinglotto one correct number pays 2.00 EUR of the 10.00 a grid costs.
- A tier nobody won shows 0 in the API (the money rolls over). A hit on it is valued at the draw's announced jackpot for the top tier, otherwise at that tier's median over the stored draws where it paid.
- EuroDreams' two annuities count at their rule value: 20,000 EUR a month for 30 years, 2,000 EUR a month for 5 years.
- A ticket longer than a grid plays its first numbers. Lotto's 7th to 9th numbers (the multi-pick below) are never part of the grid.
- A ticket that is not a full grid (fewer mains, or without its stars, dream or viking number) has no bet on that draw: hits only.
- A draw without a table or without a price has no bet.

`TrimLottoBonusGuess.py` is a one-time tool for older Lotto day files in which statistical rows stored a 7th value (a guess at the bonus column, which no grid plays): it cuts those rows to six numbers and rescores `matchingNumbers`. It is a dry run unless `--apply --backup <folder>` is given, and it takes `process.lock` while it writes.

### Where the profit is used

- The performance report ranks the four games on the capped profit bound, like the fixed-table games (see the ranking below). A game without any stored table ranks on average hits.
- The History pages show the profit next to the hits: `draws.js` (`PRIZE_GAMES`, `prizeTicketProfit`) mirrors the Python rule ticket for ticket. A draw's and a month's profit count the grids the table could score; a ticket that is not a full grid, or a draw without a table, shows no profit rather than 0.
- The combination search and its shuffled control stay on hits for these four games, and so do the tuners and the meta-learners.

Self-check: `python3 -m src.PrizeTables` (part of `npm test`) checks the tier reading and the scoring.

## Number order and the Lotto multi-pick

### Most probable first

For the number games (Lotto, EuroMillions, EuroDreams, Vikinglotto, Keno) every row's main numbers are stored in the order of the model's own probability, highest first, not small to large (`src/MultiPick.PROBABILITY_ORDER`, `in_score_order`, applied by `applyMultiPick` in `Predictor.py`).

- Stars, the dream number and the viking number stay where they are. Every keno ticket and subset of a row is ordered.
- The positional games (pick3, Joker+, the markets) keep their slots.
- Hits of a set game are set hits, so the order changes nothing in the scoring; it lets a reader see whether the lower-rated places hit about as often as the top ones, which is what a fair game predicts.
- The pages show tickets in the stored order and never sort them. The one position-sensitive reader, the RL Ticket Model's rank feature, reads every ticket sorted ascending.
- A row without a ranking keeps its ticket as it is: `HybridStatisticalModel` has no `score_numbers`, and a row whose ranking fails in a run is left alone for that run.

The ranking is each row's own `{number: score}`:

| Rows | Ranking |
|---|---|
| Statistical and boosting rows | the model's `score_numbers` |
| Deep-learning rows | the softmax probability summed over the main positions (the inclusion probability; a star's softmax shares labels with the mains and is left out) |
| Meta-learner rows | their probabilities |
| Vote rows | their weighted votes |
| RL Ticket Model | the scores of its decode |

For the sampling rows (Markov, the Monte Carlo rows, PoissonMarkov) the ranking is the model's distribution estimated by `score_numbers` sampling, a larger sample than the single ticket it played. A played number can therefore rank below an unplayed one: the played numbers stay the ticket.

When the game has a meta-learner artifact (`data/models/<game>/meta_learner*.joblib`), the base rows' rankings come from the scoring that artifact needs anyway. Without one, `Predictor.py` computes them itself and says so in the log.

### Lotto: the 7th to 9th numbers

Every Lotto row still plays six numbers. After its six, a row names the three next most probable numbers it did *not* play: `multiPick` in the day file (`src/MultiPick.py`). The day files keep the six as the ticket and the three as `multiPick`, so every reader of a ticket is unaffected.

They are meant for a **system play** as Belgian Lotto sells it, at 1.50 EUR a grid:

| Numbers | Grids | Price | Chance of 3 or more main hits |
|---|---|---|---|
| 6 | 1 | 1.50 EUR | 2.38% |
| 7 | 7 | 10.50 EUR | 3.94% |
| 8 | 28 | 42 EUR | 5.94% |
| 9 | 84 | 126 EUR | 8.39% |

On the pages the extras are three shaded, dashed cells after the six. The hits column shows the hits of the six and, after a dot, the hits among all nine; the day list shows the best nine-number hits of the day. A row without extras shows empty shaded cells. The users' Home shows each game's top five tickets in the stored order, Lotto's 7th to 9th shaded after the six.

### Reading the multi-pick figures

More numbers win more by arithmetic alone. The chances in the table are exact hypergeometric values for 6 of 45 drawn ("3 or more main hits" is the smallest prize that needs no bonus number; the ranks that use the bonus are not scored here). Nine numbers have 3.5 times the chance of six at 84 times the price, and any single extra number is drawn 6/45 = 13.3% of the time.

So the questions the tracking answers are not "do nine numbers hit more than six", but:

- does a model's 7th, 8th or 9th number hit more often than 13.3%, and
- do nine model numbers reach the smallest prize more often than 8.39%.

The report scores both per row (`multi_pick`, over the draws with extras): the average hits of the six and of the nine, the share of draws with 3 or more main hits with six and with nine, and the hit rate of each extra number. Per game it carries the chance table (`games[game].multi_pick`). The administrator's ranking page shows them in Lotto's *Multi-pick* section, tagged when above the level. One row above a level over a few draws is noise; a row that stays above it over many draws is the finding.

The euro value of a 7- to 9-number system play is not built: the pages show hits and chance for the extras, not money. The planned reading of the extras is on the [roadmap](roadmap.md).

Self-check: `python3 -m src.MultiPick` (part of `npm test`).

## Which draw a prediction is for

A day file (`data/database/<game>/YYYY-M-D.json`, day and month not zero padded) is named after the draw it scored. Its `newPrediction` is the prediction for the **next** draw of that game: the rows that reappear as the following file's `currentPrediction`.

- The coming draw's table (Home, the Lottery pages, the newest draw under *Games in detail*) is labelled with the draw it applies to.
- The date comes from the game's own stored history, not a hardcoded calendar (`nextDrawDate` / `nextDrawDay` in `draws.js`): the weekdays of the last 40 draws up to that file are the game's schedule, and the prediction applies to the first later date on one of them. A schedule change is picked up by itself, and an older page is read with the schedule of its own time.
- No date is shown when no such day falls within two weeks.
- The coming draw also shows its sales close ("sales close 13:00"), with a warning when its tickets were made after it.
- A date already past says so ("that day has passed; the next run scores it"): the newest stored prediction is then for a draw that has taken place, which happens when the predictor has not run since.

## Model performance report (the History ranking)

After each prediction run, `Predictor.py` writes `data/database/modelPerformance.json` (`Helpers.generate_model_performance_report`). Per game it holds every model's record over all scored history (each file's `currentPrediction` against its `realResult`).

### What a row's record holds

- `draws`, `avg_hits` and `best_hits` of the main ticket, `avg_special_hits` where the game has a special pool.
- For the money games: `bets`, `profit_total`, `profit_per_bet` and `profit_bound`.
- `live_draws` and a `live` block: the same figures over the live record (below).
- Keno: `profit_by_size`. Lotto: `multi_pick`.
- `learning_check` for the deep-learning rows (see [models](models.md)) and `chance`, the per-row chance test (see [research](research.md)). Neither is read by the ranking.

Per game the report also holds the ranking's `metric`, `minDrawsForRanking`, `bestModel`, `rankedOn`, the `live`, `since`, `liveness`, `chance` and `combinations` blocks, the `lagAnalysis` (below) and the `anomalyWatch` / `randomnessWatch` blocks of the *Randomness watch* page (see [research](research.md)).

### The ranking metric

| Games | Metric |
|---|---|
| Keno, Pick3, Joker+ | `profit_bound` |
| Lotto, EuroMillions, EuroDreams, Vikinglotto | `profit_bound` (average hits while a game has no prize table) |
| Crypto, shares and their week games | `avg_hits` (slot hits) |

The **profit bound** (`Helpers._profit_bound`, `src/TuningScore.score_bets_by_day`'s capped bound):

- per draw, the profit per bet with every bet capped at the 20 EUR lucky-strike threshold, so one jackpot cannot carry a row;
- a keno bet is measured against a fair ticket of its size, so 0 is a fair ticket;
- the value is the mean minus one standard error over the draws with bets;
- a row with fewer than two such draws has no bound and sorts last.

The raw profit per bet stays in the record as its own column. Rankings use per-bet and per-draw averages rather than totals, because models joined the tracking at different times.

### Guards on the order

- **Short records.** Models with fewer scored draws than `minDrawsForRanking` (10, or the largest draw count in the game if that is lower) are listed, marked as a short record and sorted below the ranked ones.
- **Variant rows.** A model's variant rows (`<row> · Sampled`, `<row> · Memorise`, see [models](models.md)) are scored and flagged `variant`, and always sort after every other row. They are never the best model, never in the users' top five, and not part of the chance test's family or the combination search.

### The live record

History rebuilds and late runs can write a lottery ticket after its draw's sales close, and a ticket can be rewritten after the draw. A row's draw counts in its **live record** only when its ticket (the row in the `newPrediction` of the day file before the draw) was saved before that draw's sales close (`src/Liveness.py`).

`src/drawClose.json` holds the close per game and weekday in Brussels time, taken from the draw API's `drawTime`:

| Game | Sales close |
|---|---|
| Lotto | Wed, Sat 19:00 |
| Vikinglotto | Wed 18:00 |
| EuroMillions | Tue, Fri 20:00 |
| EuroDreams | Mon, Thu 20:00 |
| Keno, Pick3, Joker+ | Mon, Tue, Thu, Fri 20:00; Wed, Sat 19:00; Sun 13:00 |

A draw on a weekday the table does not list is judged against the game's earliest close, and the log asks for the schedule to be checked.

A ticket is judged in this order, first match wins:

1. **Live by the stamp**: the file's `madeAt` is before the close and the first commit at or after the stamp holds the ticket (or the file has not been committed since and holds it on disk).
2. **Live by a commit**: a commit before the close holds the ticket.
3. **Recovered**: the version written by the latest commit before the close holds the row with other content. The ticket was rewritten after the draw, and the live record scores it as it stood at the close.
4. **Back-built or late**: no version before the close, or the row was added after it.

Further rules:

- A second row of one name on a draw is not live (a duplicate).
- A file's modification time is never evidence.
- Git is trusted only for the checkout's own `data/database` in a full (non-shallow) clone, under one 60-second deadline for the whole scan. All lottery games are judged before any game is scored.
- Without git, a stamped ticket file is judged by its stamp, and an unstamped draw by the previous `lotteryLiveness.json` up to its `checkedThrough`. With neither, the game reads **unchecked**: every row counts as not live, the admin pages show its draws without a live split, and the game is never ranked on its live record, so a one-day git failure never reaches the users' history.
- A game whose judging fails loses its live blocks, never the report.

What the report adds:

- per row `live_draws` and `live` (draws, recovered, not live, and the same figures as the all-draws record);
- per game `live` (a ranking shaped like `since`), `liveness` (how it was judged, the counts, and the rewrite audit: hits of the tickets at the close against hits of the rewritten ones) and `rankedOn: "live"`.

The lottery games are **ranked on the live record**: same metric, the 10-draw guard applied to live draws. The users' top five and the point-in-time History ranking follow that order. The admin pages show every stored draw's value beside it.

The per-draw verdicts go to `data/database/lotteryLiveness.json`: only the draws with a row that is not live as stored, sorted keys, rewritten only when a verdict changes, committed by the daily run.

On the pages:

- The admin draw table marks each such row: back-built or late, rewritten (with the ticket that stood at the close, scored) or duplicate.
- The users' history greys a draw whose five included a ticket made after the sales close and leaves it out of the totals ("over N draws (M left out: tickets made after the sales close)").

Not on the live record: the combination search, the phase-shift analysis, `HyperoptEnsemble.py`, `HyperoptRLTicket.py` / `src/RLTicketModel.py` and `src/RLPositionModel.py` read every stored ticket.

The market games use a day-level check instead (`Helpers._days_predicted_before_draw`): a market day is live when its ticket was made before the day's close. Their `live_draws` count those days; their ranking stays on every stored day.

Self-check: `python3 -m src.Liveness` (part of `npm test`), on temporary git repositories.

### The forward record (`since.json`)

A `since.json` at the repository root declares a design freeze (`src/Since.py`):

```json
{"since": "2026-09-27", "label": "September 2026 design freeze"}
```

- The report then adds, per game, the same ranking counted only over the days on or after that date (`games[game].since`: same metric, same minimum-draws guard, variant rows last).
- It counts the live record only: a lottery row whose ticket stood at its close, a market day whose ticket was out before its close. A lottery game whose liveness could not be judged falls back to the stored rows.
- Nothing is rebuilt and no file is touched: the date is a filter that turns the record after a freeze into that design's own out-of-sample track record, the forward counterpart of the lockbox ([research](research.md)).
- The ranking page shows the best row since the freeze next to the all-history one, with each model's since value and draw count. A game reads "no scored draw since ... yet" until its first such draw is scored.
- No chance band labels the since column or any other ranking column; the per-row chance test is its own card ([research](research.md)).
- A malformed `since.json` costs the since ranking, never the report.

### Who sees what

- **Administrators**: *Best model & rankings* (`/app/admin/models`, `/api/admin/models` and `/api/admin/models/:game`): a "Best model per game" card and per game the full ranking, Lotto's multi-pick table and the best combination.
- **Users**: the first five ranked rows that are present in today's table and are not variant rows (`ranking.js`, `TOP_MODELS_SHOWN`). A ranked row missing from today's table is passed over and the sixth comes in.
- **History, point in time**: a past draw is read with the five rows ranked best *before* that draw, never today's five (`rankhistory.js`, from the committed versions of the report). The cut-off is 15:00 Brussels time on the draw's date, or the draw's sales close when that is earlier (Sunday keno, pick3 and Joker+ at 13:00). A draw before the first saved ranking is left out, and the page says how many.
- **Home summary** (`lottery.js` `summaryOf`): per game the best row by total profit on its live tickets, among the rows with at least the ranking's minimum of live draws (by average hits for a game without a money record), and the best combination with its control's p-value.

### Best combination per game

Next to the single-row ranking each game carries a `combinations` section (`Helpers._build_combination_report`), shown as the "Best combination per game" card.

- **Candidates**: the established rows (at least `minDrawsForRanking` draws; on the payout games with at least one bet), without variant rows, capped at the 32 strongest in the all-draws order.
- **Search**: every pair and triple, scored on the draws all of its members were scored on (the same minimum applies to the shared draws). It reads every stored draw, not the live record.
- **Keno, Pick3, Joker+** rank sets by **net profit per draw** of playing every member's tickets. Profit is additive, so a set beats its best row only when at least two rows are positive on those draws; the card states whether it does. A greedy build-up beyond the best triple keeps adding rows while profit per draw improves. Profit per *bet* is a stake-weighted mean of the members' and can never beat the best member: it is shown, not ranked on.
- **Every other game** (the four prize-table games and the markets) ranks by the **best line held per draw**: the mean over draws of the best member's hits, which rewards rows whose good days do not coincide. It grows with every extra line, so pairs and triples are ranked separately and no greedy build-up is run.
- **Control**: because hundreds of combinations are scored, a best one always exists by luck. The card shows how many were evaluated and a **shuffled-history control**: the identical search re-run 100 times with each draw's real result moved to another day (every ticket keeps its day). The p-value is the share of shuffles whose best combination did at least as well, with +1 smoothing. A p-value that is not small means the card shows the winner of a lottery among combinations, not an edge.

Ticket-versus-result scores are computed once (one-hot matrix products for hits and for keno, whose payout depends only on played and matched counts; the positional payout functions for pick3 and Joker+), so the control stays cheap in the daily run. A failure of the section costs only the section.

### How long a place is held

Home and the Lottery page say, beside the best model, the best combination and each of the top 5, how long it has held that place (`standing.js`). Nothing is computed about a model: the saved rankings are counted.

- **A draw's standing** is read from the ranking saved before that draw (the cut-off of `rankhistory.js`, so no hindsight): its best model as Home names it (the highest total profit, or average hits, among the rows with enough live draws; `ranking.bestOf`), the first five of its ranking, and its best combination. A draw without a ranking saved before it is not counted; the first ranking was saved on 25 August 2026.
- **The coming draw** is the last draw of every count: today's report and the five rows the pages show.
- **In a row**: the draws up to and including the coming one over which the place was held without a break.
- **The score**: the share of all counted draws on which the place was held, in percent. A row that was in the top 5 before 40 of 61 draws has a score of 66%.
- **A combination** is the same one when its members are the same, in any order.
- **Who held the top**: administrators have, on *Best model & rankings*, a card with per lottery game the runs of the best model and of the best combination, the newest first (at most eight; `/api/admin/standing`). Users do not get it: it names rows outside the day's five.
- A ranking saved before the live record (27 October 2026) has no live figures: its best model is its first ranked row with enough draws.
- How long a place was held says nothing about the next draw; the line says so in its tooltip.

## Phase-shift (lag) analysis

Also part of the report (administrators: *Phase-shift*, `/api/admin/phase-shift`). Every day's `newPrediction` is scored not only against the draw it was made for (**lag +1**) but at every lag up to +30: the target draw and the 29 draws after it.

- If a model's signal were real but time-shifted, its average hits would peak consistently at some lag above 1. A flat curve across all lags means the hits come from draw-independent number-frequency structure, not timing.
- Main columns only. Positional games are scored positionally: pick3 and the market games by slot (pick3's chance level is 0.3 per draw), Joker+ by L + R. It reads every stored ticket.
- Only lags with at least 10 scored draws can be a peak.

Each run keeps only its single best peak per model, with a z-score against the model's own lag profile (a flat profile gives z around 1 whatever lag wins). The peak is appended to `data/database/lagPeakHistory.json`:

- one entry per run date; a second run on the same date replaces the first;
- a run is recorded only when the peak's sample size differs from the last entry, so a run that added no draw is not a new vote;
- the last 60 recorded runs are kept per model.

The page shows this run's peak and z-score, the most frequent peak lag across the recorded runs (highlighted as tracked once at least 3 runs exist and at least half agree), and a run-length-encoded peak trail (`+30×5` = holding still, `+26 → +28 → +30` = drifting).

Reading it: with under about 100 scored draws per lag the best lag moves around by chance. A real phase shift would show the *same* peak lag persistently over time, and plausibly across related models, not a one-off maximum.

### What if the peak lag had been played

On the Phase-shift page each lottery game has a folded table, *What if the peak lag had been played* (`phasewhatif.js`, `/api/admin/phase-shift/:game/what-if`). It stays on that page: no ranking, ticket or plan reads it.

- **The lag used**: for every draw, the row's peak lag in the newest run recorded before that draw's day (`lagPeakHistory.json`). A lag is never chosen with the draw in view, and a draw before the row's first recorded run is not counted.
- **What if**: the ticket the row made that many draws earlier, played on this draw and paid by the money rules of the History pages (the payout tables, or the draw's own prize table).
- **As played**: the row's own ticket for the draw (lag +1), over the same draws.
- **Any lag**: the average over every lag from +1 to +30 the day files can give for the draw: what a lag picked at random pays. A peak lag has that to beat, not only the row as played.
- A draw counts when the row has a ticket at lag +1 and at the lag used and both could be scored in euro. The market games have no what-if.
- Limits: the peak is found on hits and the what-if is money, so a few big prizes can carry a row (the column *Draws better* shows on how many draws the what-if paid more); and no control measures what the difference reaches by chance.
