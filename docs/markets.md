# Markets: crypto and shares

The crypto and shares track runs the lottery predictor's models on market prices: each trading day (and each week) is turned into a lottery-shaped draw, every model row predicts it, and the predictions are settled against the real returns and booked as paper money. It is a predictor with tracked, scored predictions, not a trading bot: every figure is paper money, nothing places an order, and nothing here is financial advice. This page covers the data and the store, the four market games, settlement, the market-specific models, the paper books, the week game, the guarded week, the crypto order book, and what the market pages and the admin workbench show.

## The research question

Do the principles that fail to find structure in a fair lottery find any in market prices? The same statistical, deep-learning, boosting and quantum model families, the same walk-forward evaluation and the same negative controls are applied. The bar is the efficient-market prior: accuracy above chance is read as predictability only after the controls and a proper score (see [research](research.md)).

## The four games

| Game | Draw | Instruments | Run |
| --- | --- | --- | --- |
| `crypto` | one per UTC day | BTC, ETH, BNB, XRP, SOL (USDT, Binance) | daily |
| `shares` | one per Euronext session | SHELL, INGA, ABI, KBC, UCB (EUR) | Monday to Friday |
| `cryptoweek` | one per ISO week | the crypto instruments | with `crypto` |
| `sharesweek` | one per ISO week | the shares instruments | with `shares` |

- A game is positional: an instrument is a slot, a return bin (0-9) is the digit. All four games have five slots.
- `Helpers.MARKET_GAMES` lists the four games; `Helpers.is_market_game` is the test the pipeline uses.
- The games are registered in the tuners' `GAME_CONFIG` and as positional games, so `DataLoader`, `Backtester`, `ModelFactory`, the tuners, the meta-learners and `Predictor.py` run them like pick3 or Joker+ (see [models](models.md)).
- A hit is the bin in the right instrument's slot, never a set intersection.
- The universe is fixed in the `instruments` table and is not re-selected, so no survivorship or look-ahead enters the results.

## Data sources

| Source | Used for |
| --- | --- |
| `binance` | crypto daily bars (`/api/v3/klines`, keyless, 1,000 per page, the running day's candle skipped until it closes), and the 1h and 1m bars of the order book |
| `euronext` | the shares' end-of-day download: about two years (730 days) per answer; today's row is kept only after 18:00 Brussels |
| `yahoo` | the one-time back-build of the shares' older history from the Yahoo listing (SHELL.AS, INGA.AS, ABI.BR, KBC.BR, UCB.BR), and the shares' live quote in the web app |
| `nasdaq`, `alphavantage` | implemented alternates, not in use; Alpha Vantage reads its key from the `ALPHAVANTAGE_KEY` environment variable |

- A source is operational, not identity: `--set-source` moves an instrument to another source without touching its slot.
- A source change clears the instrument's stored bars, because price bases differ between sources and cannot be spliced.
- A source symbol with an exchange suffix is refused for a USD or USDT instrument.

## The store

Everything lives in `data/markets.sqlite3` (`src/MarketData.py`). The file is gitignored; the tracked record is the text exports (see *Settlement and exports*).

| Table | Content |
| --- | --- |
| `meta` | store-level keys |
| `instruments` | the universe: market, slot, symbol, source, currency, active flag; the week games' instruments are rows of their own that read their bars through the base market's instrument |
| `bars` | daily open, high, low, close per instrument |
| `intraday_bars` | Binance 1h and 1m bars for the crypto order book |
| `game_days` | per game day and slot: return, bin and the edges the bin was cut with |
| `predictions`, `results` | every model's settled calls: predicted and actual bin, the three hits, the P&L |
| `ticket_published` | when a card was first seen published (order book) |
| `model_performance` | created, not written by any code |

- `install_universe` adds missing instruments and never changes a row; an entry whose slot is held by another symbol is reported as a conflict.
- Week-game instruments are never fetched.

## Fetching and checks

`FetchMarkets.py` is the command for the data layer. It takes no `process.lock`.

| Flag | Effect |
| --- | --- |
| `--init` | create the store and install the universe |
| `--update [--market M]` | fetch new bars |
| `--refetch-all` | with `--update`: refetch every instrument's full history (the manual repair) |
| `--check` | integrity report |
| `--status` | what is stored per instrument |
| `--set-source MARKET:SYMBOL SOURCE[:SOURCE_SYMBOL]` | move an instrument to another source |
| `--back-build` | join the older history (from 2016-01-01) onto the stored bars, once |
| `--db PATH` | another store |

How an update works:

- As-traded sources (Binance) refetch the newest 3 days.
- Adjusted sources (`nasdaq`, `yahoo`, `euronext`) serve history on the share basis of the day of the call, so their whole answer is refetched and rewritten in place.
- For a window source (`euronext`) the seam between the new window and the older stored bars is compared over 5 bars; a uniform factor beyond 0.05% rescales the older bars, so no log return moves.
- A uniform rescale of overlapping bars (at least 3 bars, agreeing within 1%, at least 5% from 1) is logged as `HISTORY RESCALED`; the older bars are put on the new basis when the full answer has at least 250 bars.
- `--back-build` is refused on an empty store or when the join factors disagree by more than 0.5%.
- No exception escapes the per-instrument update; a failing source leaves that instrument a day short.

What `--check` counts per instrument:

- Gaps: any missing calendar day for crypto, a hole of more than 4 calendar days for shares.
- Bad prices: non-positive or inverted.
- Jumps: a close-to-close log move beyond 0.35 (the detector of a split seam that slipped through).
- Staleness: days since the newest bar.

In normal operation nothing fetches by hand: each predictor run refreshes its market first (see *The daily runs*). `python3 -m src.MarketData` is the self-check (no network).

## How a day becomes a draw

`src/MarketGame.py` cuts the game from the bars.

- Returns are log returns on aligned days only: dates where every active instrument has a bar.
- Each instrument's return is cut into ten quantile bins (`K_BINS`). The edges are the quantiles of that instrument's returns strictly before the day, on an expanding window of at least 250 returns (`MIN_HISTORY`), so a day never leaks into its own label and the past never changes when a day is added.
- The bins are equally likely over the history each day is cut from, not within every stretch of it: volatility drifts, so a calm period over-fills the middle bins and a wild one the outer bins. The chance levels (10% exact, 28% adjacent, 50% direction) are the design target and approximately true; a row that keeps naming an over-filled bin beats 10% without skill.
- The draw is the vector of bins across the instruments, written to `data/trainingData/<game>/<game>-gamedata-NL-<year>.csv` in the lottery file shape (`Datum;Nummer 1;..`, newest first).
- The aligned log returns are written next to it as `<game>-returns.tsv` (every aligned day including the warm-up, oldest first); the market rows model this file. It is not a `.csv` because every `.csv` in a game folder is read as draws.
- Each day's edges are stored in `game_days`: a predicted bin only means something with the edges it was cut from.
- A predicted bin becomes a price through its representative return: the bin's midpoint, and for an open-ended bin the edge moved out by half the neighbouring bin's width.
- A refresh that moves a game's first day warns that `Predictor.py -r` rebuilds the stored days.

The Euronext calendar is computed by rule: sessions 09:00-17:30 Brussels, half days to 14:00 on 24 and 31 December, closed on 1 January, Good Friday, Easter Monday, 1 May, 25 and 26 December.

## The week games

`cryptoweek` and `sharesweek` have one draw per ISO week: the week's log return per instrument (the sum of its aligned daily returns), cut into ten quantile bins of the instrument's own past weeks, with at least 100 weeks before the first can be binned (`MIN_WEEK_HISTORY`). The draw is dated on the week's last aligned day.

- A longer horizon is a second game rather than a rolling five-day return, because every row predicts the next draw after the newest one.
- A crypto week is complete when a later ISO week has started (Monday 00:00 UTC). A shares week is complete from Saturday.
- The newest week is cut only once its last bar is in the store (`week_closed_by`: Sunday's bar for crypto; Friday's, or the last weekday before a closure, for shares). A lagging source costs a day, not a wrongly dated week.
- The week games are games like the others: every row runs on them, a new draw arrives once a week, they have day files and they are settled and exported like a market.
- GARCH, the Regime HMM rows and the RL Position Model run on the weekly returns with the same kind of knobs as on the daily games.

When the week ticket is made:

- `cryptoweek`: the run of Monday 00:05 UTC cuts the finished week and makes the ticket for the week that has just started. The weekly book's entry limit is the end of the week's first UTC day, so the ticket counts as followable.
- `sharesweek`: the shares job runs Monday to Friday at 18:05 Brussels. On Friday the week is not yet complete (that is from Saturday), so the finished week is cut by the next run, Monday 18:05. The ticket for a week is therefore made on that week's Monday evening, after the Monday 09:00 open at which the weekly open-to-close book buys. Consequences: the followed-plan record never marks a shares week as live (the live test is ticket time before Monday 09:00 Brussels), and the guard has no current week between Friday's run and Monday evening. A run at the weekend (a caught-up Friday run or a hand run) is the only way the ticket exists before Monday's open.

## The daily runs

Two scheduled predictor jobs carry the markets (`jobs.js`; the schedule itself is on [operations](operations.md)):

- `Predictor.py -a true -g crypto,cryptoweek`, daily at 00:05 UTC.
- `Predictor.py -a true -g shares,sharesweek`, Monday to Friday at 18:05 Brussels, after the Euronext close, so the shares card is up the evening before the session it is for.

For a market game `Predictor.py` does what it does for a lottery game, with three differences:

- Where a lottery downloads its CSV, a market calls `MarketGame.daily_refresh`: install the universe, fetch new bars (crypto also its 1h bars), re-cut the bins and rewrite the yearly files. A source that does not answer leaves the history a day short and the run predicts from what is on disk.
- Each ticket is stamped with `madeAt` in its day file (`stampTicket`).
- After the performance report, `MarketSettle.settle_and_export` settles and exports each market game. A failure there costs the pages a day, never the predictions.

On its first run for a new market game `Predictor.py` back-builds 31 draws (31 weeks for a week game): the statistical, boosting and meta rows for every day, the deep-learning rows for the newest day only.

`MarketsDaily.py` is the hand-run refresh and settlement. It takes `process.lock`.

| Flag | Effect |
| --- | --- |
| `--market` | any of the four games; default both markets, then the week games |
| `--no-fetch` | do not contact the sources; cut from the stored bars |
| `--settle` | also settle every stored day and rewrite `data/markets/<game>.json` and the results CSVs |
| `--db`, `--root` | another store or checkout |

## Settlement and exports

`src/MarketSettle.py` settles every stored day file (`data/database/<game>/<Y-M-D>.json`) against the real returns.

Per model and instrument a settled call carries:

- Exact hit: the predicted bin is the actual bin (chance 10%).
- Adjacent hit: within one bin (chance 28%).
- Direction hit: the sign of the predicted bin's representative return against the real return (chance 50%).
- The return-unit P&L of the fixed rule: long when the call is up, the real return minus a 0.1% fee (`FEE`). This scores calls, not money, and is the same for every market.

A call is up when its bin is in the upper half (5-9). A sized row (the RL Position Model) is up when its size is above zero and scales the position by its size (`position_of`).

Exports:

- `data/markets/<game>/results-<year>.csv`, the tracked text record, with columns `date;model;symbol;predicted_bin;actual_bin;return;hit_exact;hit_adjacent;hit_direction;pnl;pnl_cash`.
- `data/markets/<game>.json`, what the pages read: instruments, per-model rates and money, the best model, the `trading` block (stake, costs, rules, the books, benchmarks, `followed`, `guarded`, `orders`), the next day's calls with `made_at`, the daily accuracy series and the newest 30 settled days in detail.
- `data/markets/<game>-regimes.json`, the Regime HMM rows' reading (written by the live run).

A model with fewer than 10 settled days (`MIN_RANK_DAYS`) ranks after the others and cannot be the best model or head a book.

## Money

| | Crypto | Shares |
| --- | --- | --- |
| Stake per position | 100 USDT | 1,000 EUR in whole shares (rounded down at the entry price) |
| Cost per leg | 0.1% | broker fee by order value plus 0.35% stock exchange tax |

- The shares' broker fee is tiered: 2.50 EUR up to 250, 5.00 up to 1,000, 7.50 up to 2,500, then 15.00 per started 10,000 (`BOLERO_TIERS`, `BOLERO_PER_10K`, `TOB`). At the stake this is about 0.85% a leg (`leg_rate`), so a daily round trip needs a move of about 1.7% to break even.
- A share position whose stake buys no whole share is not taken.
- Costs apply to every buy and every sell in every book (`MarketSettle.Costs`).
- The RL Position Model and its tuner price a position from its return at a rate; for shares that rate is `leg_rate`.
- The books export each position's move next to its money, because whole shares and fixed fees hide the move in the money.

## Paper books

`MarketSettle.ledger` books every model's calls day by day under three rules, each with a market benchmark that buys everything under the same rule, stake and costs.

| Rule | Position | Benchmark |
| --- | --- | --- |
| Daily | every up call is a round trip: bought at the previous game day's close, sold at the day's close | every instrument bought every day |
| Hold | a position is kept while the calls stay up and sold at the close of the last up day: one cost pair per run, the stake compounds; an open position is valued at the newest close | everything bought on the first settled day and held to the last |
| Open to close | bought at the session's open, sold at its close; for a week game the week's first open to its last close | every instrument bought at the open every day |

- Open to close is exported for `shares` and `sharesweek` only: it is the rule a reader of the shares page can follow. Crypto's UTC day opens at the previous close, so its open book would repeat the daily one.
- The daily book is the model's call scored on the daily candle, not what a reader could have made: for crypto it buys at the 00:00 UTC close, before the ticket of that day's run exists.
- Each rule has a long-and-short twin: a lower-half bin is a short (sold at the previous close and bought back at the close; kept while the call stays down; or sold at the open and bought back at the close), with the same stake and costs. Its two benchmarks are buying everything and selling everything. A sized row never shorts. Shorts are paper only: selling what one does not own needs a margin or derivatives account, and no card gives a short order.
- Headline books are taken among rows with at least 10 settled days.

## Following the plan

`MarketSettle.followed_plan` answers what doing exactly what the page suggested would have made.

- On every settled day it picks the row the page's rule would have followed, from the record as it stood before that day: the best book under the plan's rule among rows with at least 10 positions (5 for a week game), else the best exact rate. `markets.js` (`pickPlan`, `pickWeek`) picks the same way for the pages.
- The plan's rule is open to close for the shares games and the daily round trip for the crypto games.
- The record books that row's money on the day under the same rule, next to buying everything under it.
- A day is *live* only when its ticket was made before the plan's entry (`entry_instant`): the Euronext open of the day for shares; 09:00 Brussels of the first weekday after the previous game day for a shares week; the end of the week's first UTC day for a crypto week; the day's close for a crypto day.
- The ticket's time is the day file's `madeAt`, else the file's modification time.
- Days that are not live (back-built history, a late or missed run) are shown apart and left out of the live total.

## The guarded week

The guard combines the week call with the daily calls (`MarketSettle.guarded_week`, exported as `trading.guarded` of the week games).

- The week call is the permission, the daily call the trigger: an instrument the week's followed row calls up is bought at the entry of the first day of the week whose daily call for it is up.
- It is held while the daily call does not turn flat or down. A day without a daily call keeps a held position and buys nothing.
- It is sold at the close of the first day whose daily call is flat or down, bought again on the next up call, and sold at the week's close at the latest.
- Consecutive held days are one position with one cost pair; a sized week row's size scales the stake.
- A daily call is *up* (upper-half bin, or size above zero), *flat* (the bin just below the middle, or a sized row at 0) or *down*.
- A guarded week is live only when the week ticket is live, every daily ticket of the week is live, no day is missing and every position is priced.
- Weeks before the daily record exists carry the plain week (held to the week's end).
- `guarded_now` gives the current state per instrument (hold, sell, buy, out, with the reason); it has no answer while the running day's week is not the week of the newest week ticket.
- `GUARD_FORWARD_FROM` (`2026-10-19`) marks the first week booked forward; the user pages count the guard from `GUARD_SHOWN_FROM` (`2026-10-26`) and the plain week before it. The guard's whole book is on the admin workbench.

## The crypto order book

`src/MarketOrders.py` books exactly what the crypto day and week cards tell a reader to do: buy at market when the card appears, place a sell limit, sell at market at a deadline or on an exit call. It is a separate book next to the others, exported as `trading.orders` of `crypto.json` and `cryptoweek.json`, and it is what the crypto user page shows and counts for cards from `ORDERS_SHOWN_FROM` (`2026-11-02`) on.

It reads Binance intraday bars: 1h bars continuously from `INTRADAY_SINCE` (`2026-10-12`) and 1m bars in a two-hour window from each card's publication hour. A card's publication is recorded the first time it is seen (`ticket_published`); when no publication is recorded, the ticket's `madeAt` plus 15 minutes is used and the order is flagged *estimated*.

### Rules

- **Entry**: the open of the first 1m bar at or after the card's publication. Never priced from an hourly or daily bar; a missing 1m bar leaves the order *pending*.
- **Limit X**: set from the predicted bin, anchored at the previous game day's close (day) or the previous Sunday's close (week).
- **No order**: a target less than 0.3% above the base close ("target below fees"), a price already at or above X ("already reached"), or X less than 0.3% above the entry ("too close to the limit"). Counted with money 0, never a position.
- **Fill**: at X, never better, at the first bar whose high is at least X × 1.0005; a touch is not a fill. A fill and a card in the same minute: the card wins.
- **Exit**: the first of the fill, a card-time exit, or the deadline. The deadline is the publication of the next card of the same game, or 12 hours after the interval's close when no card has appeared. Card-time exits are priced at the open of the 1m bar at or after the card.
- **Money**: the stake on the move from entry to exit, a 0.1% fee on both legs, scaled by a sized row's size.
- **Day game**: the followed day row's up calls. Roll: the same coin called up again on the next card with a new limit at least 0.3% above the price keeps the position, moves the limit and costs no fees; the position is booked on the day it closes.
- **Week game**: the followed week row's up calls are the permission, Monday's followed day call the trigger (not up, no Monday buy). One round trip per coin per week, no buy-back. The day card that turns the coin flat or down is the early exit. Entry at the later of the week card and Monday's day card.
- An order whose card was published after its interval's end is *not followable*.

### Separate ledgers

- **Late entries** (week only): on each day card Tuesday to Sunday, when the week call and that day's daily call are up, the price is at least 1% below X and the deadline is at least 24 hours away. Each is one cohort position with the same X, early exit and deadline; never part of the user line.
- **Newcomer entries** (day only): on every card that rolls a held position, a buy for the reader who did not buy before, at that card's price with the same limit, deadline and no-order tests; it then rolls like the main position. Its own line, never part of the user line.
- **Benchmark**: every active coin bought at every card's publication without a limit and sold at the same deadline.
- **Reference** (admin only): the same orders ending at the interval's own close (00:00 UTC) with no roll.

### Limit variants

All three are booked; the admin workbench shows them side by side.

| Variant | Limit |
| --- | --- |
| `edge` (primary) | the predicted bin's lower edge: a right call fills |
| `mid` | the bin's representative return, the price the pages draw |
| `fixed` | entry × 1.01 (day) or × 1.04 (week): a control that ignores the bin |

The user line books per card the variant that card showed. A switch of the primary is a logged decision at the review point (`REVIEW_RULE`: one review at 60 live day orders or 31 January 2027, whichever is later, on a paired comparison whose lower bound is above 0), never a "best so far".

### Constants

| Constant | Value |
| --- | --- |
| `ORDER_SINCE` | `2026-10-13`, the first day the book covers |
| `MIN_GAP` | 0.003 |
| `FILL_THROUGH` | 0.0005 |
| `PUBLISH_LAG_FALLBACK` | 15 minutes |
| `DEADLINE_FALLBACK_HOURS` | 12 |
| `LATE_MIN_GAP`, `LATE_MIN_LEFT` | 0.010, 24 hours |
| `FIXED_GAP` | 0.01 a day, 0.04 a week |

### Commands

```
python3 -m src.MarketOrders                      # self-check, no network
python3 -m src.MarketOrders --backfill 2026-10-13 --root <scratch> --db <scratch>/data/markets.sqlite3 [--export] [--no-fetch] [--lag MIN]
```

The backfill is a module command, not a pipeline entry point: it writes only the store and root it is given, takes no `process.lock`, and refuses a root or store inside a git checkout unless `--allow-checkout` is given.

## Market models

Besides the rows every positional game gets (see [models](models.md)), the market games have rows of their own.

### GARCH Model

- A GARCH(1,1) per instrument by maximum likelihood on the newest returns; the predictive distribution of the next return is integrated over the ten bins.
- It predicts volatility, not direction: bin accuracy above chance from this row is volatility clustering, the reference every other row is read against.
- Its ticket is the most likely bin per instrument.

### Regime HMM Model

- One full-covariance Gaussian hidden Markov model per market over every instrument's day return, rolling volatility and rolling mean return (both built from returns strictly before the day), z-scored on the training window. The design follows arXiv [2603.04441](https://arxiv.org/abs/2603.04441); its allocation layer is not built.
- Refit on every prediction day on an expanding window. The number of regimes is reselected every `regimeHmmReselectEvery` game days by the one-step-ahead predictive log-likelihood of a validation slice, penalised per parameter.
- The next return per instrument is the regime mixture of the return coordinate's Gaussian, conditioned on the next day's known features and integrated bin by bin.
- The EM is written out in numpy (no `hmmlearn`, no `arch`): passes that underflow or flush a reachable regime are redone in log space, and the transition matrix takes one pseudo-transition per cell (`TRANSITION_PSEUDOCOUNT`).
- The ticket of the three HMM rows is the bin that holds the mixture's predictive median, so the up call follows the forecast's location rather than the asymmetry of the bin edges. The probabilities are the same either way.

Two ablations run as rows of their own and read the same `regimeHmm*` keys:

- `Regime HMM ZeroMean Model`: return means fixed at zero, regimes carry variance only. If the full row is not better than this one under a proper score, the regimes add nothing beyond volatility.
- `Regime HMM Single Model`: one regime, the homoskedastic model.

In the live run each fitted regime is matched to a persistent template by the closed-form 2-Wasserstein distance between Gaussians (one-to-one assignment, a distance above the threshold spawns a template). Templates are kept under `data/models/<game>/`; the reading goes to `data/markets/<game>-regimes.json`. The probabilities never depend on the labels.

### Knobs

All knobs live in `bestParams_<game>.json` and are read through `MarketModels.build_market_models`; a missing key takes its `MARKET_DEFAULTS` value. `useGarch` and `useRegimeHmm` (both default on) switch the rows off in `Predictor.py`.

| Key | Default |
| --- | --- |
| `garchMean` | `constant` (or `zero`, `ar1`) |
| `garchDist` | `t` (or `normal`) |
| `garchWindow` | 2000 |
| `regimeHmmMin`, `regimeHmmMax` | 2, 5 |
| `regimeHmmPenalty` | 1.0 nat per parameter |
| `regimeHmmValidation` | 250 |
| `regimeHmmReselectEvery` | 20 |
| `regimeHmmWindow` | 0 (expanding) |
| `regimeHmmFeatures` | `returns,vol,mom` |
| `regimeHmmVolLookback`, `regimeHmmMomLookback` | 60, 20 |
| `regimeHmmShrinkage`, `regimeHmmMinCovar` | 0.1, 1e-4 |
| `regimeHmmRestarts`, `regimeHmmMaxIter` | 2, 100 |
| `regimeHmmTemplateRate`, `regimeHmmTemplateThreshold` | 0.1, 1.0 |

### RL Position Model

The RL row of a market game sizes positions instead of building a ticket (`src/RLPositionModel.py`).

- Per instrument and day it chooses a size from 0, 0.5, 1, 1.5 or 2 times the paper stake. A size of 0 is a sit-out.
- Features, all in bin space: the share of voters calling the upper half, the mean of their bins, the GARCH, Regime HMM and WeightedEnsemble bins, the instrument's newest realised bin, how wild its last five realised bins were, and a bias.
- The voters are the `WeightedEnsemble Model`'s voters, with XGBoost and LightGBM counted as one voter.
- Policy: a linear softmax over the sizes, trained by REINFORCE with a per-state baseline on the game's stored day files, warm-started from `data/models/rl_position/<game>_position_policy.json`, wall-clock capped. A policy file without `featureVersion` 2 is not loaded.
- Reward: money − `rlPositionRiskAversion` × money², the money being the daily rule's paper result scaled by the size. The quadratic term makes a middle size the right answer for an uncertain call.
- Its ticket is the WeightedEnsemble vote's ticket; its own contribution is `positions`, the size per instrument, which settlement stores and every book applies.
- Without a trainable history and without a stored policy it plays the plain rule: size 1 where the vote is up, 0 elsewhere.
- With `rlPositionSitOut` true the row shows the vote's ticket and takes no position; the tuner's gate sets this key (see [tuning](tuning.md)).

| Key | Default |
| --- | --- |
| `rlPositionLearningRate` | 0.05 |
| `rlPositionEpochs` | 30 |
| `rlPositionSamplesPerDay` | 16 |
| `rlPositionTrainDays` | 120 |
| `rlPositionRiskAversion` | 0.1 |
| `rlPositionMaxTrainSeconds` | 60 |
| `useRlPosition` | true |
| `rlPositionSitOut` | false |

## Rows under a proper score

`MarketRows.py` judges the market rows by a proper scoring rule before any hit rate or paper P&L is read.

- It backtests the market rows together with the positional base rows walk-forward and scores every row by the mean log-score of the probability it gave the bin that happened. The uniform forecast scores log(1/10) = −2.303; a probability below 1e-4 is scored as 1e-4.
- Every difference to GARCH, to uniform and to the column frequencies carries a 95% paired bootstrap interval over days. *Adds information beyond GARCH* means the interval's lower bound is above zero.
- Three baselines are scored the same way: `random` (1/K per bin), `column_frequency` (each position's bin counts over the earlier game days, add-one smoothing) and `global_frequency` (the same counts pooled). A row is credited beyond the frequencies only when its `vs_frequency` interval lies above zero.
- Hit rates, the reliability of the probabilities by bin and the fixed rule's P&L are reported after the score, as information.
- Lockbox days are withheld.
- The record is `data/controls/markets/<game>-rows.json`; the verdict is shown on Admin > Controls.

```
python3 MarketRows.py                              # crypto and shares, newest 250 game days
python3 MarketRows.py -g crypto -d 60 --market-only
python3 MarketRows.py --root /elsewhere/checkout --seed 1
```

It takes `process.lock`. It runs as the last job of the Sunday controls plan with `-g crypto,shares,cryptoweek,sharesweek -d 250`.

## Tuning and controls

- Tuning of the market games (the `Garch` and `RegimeHmm` strategies scored by the proper score, the position studies of the RL tuner, the ensemble tuner, the narrower Markov box of the week games, the gates) is on [tuning](tuning.md). The boosting tuner tunes no market game, and the quantum tuner skips the week games unless named.
- The controls take all four market games. A market's synthetic control is a geometric random walk with each instrument's own mean and standard deviation, cut by the game's causal quantile rule; its shuffled control reorders whole days of returns and re-cuts the bins by the same rule. Both carry their own returns file, so the market rows run on a control as on the real history (`src/ControlHistories.control_market`). The controls, the lockbox and how they are read are on [research](research.md).
- Ranking and the performance report are on [scoring](scoring.md).

## What the market pages show

The web app has two user pages, `/crypto` (Crypto) and `/shares` (Stocks), served from `GET /api/markets/:market` (`api.js`, `marketView`). They show the followed plan as positions, in the reader's perspective ("if you bought", "if you have not bought").

- A status strip of six tiles: what following the page made (daily and weekly together), the comparison with buying everything, positions, days, instruments and what is open now.
- Open positions, with live prices from `GET /api/quotes/:market`: Binance for crypto, Yahoo's quote of the Euronext listing (about 15 minutes late) for shares, cached for a minute; `{at: null, quotes: {}}` when unavailable.
- Three charts: the cumulative result, the result per instrument and the result per day.
- A trades log and the closed positions.
- The weekly part counts the plain week before `GUARD_SHOWN_FROM` and the guard from then on.
- On the crypto page, cards from `ORDERS_SHOWN_FROM` on are the order book's: buy now at market, the sell limit, when to sell at market, and separate lines for late and newcomer entries. Until a card of that day is on the page the page keeps the book-based card.
- The page states that every figure is paper money and where the book's entry differs from a reader's.

## What the admin workbench shows

`/admin/markets/crypto` and `/admin/markets/shares` (Crypto workbench, Stocks workbench) show everything in detail.

| Card | Content |
| --- | --- |
| Today's plan | the followed model's calls as orders, opening with labelled lines: one model's calls, following this card so far, the sell |
| Week game | the week card: the followed weekly book's calls, the band at the week's close, how far the week has gone |
| Order book (day card, week card) | crypto only: the three variants, the reference, late and newcomer ledgers, per-order tables |
| Following the plan | day to day and week to week, against buying everything, live days apart from the others |
| Paper books | daily, hold, open to close and their long-and-short twins, every model switchable, against the market |
| Models | per model the rates against chance and the totals of each book |
| Accuracy per day | mean exact rate of every row, the day's best row, chance |
| Coins / Stocks | per instrument the closes or moves with a model's calls and bands |
| Day by day | the newest 30 settled days as returns, bins and money |
| The guard on the week positions | the guarded book, week by week |
| Regime reading | which regime template the HMM rows believe the market is in |
| Background: how a day becomes a draw | the worked example, the calendar and the chance levels |

Routes (`workbench.js`, read-only, admin only):

- `GET /api/admin/market/:market/workbench`
- `GET /api/admin/market/:market/workbench/books/:rule` with rule `daily`, `hold`, `open`, `daily_ls`, `hold_ls` or `open_ls`
- `GET /api/admin/market/:market/workbench/orders/:game`
- `GET /api/admin/market/:market/workbench/instrument/:symbol`

`markets.js` holds the loaders, the view models and the plan and week picks; it has no routes. The web app as a whole is on [operations](operations.md).

## Self-checks

All are part of `npm test` and need no network:

- `python3 -m src.MarketData`, `src.MarketGame`, `src.MarketSettle`, `src.MarketOrders`, `src.MarketModels`, `src.RLPositionModel`, `src.ControlHistories`
- `python3 -c "import MarketRows; MarketRows._self_check()"`
- `node test/markets.test.js`, `node test/api.test.js`, `node test/workbench.test.js`
- `python3 -c "import SwitchSharesEuronext; SwitchSharesEuronext._self_check()"`, the self-check of `SwitchSharesEuronext.py`, a one-off tool that replaces the shares universe

## Not built

- News and social-sentiment feeds. If added, a sentiment score would be an indicator only (one more meta-learner feature column, held to the irrelevant-feature control), never a predictor row.
- The allocation layer of the regime paper: the track builds a predictor, not a trading bot.
