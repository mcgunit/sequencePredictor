# Roadmap: open items

This page lists what is open or partly open in the project: for each item, what is built (with a link to the page that documents it) and what remains. Completed items are named in one line each at the end; their decisions, measurements and findings are in the changelog. Item numbers are kept, because code comments cite them ("roadmap item 8").

## Overview

| Area | Open items |
|---|---|
| Lottery | best-combination search on profit; tuners and meta-learners on profit; lotto multi-pick (item 10); positional soft vote (item 7); the Council rows (item 12) |
| Markets | item 11: instructions from a paper trader instead of fixed intervals, with the month game; item 4: Bolero tiers, the order-book review, sentiment feeds, the first shares tuning reading |
| Users' side | a beginners' page |
| Research | item 6: Q2 extensions, Q3, Q4; dead deep-learning settings; the deep-learning tuner's lockbox; the control readings |
| Platform | item 8: a mail on failure, a predictor job per game; item 9: day files to SQLite; cleanup |

## Lottery

### Best-combination search on profit for the prize-table games

**Built.** The combination report (`Helpers._build_combination_report`) ranks portfolios of rows per game, with a shuffled-history control; it is shown under Admin > Models. Keno, Pick3 and Joker+ are ranked on money. See [scoring](scoring.md).

**Remains.** Lotto, EuroMillions, EuroDreams and Vikinglotto are scored in euro per row from each draw's prize table, but the combination search still ranks them on hits (the best line held per draw). Moving it to profit also needs the shuffled control to carry a prize table for every shuffled draw.

### Tuners and meta-learners on profit for the prize-table games

**Built.** Prize tables are fetched with the draws and used by the daily scoring and the performance report (`src/PrizeTables.py`). See [scoring](scoring.md).

**Remains.** No tuner and no meta-learner training reads the prize tables: for Lotto, EuroMillions, EuroDreams and Vikinglotto they still optimise hits. See [tuning](tuning.md).

### 10. Lotto multi-pick: from three extra numbers to a rethought first six

**Built (step 1).** Every lotto row carries three extra numbers for a 7-, 8- or 9-number system play, the numbers are stored most probable first, and the extra numbers are scored against the exact chance levels (`src/MultiPick.py`). See [scoring](scoring.md).

**Remains.**

- Step 2: read the 7th to 9th numbers after about fifty draws against chance: 13.3% per extra number, 8.39% for three of six among nine. The reading belongs with the control readings: a hit rate above chance over a few weeks is what the best of some thirty rows shows by luck, unless it stays.
- Step 3: if the extra numbers carry signal, rethink the first six from the same ranking (argmax by ranking).
- Step 4: the euro value of a 7- to 9-number system play. The prize per rank per draw it needs is available from the prize tables; the system-play value itself is not built.

### 7. Vote per position for the positional games: phase 2

**Built (phase 1).** `WeightedEnsemble Model` and `SubsetEnsemble Model` vote per slot on the models' final tickets for Pick3 (Joker+ is covered but off behind `useJokerplusEnsemble`). See [models](models.md) and [tuning](tuning.md).

**Remains.**

- Phase 2, the soft vote: blend each model's normalised `score_positions()` distribution into the slot vote with a tuned mixing weight. It first needs a `positionScores` sidecar in the day file, because the history rebuild's second step runs in a separate process and only sees the day file. No such sidecar is written.
- Leaving the ensemble row out of the RL Ticket row's Pick3 vote features, so an aggregate is not counted next to its own members. Deferred: the RL row treats Pick3 like every other game.

### 12. Council rows: tickets from the LLMs

**Built.** The Council (`/app/council`, `src/llmCouncil/`): local LLMs answer questions, one model combining the others' answers. See [operations](operations.md).

**Remains.** Tracked rows whose ticket comes from the Council, scored like every other row. Two kinds, each as one row per Council member and one row for the Council's combined answer, so the ranking shows whether combining helps:

- **From the rows' record.** The LLM reads a summary of what the other rows predicted on recent draws and what was drawn, and gives a ticket for the coming draw. It is a meta-learner done by an LLM: its comparison is the weighted ensemble and the RL ticket row. The risk to read for is that it follows whichever row did well lately.
- **From the draws alone.** The LLM reads only past draws and gives a ticket. On draws that test fair this is expected to behave as a biased random generator; it serves as a control, and its favourite numbers show as repetition.

What the build has to settle:

- **A job of its own, after the predictor.** A slow or failed LLM never holds up the predictor; the row is absent that day, as a foundation row is. The Council runs on the CPU worker, so the job wakes it.
- **Forward only.** The rows are built for the live day and never inside a history rebuild (one LLM call per row, game and day); they are scored from their first day.
- **A valid ticket or none.** An answer that is not a ticket of the game (range, count, distinct numbers) is asked again a fixed number of times and then dropped. The prompt, the model, the temperature and the seed are recorded with the row.
- **What fits the context.** The whole history of every row does not fit a local model: the first kind reads a summary (recent draws, each row's ticket and result).
- **Which games.** The lottery games first; the market games after the first readings.

## Markets

### 4. Same models for crypto and shares

**Built.** The whole phase plan: the data layer and the markets store (M1), the quantile-bin games with settlement and paper books (M2), the market rows, the proper-score report and the market controls (M3), tuning for the market rows, the RL Position Model, the meta-learners and the ensemble tuner (M4), the week games (M5), shorts, the guarded week, the crypto order book, and the five Euronext shares with whole-share positions at Bolero's costs. See [markets](markets.md).

**Remains.**

- **First shares tuning reading.** Read the first weekend tuning of the five Euronext shares: which rows the gates replaced, and whether the tuned rows differ from their defaults.
- **Bolero costs above 1,000 EUR a trade.** `BOLERO_TIERS` in `src/MarketSettle.py` holds tiers up to 2,500 EUR and a rate per started 10,000 EUR above that, but the tiers above 1,000 EUR are not checked against Bolero's tariff sheet. They only matter for a sale after a large rise.
- **Order-book review.** One review of the crypto order-book variant after 60 live day orders or on 31 Jan 2027, whichever is later (`REVIEW_RULE` in `src/MarketOrders.py`).
- **Sentiment feeds.** Sentiment signals (CryptoPanic, LunarCrush, QuantMuse) as meta-learner features, not as predictors. Not built.

### 11. Instructions from a paper trader, not from fixed intervals

**Built.** Day and week games per instrument, each with a plan card whose instructions end at the game's close: buy at the open of the interval, sell at its end ([markets](markets.md)).

**Decided (15 November 2026).** Tying the holding period to the prediction interval was the wrong choice. The models keep predicting a day, a week and a month ahead; the instructions come from a paper trader that holds a position for as long as its rules say, and the card translates that trader's state and results into the same plain wording as today ("if you have it", "if you do not have it").

**Remains.** All of it:

- **The month game.** A month-ahead prediction per instrument, for crypto and for shares, beside the day and week games and scored like them. It is a prediction, not a holding period, and it is one of the inputs of the instructions.
- **The paper trader.** One per instrument, in or out of a position. It enters and leaves on rules that read the day, week and month predictions together (a signal that persists, a signal that turns, a stop, a longest holding time), not at an interval's close. Costs as in the paper books today.
- **Shares: long term, long only.** No shorts (the costs), and never a buy and a sale of the same position on one day. The month prediction weighs most, and later the sentiment feeds (item 4).
- **Crypto: both directions.** Shorts are allowed, and in an extreme case a buy and a sale on the same day.
- **The card.** The instruction follows from the trader's state: hold or sell for a reader who has the position, buy or wait for one who has not. Beside it the trader's own result since its first day, with two comparisons: holding the instrument throughout, and entries at random with the same time in the market. Without those a good result cannot be told from a rising market or from holding less.
- **The day and week cards.** The day, week and month games stay as scored predictions and become inputs only: the reader gets one card per instrument, the trader's, and no instruction that ends at a day's or a week's close.
- **The crypto follower's book** (a book that buys on an up ticket and sells at 00:00 UTC, once listed under item 4) is not built: this item replaces it.

## Users' side

### A beginners' page

**Built.** The research notice: a sticky footer on every page of the app (`web/src/components/Disclaimer.tsx`), in step with the README's disclaimer.

**Remains.** A beginners' page that explains the pages to a reader with no experience of the games or the markets. Its text is outward-facing and is proofread before it goes live.

## Research

### 6. Quantum research: what phases Q0 to Q4 still need

The design of the track, the controls and how to read them are in [research](research.md). Suggested order: Q2's extensions, then Q3, with Q4 only if the earlier phases leave a reason.

| Phase | State | Remains |
|---|---|---|
| Q0 classical controls | built | the dead LSTM layer-count setting (below) |
| Q1 quantum meta-learning | built | nothing in code; the comparison of the quantum rows with their classical control is read from the ranking over time |
| Q2 randomness discrimination | built, runs weekly | other window lengths and feature families; a rule-change map |
| Q3 direct quantum scoring | not built | all of it |
| Q4 candidate search | not built | all of it |
| Negative-control suite | built | reading the results |

**Q2: randomness discrimination, extensions.** `RandomnessDiscrimination.py` compares windows of real draws with synthetic fair histories. Open:

- Other window lengths (the `-w` flag exists; the scheduled run uses 10 draws) and other feature families than the multi-hot counts.
- A rule-change map: synthetic histories with the exact rules of each period. A synthetic history reproduces today's rules over the whole calendar, so a longer window over a rule change differs for that reason alone. Only the era start per game (`Helpers.GAME_ERA_START`) is applied.
- Repeating the experiment across several chronological splits, next to the several synthetic seeds it already uses.

**Q3: direct quantum scoring.** The quantum meta-learner rows already emit a per-number probability and reuse ticket construction, special columns, Keno subsets, the Backtester and the pages. Q3 adds:

- a quantum model that scores every candidate number from draw-history features rather than from the base models' scores;
- routed through `Backtester(collect_scores=True)` like a base model;
- compared on calibration and ranking before final ticket hits.

**Q4: candidate search.** A cost study, not a predictability study, and last in order:

- a ticket-level score and threshold;
- a toy Grover oracle for a deliberately small candidate space, in the NumPy simulator;
- quantum search cost against direct classical ranking;
- reporting of circuit depth, native two-qubit gate count, noise sensitivity and data-loading cost.

**Fuller layout.** A `src/quantum/` package and `experiments/quantum/` scripts only when Q3 or Q4 code arrives; the quantum code lives in `src/QuantumModels.py`.

### Dead deep-learning settings

- `num_lstm_layers` in `src/LSTM.py` is stored and written to the model's metadata but not used when the network is built; the deep-learning tuner offers only the value 1. Wire it or remove it.
- `SelfAttentionBlock` in `src/LSTM.py` is defined and never instantiated (the classes of the same name in `src/TCN.py`, `src/UnifiedLstmTcn.py` and `src/UnifiedLstmGruTcn.py` are used). Wire it or remove it.

### The deep-learning tuner's score outside the lockbox

**Built.** `HyperoptDeepLearning.py` runs behind a gate with resume, pruning and the production training budget. See [tuning](tuning.md).

**Remains.** Its validation split is the last 20% of each retrain's history, which contains the lockbox period; the gate records say so. Open: score on validation windows that leave the lockbox out, like the other tuners.

### Control readings

**Built.** The Sunday control jobs and the reading the scheduler writes to `data/readings` when the last of them finishes. See [research](research.md) and [operations](operations.md).

**Remains.** Reading them, week after week: with many rows tested against a two-sigma band, one clears it by chance in many weeks, so only a result that stays counts.

### Later, only if the tracked results justify it

- Fine-tuning a foundation model instead of using it zero-shot.
- A third foundation checkpoint: a subclass of `src/FoundationModel.py` and a worker script.

## Platform

### 8. Pipeline scheduling inside the server

**Built.** The server's scheduler (`jobs.js`) owns the daily predictor runs, the Saturday tuning chain, the Sunday controls and the on-demand jobs. See [operations](operations.md).

**Remains.**

- A mail to the administrator when a scheduled job exits non-zero. There is no mail code in the server.
- A predictor job per game, so one game's failure cannot cost the others their run. The predictor runs as three jobs per group: crypto at night, the lottery games in the morning, the stocks in the evening.

### 9. Day-file storage: from one JSON per game and day to SQLite

**Built.** Nothing; the store is one JSON file per game and scored day under `data/database/<game>/`.

**The cheap first step.** Every day file carries two raw probability arrays, `currentPredictionRaw` and `newPredictionRaw`: the predictor copies the previous file's array into the new file and nothing reads them afterwards (the two helpers that did, `extractFeaturesFromJsonForRefinement` and `extractFeaturesFromJsonForDetermineTopPrediction`, have no caller). Dropping them shrinks every new day file without a migration; the model rows, with up to three variant rows per model, are the larger part of a file. The arrays are kept for now.

**What a migration has to carry.** The day files are written by:

- `Predictor.py`: the fresh day, both steps of the history rebuild, `update_matching_numbers`;
- `HyperoptDeepLearning.py`'s rebuild;
- `src/MarketSettle.py`, which settles the market days.

They are read by:

- the app's JSON: `workbench.js` and `lottery.js` (a game's draws, one draw and the coming one, scored by `draws.js`);
- `Helpers.generate_model_performance_report`, which scans every file of every game in each run and feeds the ranking, the null band, the since ranking and the combination search with its shuffled control;
- `HyperoptRLTicket.py` and `HyperoptEnsemble.py`, which learn from the stored days.

**The natural shape.** One SQLite file next to `db.sqlite3` and `data/markets.sqlite3`, with tables `days` (game, date, real result), `rows` (day, model, ticket, hits, profit) and `raw` (the arrays, if kept). The daily commit of that data becomes a text export per game and year, the way the markets keep `data/markets/<market>/results-<year>.csv`, so the repository keeps a diff-friendly record and the binary store stays out of it. The predictions and results of a day are then one query instead of a file per game, and the page reads stop scaling with the number of days.

**Trigger.** Not size on disk. Do it when the pages need queries the files cannot give cheaply (a model's history over years, a per-instrument view across markets), or when the report's scan over every file starts to cost seconds in the daily run. Until then, dropping the raw arrays is the lever.

### Cleanup

- Remove the old `Council_Integration.md` at the repository root.
- Remove stale shares control files left in the controls folder on the production machine.

## Completed

- **1. Settings dropdown removed.** The global model filter and the unused Keno form are gone; every page shows every row.
- **2. Best combination of models.** The portfolio report with its shuffled control, and the tuned `SubsetEnsemble Model` row ([scoring](scoring.md), [tuning](tuning.md)).
- **3. Login.** A login form with roles, sessions and user management ([operations](operations.md)).
- **5. Foundation models.** `Chronos Model` and `TimesFM Model` as zero-shot per-position rows beside the `OrderStatistics Baseline` ([models](models.md)).
- **6, Q0 and Q1.** The null, shuffled, synthetic and irrelevant-feature controls, the lockbox, calibration metrics, and the quantum meta-learner rows with their classical SVM control ([research](research.md)).
- **6, Q2 first build.** The randomness-discrimination experiment and its weekly run ([research](research.md)).
- **7, phase 1.** The per-slot vote for Pick3 ([models](models.md)).
- **8. Scheduler.** The server owns the schedule; the crontab entries are gone ([operations](operations.md)).
- **10, step 1.** Three extra lotto numbers per row, scored against chance ([scoring](scoring.md)).
- **Gates for every tuner**, the deep-learning tuner included ([tuning](tuning.md)).
- **Prize-table scoring** for Lotto, EuroMillions, EuroDreams and Vikinglotto ([scoring](scoring.md)).
- **Worker machines.** A GPU lane and a CPU lane, with the Sunday controls on the CPU worker ([operations](operations.md)).
- **Hyperopts, backtests and parameter overrides from the admin pages**, and variant rows per model ([tuning](tuning.md), [models](models.md)).
