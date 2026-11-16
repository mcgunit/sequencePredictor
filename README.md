# Sequence Predictor

> ## ⚠️ Disclaimer - read this first
>
> **This is a research project. It is not financial, investment or gambling advice, and nobody involved in it is a licensed financial advisor.**
>
> - Nothing in this repository or on its pages - a lottery ticket, a "buy" or "sell" on a crypto or shares card, a profit figure, a ranking of models - is a recommendation to buy, sell, bet or play anything.
> - Lottery draws are designed to be random. No model here has shown a reproducible edge over chance; the project's own controls (docs/research.md) exist to check exactly that, and so far they find none.
> - Crypto and stock prices can fall as well as rise. Past results, paper profits and backtests say nothing reliable about the future. Every euro shown on the pages is paper money, not a trade that was made.
> - **You use this software and anything it shows entirely at your own risk. The authors and contributors accept no responsibility or liability for any loss of money, or any other damage, that results from using it** - whether you play, trade or invest on the strength of what it shows or not.
> - The code comes as is, without warranty of any kind (see the license below).
>
> If gambling or trading is causing you problems, look for help - in Belgium for example through the National Lottery's responsible-gaming information or your doctor.

## Abstract

Sequence Predictor tests whether a wide range of statistical, machine-learning, deep-learning, quantum-inspired and foundation models can find any **reproducible, profitable structure** in sequences that are meant to be unpredictable: the Belgian National Lottery games (Lotto, EuroMillions, EuroDreams, Vikinglotto, Keno, Pick3, Joker+) and, as a contrast with a market that is not designed to be random, daily and weekly moves of five cryptocurrencies and five Euronext shares.

Every model predicts every draw or trading day *before* it happens; the predictions are stored, scored against what really happened - hits, and money under the real payout or prize tables - and ranked on a lower bound that one lucky win cannot carry. Negative controls (shuffled and synthetic fair histories, irrelevant features, a lockbox period, a per-row chance test) check whether any apparent skill is more than chance. The research question is not "can we win the jackpot" but: *can any game be made profitable for a player over time, and if a model ever shows an edge, why?* - and, for the markets, *can these models assist a completely inexperienced user*.

A Node.js server runs the pipeline on a schedule and shows the results on a web app: a slim view for users (each game's top five models, point-in-time history) and an admin workbench with everything else.

## How it works, in one paragraph

Every day the server's scheduler runs the models: at 00:05 UTC for the next day and week of crypto, just after the crypto day closes, and at 06:00 Brussels time for the next draw of every lottery game, after fetching the new draws and their prize tables. Each run scores the previous predictions and writes a performance report and a ranking. At 18:05 Brussels on weekdays it does the same for the stocks, after Euronext's close. On Saturdays a chain of tuners (Optuna, with a champion/challenger gate) and the meta-learners retrain; on Sundays the negative controls run (the null-history band on the first Sunday of the month, the others every week). The heavy work - the deep-learning tuner, the other tuners, the Sunday controls - can run on two worker machines the server reaches over ssh, with the server itself as the fallback. Everything is committed to git, so each ranking can be read back as it stood on any day.

## Technologies

- **Python 3** for the pipeline: NumPy, SciPy, pandas, scikit-learn, TensorFlow/Keras (with keras-tcn), XGBoost, LightGBM, CatBoost, Optuna; Chronos-2 and TimesFM (PyTorch) as zero-shot foundation models; the quantum models on a NumPy state-vector simulator.
- **Node.js / Express** for the server, the scheduler (`jobs.js`), login, the APIs, and the runs started from the admin pages (backtests, hyperopts) on the server or on a worker machine (`workers.js`).
- **React, TypeScript, Vite and ECharts** for the web app (`web/`, served under `/app`).
- **Data**: JSON day files per game and draw, CSV draw histories, SQLite for the markets store and the Optuna studies; draws from the National Lottery's draw API, prices from Euronext, Yahoo and exchange APIs.
- **Local LLMs** (llama-server, OpenAI-compatible API) for the Council.

## Models and methods

Each model produces a *row*: a ticket for the coming draw (or a call per instrument for a market day). A model can store up to two variant rows next to its own - "Memorise" and "Sampled" - which are scored and ranked like any other row. The details are in docs/models.md.

| Family | Rows (as stored) | In short |
|---|---|---|
| Statistical | Markov, MarkovMonteCarlo, MarkovBayesian, MarkovBayesianEnhanched, PoissonMonteCarlo, PoissonMarkov, LaplaceMonteCarlo, HybridStatisticalModel | transition chains, Poisson and Laplace position models, Monte Carlo sampling |
| Baseline | OrderStatistics Baseline | the historical mode per position: the yardstick every model must beat |
| Foundation (zero-shot) | Chronos Model, TimesFM Model | pretrained time-series models forecasting each position |
| Deep learning | LSTM Base, LSTM Smoothed, TCN Base, UnifiedLstmTcn, UnifiedLstmGruTcn, Transformer, GNN, Autoencoder | sequence networks per position, time-boxed; the autoencoder is also an anomaly tripwire |
| Gradient boosting | XGBoost, LightGBM, CatBoost and their MultiLabel versions | per-position multiclass and per-number multi-label boosting |
| Meta-learners | MetaLearner, MetaLearnerV2, QuantumMetaLearner, QuantumVQC, ClassicalSVM | stacking on the base rows' scores; the quantum kernel and variational classifier against a classical SVM control |
| Vote ensembles | WeightedEnsemble, SubsetEnsemble | score-weighted votes over all rows or a tuned subset |
| Reinforcement learning | RL Ticket Model, RL Position Model | ticket construction from the other rows (lotteries); position sizing (markets) |
| Market rows | GARCH, Regime HMM (+ ZeroMean and Single ablations) | volatility and regime references for crypto and shares |

**Methods around the models:** walk-forward backtesting; Optuna tuners behind a champion/challenger gate with a capped lower-bound objective; a lockbox period no tuner sees; a best-combination search with a shuffled control; phase-shift (lag) analysis; an entropy/KL randomness watch; the controls (null bands, per-row chance test, randomness discrimination, irrelevant features, tuning on control histories, market rows under a proper score); a live record that only counts tickets saved before a draw's sales close; scoring on payout tables (Keno, Pick3, Joker+) and on each draw's own prize table (Lotto, EuroMillions, EuroDreams, Vikinglotto); a variety rule in the gates, so that a setting which makes a model repeat one ticket is not adopted; parameter overrides, backtests and hyperopts started from the admin pages, where a run only records its result until an administrator applies it.

**Upcoming or proposed (not built):**

- **LLM predictor** - a row whose ticket comes from large language models: the local LLMs of the Council (Qwen2.5, Granite-4.0, Falcon-H1, EXAONE-4.0, with LFM2.5 combining their answers) given the recent draws and the other rows' votes, scored like every other row. The Council itself (`/app/council`, `src/llmCouncil/`) is built and answers questions today; it was built with this row in mind.
- Q3: direct quantum scoring of each candidate number; Q4: a Grover-style candidate-search cost study (docs/research.md).
- Sentiment signals for the markets (CryptoPanic, LunarCrush, QuantMuse) as meta-learner features, not as predictors.
- Lotto multi-pick: a first six rethought from the ranking (argmax by ranking) if the 7th-9th numbers turn out to carry signal.

## Roadmap and to-do

The open work, in short. Each item is described in [docs/roadmap.md](docs/roadmap.md), with what is built and what remains; what was decided and found along the way is in [docs/changelog.md](docs/changelog.md).

**Lottery**
- [ ] Best-combination search on profit for Lotto, EuroMillions, EuroDreams and Vikinglotto (it ranks them on hits).
- [ ] Tuners and meta-learners on profit for those four games (they tune on hits).
- [ ] Lotto multi-pick: read the 7th-9th numbers against chance once enough draws are scored, then the euro value of a 7- to 9-number system play from the prize tables.
- [ ] Vote per position for the positional games, phase 2.
- [ ] Council rows: tickets from the Council's LLMs, per member and combined, from the other rows' record and from the draws alone (the LLM predictor above).

**Markets**
- [ ] Shares: read the first weekend tunings on the five Euronext shares.
- [ ] Bolero costs above 1,000 EUR a trade: check the upper tiers of the cost model against the tariff sheet.
- [ ] Instructions from a paper trader instead of fixed intervals, with a month prediction beside day and week: shares long term and long only, crypto in both directions.
- [ ] Review the crypto order-book variant.
- [ ] Sentiment feeds (above).

**Users' side**
- [ ] A beginners' page.

**Research**
- [ ] Q2 extensions, Q3 and Q4 of the quantum track.
- [ ] Wire or remove the dead LSTM layer-count setting and the unused `SelfAttentionBlock` in `src/LSTM.py`.
- [ ] The deep-learning tuner's score on validation windows outside the lockbox.
- [ ] Follow the control readings the scheduler writes after every Sunday's controls.

**Platform**
- [ ] A message outside the app (a chat webhook) when a notification is raised, if the app's own notifications prove not enough.
- [ ] Day files to SQLite, when its trigger is reached (docs/roadmap.md item 9).
- [ ] Clean up stale control files and the old `Council_Integration.md`.

## Quick start

Python 3 with `requirements.txt`, Node.js with `npm install` and `npm run build:web`; then `npm start` (port 3001; a development start that restarts on changes - a production install runs `server.js` under a process manager). The scheduler only starts jobs when the server runs with `SCHEDULER=on`; by hand: `python3 Predictor.py -a true -g lotto`. Tests: `npm test`. Everything else - installation details, the foundation libraries, the schedule, the worker machines, login and users - is in docs/operations.md.

## Documentation

The pages under `docs/` describe how the software works now. What was decided, measured and found, and when, is in the changelog. Administrators can read all of it inside the app, under Admin > Wiki.

| Page | What is in it |
|---|---|
| [docs/models.md](docs/models.md) | every model, how predictions are made and combined, variant rows, history rebuilds, the RL rows |
| [docs/scoring.md](docs/scoring.md) | hits, profit, payout and prize tables, Joker+, multi-pick, the performance report and ranking, phase-shift |
| [docs/tuning.md](docs/tuning.md) | the tuners and their gates, the variety rule, backtests, hyperopts and parameter overrides from the admin pages, the meta-learner |
| [docs/research.md](docs/research.md) | the quantum track, randomness discrimination, the negative and null controls, interpretation |
| [docs/markets.md](docs/markets.md) | crypto and shares: data, games, models, paper books, orders |
| [docs/operations.md](docs/operations.md) | installation, running, the server, the web app, login, jobs and the schedule, the worker machines, testing, data |
| [docs/roadmap.md](docs/roadmap.md) | the open items: what is built and what remains |
| [docs/changelog.md](docs/changelog.md) | the decisions and findings, by date |

Code comments that cite a README section ("README *Strategic Optimization*", "README roadmap item 8", ...) refer to these pages.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute this project as long as you give attribution to the original author.
