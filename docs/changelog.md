# Changelog

What was decided, built, fixed, measured and found in this project, by date, newest first. The other pages describe how the software works now; this page records how it came to be that way. Each entry names the page that documents its subject. Figures in an entry are the figures of that day.

## 27 November 2026

### Backtest record: a row that is too slow no longer holds up the others

**Found.** On 25, 26 and 27 November the Backtest record job ran 3.1 to 3.2 hours and ended on the same row each day: Keno's XGBoost. Its replay takes about 70 seconds a draw; the first two months went through in 72 minutes, the other eight were cut off at the two-hour limit after 110 of 245 draws. Nothing was recorded, so the next day started on it again, and the four other Keno boosting rows behind it never ran. The job also ran past its own two hours because each stretch of months got the full time that was left.

**Changed.**

- A row cut off at the time limit after an hour or more of its own is marked `slow` with the settings it ran on. It is not started again until those settings change, and reads "too slow to validate" on Home and the Lottery page instead of "not validated".
- A row cut off with less than an hour, because the job's time was nearly up, is left for the next run without a mark.
- A row that failed before is taken behind the others.
- A row's time limit holds for all its months together.

Open: whether such a row gets a longer run of its own on the CPU worker. Keno's XGBoost row would need about 4.5 hours each time its settings change.

## 24 November 2026

### The phase-shift page as a "what if" (roadmap item 17)

*Documented in [scoring.md](scoring.md), What if the peak lag had been played.*

**Decided.** The Phase-shift page also answers what the profit would have been had the peak lag been taken into account. The result stays on that page.

**Built.** `phasewhatif.js`: for every draw the ticket a row made *lag - 1* draws earlier is played on that draw, with the lag the row's peak in the newest run recorded before the draw. Beside it stand the row as played and the average over all thirty lags, what a lag picked at random pays. Computed from the day files when the table is opened; no new job and no new file.

**First reading, on production's day files of 23 November.** Playing the peak lag does not beat a lag picked at random. Over all rows together the what-if ends behind any lag in six of the seven games (keno -4,058 against -3,917 EUR, Joker+ -1,215 against -1,025, Vikinglotto -900 against -833, Lotto -566 against -542, EuroMillions -979 against -971, EuroDreams -788 against -786). Pick3 is the exception (-742 against -1,288), carried by a few straight hits: its two best rows gained on 2 and 3 draws of about 60. In every game fewer than half of the rows are ahead of any lag (7 of 23 up to 17 of 39).

### Backtest record and held place as chips

The lines beside the best model, the best combination and each of the top 5 on Home and the Lottery page were sentences, and five of them beside five tickets read as a block of text. They are chips now, with the sentences as tooltips:

- **"30% validated"**: profit in 3 of 10 backtested calendar months. Green when that is more months than random tickets played the same way; "not validated" (dashed) without a backtest record. A `*` marks settings that changed since the backtest.
- **A bar with a percentage**: the share of the checked draws on which the row held its place; beside the best model and the best combination also "checked over 92 draws".
- **"39 in a row"**: the draws up to the coming one without a break.

The word "score" is gone: it was the share of the checked draws in that place, and "score 1%" beside a row that entered the top 5 today (1 of 92 draws) read as a verdict on the row. The figures are indicators, not proof.

### Home's best model is never a variant row

**Found on 23 November.** Home named `XGBoost Model · Memorise` as Joker+'s best model; the Sampled and Memorise rows are for administrators only.

**Changed.** The best model is chosen among the rows that are not variant rows (`ranking.bestOf`, `lottery.summaryOf`); the saved rankings are read anew once, so the count of how long a best model has held its place follows the same rule.

## 23 November 2026

### How long the best model, the top 5 and the best combination have held their place (roadmap item 16)

*Documented in [scoring.md](scoring.md), How long a place is held.*

**Decided.** Users see it too, as a score. Beside the best model, the best combination and each of the top 5, Home and the Lottery page show over how many draws in a row the place has been held and a score: the share of all counted draws in that place. Administrators have a card *Who held the top* on *Best model & rankings*, with each game's runs; users do not get it, because it names rows outside the day's five.

**Built.** `standing.js` counts the rankings saved before each draw (`rankhistory.js`, whose saved versions now also keep the best model and the best combination; its cache is read anew once) and today's report for the coming draw. No new job and no new file: the count is made when the page is asked for.

**First reading, on the saved rankings of 23 November.** Places change often. Lotto's best model had held its place for 4 draws in a row and before 7 of the 27 counted draws (score 26%); keno's for 27 in a row (30%); Pick3's best model had changed that morning. The longest holders are top-5 rows: Lotto's WeightedEnsemble and Vikinglotto's MarkovBayesian were in the top 5 before every counted draw.

### Vikinglotto: a grid costs 10.00 EUR, not 2.00

*Documented in [scoring.md](scoring.md), Profit for Lotto, EuroMillions, EuroDreams and Vikinglotto.*

**Found in the first backtest record.** Vikinglotto rows ended 4 to 9 of 9 months in profit and random tickets about half of them. The stored prize tables with a grid at 2.00 EUR pay random tickets back 2.5 times their stake, which no lottery does. The play form settles it: a grid is six numbers played with all five vikings, five times the API's base price, and the chosen Super Viking doubles the prize (the `k+Fav` tiers are twice the `k` tiers).

**Changed.** A Vikinglotto grid is booked at 10.00 EUR (`PrizeTables.GRID_PLAYS`, the same in `draws.js`); the stored prize tables keep the API's base price. Random tickets now get back half their stake. Every Vikinglotto euro figure shown between 5 and 23 November was too high by 8.00 EUR a grid; the figures are computed from the day files, so they are right from the first run after this change. The Vikinglotto backtest record made on 23 November is discarded and built anew (`BacktestRecord.MONEY`).

### Council rows: the combined row reads the tickets already in the day file

*Documented in [operations.md](operations.md), the Council rows job.*

**Found on the morning of 23 November.** EuroMillions and Vikinglotto had no new draw, and their day files already held every member's row from earlier mornings. The job asked no member, so the head had nothing to combine and both games stayed without a combined row; Lotto's combined row was made from three of its five member tickets.

**Changed.** The head combines the member tickets of the day file together with the ones of the running morning. A later run of the same day adds the missing combined row without asking a member again.

## 22 November 2026

### Controls: a margin on the shuffled bound, a higher minimum for the row test

*Documented in [research.md](research.md), The verdict and the per-row test.*

**Found in the reading of 22 November.** Crypto's real figure stood at 0.951 to 0.953 for three weeks while its label went "above", "above", "above band, not shuffled": the shuffled control's bound moved 0.943, 0.929, 0.957 between blocks of seeds. And rows were flagged on 13 to 15 live draws.

**Decided and built.**

- The shuffled control's bound carries a margin of two of its own standard errors (verdict version 3). Read back over the four weeks from 1 November, crypto is "above band, not shuffled" in each. A run of weeks on the pages starts anew at the change of rule.
- The per-row test needs 30 live draws (was the ranking's minimum of 10). Rows followed before the change stay followed until their verdict.

**Not changed.** The controls decide nothing: no tuner, gate, ranking or ticket reads them. They are read by the admin pages and the weekly reading only.

### The backtest record on the lottery pages (roadmap item 13)

*Documented in [tuning.md](tuning.md), The backtest record.*

**Decided.**

- The months are the ten newest complete calendar months outside the lockbox: October and November 2025, then March to October 2026. The lockbox months stay shut.
- A row whose settings changed keeps its old count with "settings changed since" until it is backtested again.
- The random reference is the average over many random tickets, not one random ticket's count, which is luck itself.
- The job runs on this server.

**Built.** `BacktestRecord.py`, `src/BacktestMonths.py`, `backtestRecord.js`, the job `backtestRecord`, and the line on Home (best model, best combination, top 5) and on the Lottery page's ticket table.

**Changed from the plan.** The job was to close the weekend chain. That chain already ends on Sunday morning, hours after the night's crypto run was due, so the job runs every morning behind the Council rows instead: with the months it keeps, a usual morning takes seconds, and a row whose settings changed on a weekday is backtested the next morning.

### The Sunday controls wait for the Council rows

*Documented in [operations.md](operations.md), The Sunday controls on the DL360.*

**Found.** On the first Sunday with both, the first control started on the DL360 in the same second as the Council rows. The control took the processors: the row models answered at 0.27 tokens a second instead of 3.8.

**Fixed.** The DL360's queue starts nothing while a job that works the DL360 itself (the Council rows) runs or waits in the scheduler's queue. The Council rows come first because their rows are read in the morning; the controls follow when they have ended.

**Also found and decided.** On the same morning the three row models on the GPU worker did not load: another program held 7.3 GB of the card's 11 GB, and the models were set to load onto the card whole. Decided: the row models fit themselves to the memory that is free and run the rest on the processor. Measured beside the occupied card: about 10 tokens a second instead of 85 to 95, and a load of 2 to 12 minutes a model.

### Experiment: a Pick3 network with hand-written weights (roadmap item 15)

*Roadmap item 15, closed with this entry.*

**Set-up.** A network of 970 weights (a digit's share of the last 300 and the last 30 draws in, ten chances a position out). Four sets of weights written by an LLM from figures of 2012 to November 2025 only, compared on 153 days (March to July 2026); the best of them then scored once on 112 later days (August to November 2026). Beside it: the plain counts and the same network trained three times. The lockbox period is in none of it.

**Found.** Loss per digit on the later days, lower is better; a blind guess is 2.30259.

| | Loss | Against the counts |
| --- | --- | --- |
| Plain counts | 2.30193 | |
| Written network (chosen on the earlier days) | 2.30429 | +0.0024 (standard error 0.0013) |
| Trained network, three runs | 2.308 to 2.310 | +0.006 to +0.008 (standard error 0.005) |

Nothing beats the counts, and the counts beat a blind guess by less than their own noise. The written network loses less than the trained one because it holds fewer weights that differ from zero, not because it found something. In the figures the weights were written from, a digit that came often lately is drawn 10.07% of the time and one that came seldom 9.95%: no rule to write down. The experiment ends here.

## 21 November 2026

### A backtest record on the lottery pages (roadmap item 13)

Decided: the best model, the best combination and the top 5 get a count of the past periods a row ended in profit, out of ten. The periods are ten different calendar months. A row without a record shows "not tested yet" and is not hidden. Lottery games first. Nothing is built yet; the roadmap lists what it needs, with the chance count beside the row's as the first condition for reading it.

### Council rows: the machines answer side by side

The first morning on the larger models made the GPU worker's rows only: 21 rows, three a game. The job asked the machines in turn, the GPU worker first. Its three members took 11 minutes, and in that time the CPU worker, woken for the job and not yet asked anything, switched itself off as idle. Every question to it was refused, so its two members, the record row and the combined rows are absent that day.

Fixed: `CouncilRows.py` asks the machines side by side, each still one member at a time. The CPU worker gets its first question when the job starts, and the head follows its members without a pause. The members' tickets reach the head in the config's order, whichever machine answered first.

Fixed on the way: with the schedule switched off, a lottery run that was still going queued the weekly plans behind it when it finished on a plan day. `jobs.js` now looks at the switch there too. Two checks of `test/jobs.test.js` read the real calendar and failed on a Saturday; one now tests only what it names.

Open: the morning's total time on the CPU worker with gpt-oss 20B is still not measured.

## 20 November 2026

### Council rows: larger models, one record row

*Pages: models, operations, roadmap*

Decided:

- The Council rows get models of their own, larger than the ones the Council pages ask: Granite 4.2 8B, Nemotron 3.5 Bolt and Llama 3.1 8B on the GPU worker, Gemma 4 26B and gpt-oss 20B on the CPU worker, and Qwen3.6 35B as the head. The Council pages keep their models.
- The row models answer with thinking off and within 400 tokens.
- The record question goes to one member (Gemma 4 26B). The draws question goes to every member.
- A member on the GPU worker is left out while a hyperopt or a backtest holds that machine.

Found, in a trial on a copy of the day files (nothing written):

- With thinking on the CPU worker gave one ticket in two hours: an answer ran to 4,096 tokens at 2 to 4 tokens a second, most without a ticket line. On the GPU worker the same seven games took 4 hours 37 minutes, about half of it loading a model for every question, and Granite gave 10 of 14 tickets.
- With thinking off the CPU worker made its rows for seven games in 48 minutes: Gemma 14 of 14 tickets, LFM2.5 10 of 14, the head 10 of 10. LFM2.5 does not follow the thinking switch and needs a thinking budget of zero.
- LFM2.5 24B, the second CPU member of the trial, is replaced by gpt-oss 20B (21 November): 14 of 14 valid tickets on the first try with thinking off, at four to five minutes a ticket against about one for LFM2.5. As a member it is asked seven tickets a morning, about half an hour.
- With thinking off and one model at a time the GPU worker made 42 of 42 tickets in 9.5 minutes, nearly all of it three model loads.
- On the record question the models gave the same ticket as each other in most games: the question shows the leading rows' tickets and they copy one.
- A model loaded on the GPU worker beside a running hyperopt does not fit on the card and stalls both.

Built: `src/llmCouncil/rows.json` (the row models, `record_member`, `guards`); in `CouncilRows.py` the order of asking (one member over every game, then the head), the single record row, the guard and a self-check in the test chain. The wait for one answer is 10 minutes, to cover a model's load.

Fixed: the CPU worker stayed on for a day. A `backtest.pid` left from 10 November named a process number that this boot gave to a kernel thread, so the idle check saw a backtest running. `worker/idle-shutdown.sh` and the new guard believe a pid file only when it is newer than the machine's boot.

Open: the new member names start new rows, so the earlier Council rows end here and the new ones have no history yet.

## 18 November 2026

### The trader's card on the market pages

*Pages: markets, roadmap*

Built: the card "Trader" on the Crypto and Stocks pages, above the day and week positions (`api.js` `traderView`, `web/src/components/TraderCard.tsx`). One row per instrument with the columns "If you do not have it" and "If you have it"; the day, week and month predictions; the buy price, a live quote and the move since the buy; the long positions' record with holding and random days beside it.

Decided:

- The card is written for one look a morning: every instruction names the booking hour (09:00 crypto, 10:00 shares), and a hold names where the stop stands that morning.
- A position the trader holds is not offered to a reader who did not buy on its day; the cell says "Don't buy" and why.
- On shares the card names no standing order: the stop is read on the close and the trader sells the next morning.
- Long positions only; a short of the paper record shows as no position.
- Decided the same day: the day and week cards ("Open positions") leave the user pages, as nobody followed them; the trader's card is the only set of instructions there. The admin workbench keeps its plan cards.
- Decided the same day: the figures under the card move to the trader's record too. The status strip, the three charts, the latest and the closed positions now count the trader's long positions (sold ones, and open ones at the last close) next to holding every instrument from the trader's first day. The record gained a per-day series for the charts (`series`, schema 2); the first charts come with the first run after the change.

### The paper trader books at 09:00 and keeps its decisions

*Pages: markets, roadmap*

Decided: the trader's rule values stay the hand-chosen ones; a hyperopt and a backtest for them, started from the menu, come later. Every instruction has to be one a reader can carry out when they look at the page once a day, between 9 and 10 in the morning.

Built: one decision per instrument and day, booked at 09:00 Brussels time: the Euronext open for the shares, the hourly Binance bar of that hour for crypto (it was the 00:00 UTC close, before the ticket exists). The crypto stop is a standing order read on the hourly bars. Every market export stores the trader's decision for the coming day in `data/markets/<market>-trader.json`; the trader starts without a position on its first stored day, and a position is live only when its decisions were stored before 09:00 of their days. The totals are given for all positions and for the long ones, which are the ones a reader can follow.

Found: under the 09:00 booking the crypto replay comes to -36.72 USDT over 24 positions (it was -32.49); its long positions alone made -33.72 USDT over 11 positions, against -25.89 for the same positions at random. The shares are unchanged. Of the 24 crypto positions 18 date from before the hourly bars and are booked at 00:00 UTC.

Open: the card on the user pages; the month game; whether a reader of the shares page, ordering later than 09:00, needs a price of their own.

### The month games

*Pages: markets, operations, roadmap*

Decided: the paper trader's rules read three predictions (day, week, month); the shares are booked at the 10:00 hourly price; the stop is moved once a day as the price rises; a take-profit limit is built but off.

Built: `cryptomonth` and `sharesmonth`, one draw per calendar month, cut from the base market's bars like the week games and predicted in the night and evening runs. A month is cut by the first run after its last close, so the shares' month ticket exists before the next month's first open. The trader does not read them yet.

Found: 51 crypto months and 106 share months can be binned. In a trial run on a copy with deep learning and boosting on, the crypto month ticket had 13 rows (the Regime HMM rows and all deep-learning rows but the two LSTM rows had too little history and left none) and the share month ticket 21. Yahoo serves hourly prices of the five Euronext listings for two years back.

Open: the month games in the tuners, the controls and the admin pages.

### The paper trader reads three predictions

*Pages: markets, roadmap*

Decided: a position is bought only when the day call (two days running), the week call and the month call are all up. It is held while the week and month calls are not both down: a day call against it, or one of the two longer calls alone, changes nothing. It is sold when the week and month calls are both down; the earlier rule that all three down also needed two down mornings is covered by this one. Long positions are what a card shows.

Built: the rules above in `src/MarketTrader.py`, with crypto's shorts as their mirror; the stop measured from the best close since the entry, with each morning's decision carrying where it stands; a take limit that is off; the shares booked at the 10:00 hourly price, fetched from Yahoo with every evening export.

Found in a replay on a copy with the month games back-built (crypto 78 game days, shares 42 sessions; back-built calls, so what the rules would have done, not a record):

| | Trader | Long positions only | Hold everything | Its own positions at random | Random runs beaten |
| --- | --- | --- | --- | --- | --- |
| Crypto (100 USDT a position) | -10.82 over 7 positions | -24.91 over 4 | -113.63 | +3.13 | 24% |
| Shares (1,000 EUR a position) | +0.98 over 3 positions | the same | +212.88 | +110.96 | 6% |

The three-prediction rule trades far less than the two-prediction one (7 crypto positions against 24). None of the four long crypto positions closed with a gain; all four ended on the stop. In the shares all three positions are still open. With 3 to 21 positions under any rule values tried, the record still cannot choose between them. The first Yahoo fetch wrote 32,740 hourly bars for the five shares.

Open: the card on the user pages.

### A run stopped half-way left a draw without a ticket

*Pages: operations*

Found: the Joker+ draw of 17 November had no ticket. A power outage on the morning of 17 November ended the Joker+ run after 42 minutes without an exit code (and the Keno run after 7). The live path writes the day file once after the deep-learning step, with an empty new ticket, and again at the end; the run died in between. The run started by hand later that morning found the file and answered "Prediction already made".

Built: a day file that exists without a ticket is made again by the next run (`dayFileWithoutTicket` in `Predictor.py`). The lottery test accepts a production draw without a ticket: it shows no row and no figures.

The existing completeness check (`entryIsComplete`) did not catch it: it runs only when the newest day file is missing, and it looks for the labels, which the half-written file has. The Joker+ draw of 17 November stays without a ticket.

## 17 November 2026

### The paper trader: rules and a first replay

*Pages: markets, roadmap*

Decided: the month game is cut per calendar month (crypto has 76 months of history, the shares 131). The paper trader is built before the month game, on the day and week calls, and reads the row each game's page follows.

Built: `src/MarketTrader.py`, one trader per instrument. It enters when the week call agrees and the day call has persisted, and leaves on a turned week call, on day calls against it, on a stop or after a longest hold; the shares are long only and never buy and sell on one day, crypto may sell short. It books at the market's stake and costs and comes with two comparisons: holding everything, and its own positions laid at random over the same days. A replay prints what it would have done over the stored record. Nothing is on a page.

Found in the replay over the stored record (crypto 77 game days from 1 September, shares 42 sessions from 21 September; most of these days were back-built, so this is what the rule would have done, not a record):

| | Trader | Hold everything | Its own positions at random | Random runs beaten |
| --- | --- | --- | --- | --- |
| Crypto (100 USDT a position) | -32.49 over 24 positions | -120.33 | -10.19 | 20% |
| Shares (1,000 EUR a position) | -20.03 over 4 positions | +212.12 | +53.57 | 16% |

The trader lost less than holding in crypto because it was in the market 39% of the time and short for part of a falling stretch, not because it timed well: its own positions at random places did better four times in five. In the shares it was out of a rising market. Under other rule values the crypto total runs from -94 to +11 USDT and the shares' from -176 to +25 EUR; with 3 to 58 positions each, no value can be chosen from this record.

Open: the rule values; the month game as the third input; the trader's own stored record; crypto's entry price, which is booked five minutes before the ticket exists.

### Council rows

*Pages: models, operations*

Decided: the Council's LLMs give tickets as tracked rows on all seven lottery games, in both kinds (from the other rows' record, from the draws alone), each as one row per member and one combined row. The CPU worker is switched on for it every morning. The rows are scored and ranked like every other row and are no input of another row.

Built: `CouncilRows.py` and `src/CouncilTicket.py`; the job `councilRows`, queued behind the morning's lottery predictors and ahead of the weekly plans, with the worker's marked wake and a notification when no model answers. The ensemble member selection, the RL ticket row and its tuner, and the combination search leave the rows out. A history rebuild puts back the Council rows a day file holds.

Found in the first trials against the models: the head thinks for about 150 tokens before it answers, so an allowance of 160 tokens cut every answer off. Given the members' tickets as lines, the head answered with all of them one after the other, also when told what was wrong; given a tally of what the members chose, it gives one ticket. The small models tend to list more numbers than asked on the long tickets (Lotto's nine, Keno's twenty), which is why a set game's answer is cut to the first numbers asked for.

Evidence: two dry runs over five games (Pick3, EuroMillions, Lotto, Keno, Joker+), nothing written. In the second, with the tally, 22 of 30 rows were valid tickets on Lotto, Keno and Joker+; the rows that failed were mostly one member answering with a single number. A game takes about 10 minutes, so the seven games take a little over an hour after the predictors; the machine answers 4 minutes after it is switched on and its models a further 3 minutes later. Some valid answers are plainly lazy (the numbers 1 to 9 for Lotto).

Open: the retry that names what was wrong with the last answer was built after these runs and has not been tried against the models. The market games have no Council rows.

## 16 November 2026

### A predictor job per lottery game

*Page: operations*

Decided: each of the seven lottery games gets a predictor job of its own, all due at 06:00 Brussels and run one after the other. Crypto and the stocks stay one job each. Each game pushes its own day files. A game's job that runs far longer than usual is stopped by the schedule, recorded as failed with a notification, and the next game starts.

Why: one run made all seven games, so a game that crashed, hung or got no data could cost the others their run; roadmap item 8 had this as its last open point. A base market game and its week game share their bars, so splitting them gains nothing.

Built: `jobs.js` has seven jobs `<game>Predictor` with their own logs, retries and usual durations; the Saturday chain and the Sunday controls start behind the last of them (none running, queued or waiting for a retry). The stop rule: twice the longest of a game's last successful runs and at least two hours, or four hours with fewer than seven runs to compare with. The notifications of a morning without data are raised once, not once per game. The run history keeps 160 runs instead of 40. `Predictor.py` ends with status 1 when a game's prediction raised an error, so that run is recorded as failed; before, it ended with 0.

Built: the admin pages read the lottery morning as one run a day, continuous with the one job's earlier runs: the Today tile and This week show how many games are made, which were not ok and which wait, and a day's drawer lists each game's run. On its first start the new schedule takes over the old job's handled slots, its pending retry and its queue entry, so a deploy after 06:00 does not make the morning a second time.

Found in review, the same day: the kill a minute after the stop never came, because the stop ends the job's wrapper at once and the run is finished before the minute is over, while a process that ignores the stop keeps `process.lock`. A stop during a write could leave git's index lock or a file written halfway, unsaid. And the weekly plans waited for the 23:00 net when the last lottery job went without finishing (removed from the queue, a late retry dropped, a launch that failed).

Built: the stopped job's process group is watched until it is gone, across a restart: the next game waits for it, a process left after a minute is killed, one left after two is named in a notification. Only the processes that were in the group at the stop count, by PID and start time. After the stop a notification names an index lock or uncommitted tracked files. The three ways a lottery job goes without finishing start the weekly plans when no other lottery job is left.

Evidence: seven separate runs start Python and load the libraries seven times, which adds about a quarter of an hour to the morning; it is usually done by about 10:30, before the first sales close.

### A lottery run without data is a failed run, tried again by the schedule

*Page: operations*

Finding: the morning lottery run of 16 November got a web page instead of JSON from the lottery API for every game, three times each. It fell back to the CSV files, predicted nothing new and ended with status 0, so the schedule recorded a success and nothing was tried again that day. The log showed the first characters of the page's markup, which said nothing about what the page was.

Decided: a run in which the fetch failed for every game asked for is a failed run. The schedule starts it again by itself, every hour, at most six times. Partial failures stay as they were: a game that failed is a draw behind and catches up on the next run.

Built: `Predictor.py` ends such a run with status 74 after finishing the report and releasing the lock; `jobs.js` plans the retries (kept in `config/scheduler.json`, shown on Jobs & services), and holds the weekly plans until a run with data, or the last retry, finishes; `src/DataFetcher.py` logs a web page as its title and visible text.

### Notifications for the administrator instead of a mail

*Page: operations*

Decided: what goes wrong unattended is shown in the app: a bell with the unread count, the unread list after signing in, and a Notifications page. The notifications are stored on the server and read per administrator.

Why: the roadmap had "a mail on failure", but the server has no mail server to send through, and the administrator opens the app daily. A chat webhook stays possible later.

Built: `notifications.js` (store `config/notifications.json`, the newest 200) with two routes; raised by the schedule for a job that failed, a run without data and the end of its retries, a queue blocked for six hours, the controls falling back from the DL360, a week's reading not written and a job running far longer than usual (1.5 times the longest of its last successful runs and at least half an hour more); raised by the workers for a hyperopt falling back, the DL360 not coming up, and the DL360 still on long after its last work.

Built: the server now looks whether the DL360 switched itself off after its work. Before, a machine that stayed on was only seen by opening Jobs & services.

## 15 November 2026

### Documentation split from decisions; in-app wiki

*Page: documentation*

Decided: the pages under `docs/` describe how the software works now, in the present tense and without dates; every decision, measurement and finding goes to this changelog with its date; the documentation names no person.

Why: the pages had grown as a running log, so a reader could not tell current behaviour from history, and a cross-check against the code found 15 to 25 outdated statements on every page (schedule times, removed pages, the exchange of the shares, model counts, tuners without their gates).

Built: every page rewritten against the code; the "Out of date" lists that closed each page are gone, their points corrected; `docs/changelog.md` assembled from the dated parts of the old pages.

Built: the wiki, Admin > Wiki, for administrators only: `docs.js` serves `README.md` and `docs/*.md`, the app renders them with its own markdown reader (`web/src/wiki/markdown.ts`), and `test/docs.test.js` checks that every page is present, that every link leads somewhere and that no page names a person.

### Code comments name no person

*Page: documentation*

Decided: the rule of the documentation holds for the code too: comments, help texts and page texts state a decision and its date, not who took it.

Built: the personal attributions in the comments of 78 files reworded; the word stays where it is technical (the process that holds `process.lock`, the reader a Council session belongs to). No behaviour changed.

### Feature control on the DL360 computed nothing

*Page: operations*

Found: the first Sunday feature control on the DL360 ended after 18 seconds with status 0 and no record. The DL360 had no foundation libraries, so it expected a set of models without `Chronos Model` and refused all eleven cached meta-learner tables. The week's reading carried the feature control of 8 November.

Built: the foundation libraries and the Chronos weights installed on the DL360 (the same forecast as the main server to 1e-6); the feature control rerun there the same day on the Sunday's own snapshot, and the reading rebuilt. `IrrelevantFeatureControl.py` ends with status 1 when it wrote no game's record.

### Market instructions leave the fixed intervals; Council rows planned

*Page: roadmap*

Decided: the buy, hold and sell instructions for crypto and shares are no longer to be tied to a day or week interval. A paper trader per instrument holds for as long as its rules say, reading the day, week and month predictions together; a month game is added as a prediction beside day and week, for both markets, and the three games become inputs only: the reader gets the trader's card and no day or week card. Shares are long term and long only, without a buy and a sale on one day; crypto may go short and may, in an extreme case, buy and sell on one day. This is roadmap item 11; the crypto follower's book is dropped in its favour.

Decided: tracked rows from the Council, per member and for the combined answer, in two kinds: from the other rows' record, and from the draws alone, on the lottery games first. This is roadmap item 12 (before: the LLM predictor row).

### The Sunday controls stay on the DL360

*Page: operations*

Found: the controls run 2.5 to 4 times slower on the DL360 than on the main server (15 November: randomness discrimination 4.3 h, feature control 2.2 h, market rows 29 min). Measured with one benchmark on both machines and the same numpy: the DL360's Xeon E5645 has no AVX, so dense linear algebra is about 9 times slower there, and code that uses no vector instructions is about 2 times slower because of the older cores. The frequency governor is already `performance`.

Decided: the controls stay on the DL360. A control that runs long or goes wrong there never holds up the predictor on the main server; the longer run time is accepted.

### Deleting hyperopt and backtest runs

*Page: [operations](operations.md)*

Built: *Delete* on the Hyperopts page removes a run's record, result, served snapshot and log; on the Backtests page the backtest's record. Both are refused while the run is queued or running and both go to the audit log.

Finding: deleting a hyperopt does not remove its trials from the worker's study, and applied values stay in `bestParams_<game>.json`.

### Models page rewritten against the code

*Page: [models](models.md)*

Fixed: the page was rewritten as present-tense documentation and checked against the code. Corrected: the meta-learner rows are served on positional games through a positional artifact (a flat artifact is refused), not skipped on Pick3; no deep-learning row blends with a Markov chain; `LSTM.py` does not compile `full_draw_accuracy`; multi-label boosting is skipped on every positional game, not Pick3 alone; the RL ticket row is skipped on Joker+ and replaced by the RL Position row on markets.

Fixed: the automatic deep-learning deadline includes the memorise rows: 8 x 300 + 7 x 180 + 120 = 3780 s (63 min) for a lottery game with `-a true`, not 42 min.

Built: added to the page: foundation rows, three further meta-learner rows, market rows, RL Position Model, manual-override labels, per-row `yearsOfHistory`, `markovTransitionMode`, statistical flag defaults.

### Scoring page rewritten as documentation; stale claims corrected

*Page: [scoring](scoring.md)*

Finding: a cross-check against the code found 16 stale or imprecise claims, among them: all seven lottery games ranked on the profit bound and only the market games on average hits; the tuners scored Joker+ on slot hits with profit as a 0.001-weighted tie-breaker, not on profit per bet; the top five skipped variant rows and rows missing from the day's table; the combination search took at most 32 established rows without variants; the lag window was the target draw plus 29, a peak needed 10 draws and a run was recorded only when its sample grew; vote and RL Ticket rows carried extras.

Decided: the page was split into documentation of the present behaviour and this changelog.

### Day files grew with the variant rows

*Page: [roadmap](roadmap.md)*

Finding: a lotto day file of 11 Nov 2026 held 80 rows (32 model rows, 17 Memorise and 31 Sampled variants, since 8 Nov) in 101 KB, of which the two raw arrays were about 12%; the default run covers eleven games. The 30 Sep growth figures understate the files, and dropping the raw arrays saves far less than a factor of four.

## 14 November 2026

### The deep-learning tuner's aim: variety by default

*Page: [operations](operations.md)*

Decided: the tuner got two aims (`--aim`). Variety, the default: per retrained draw the share of validation windows whose ticket differs from the most common one, minus a penalty once the excess loss passes the tolerance (`--loss-tolerance`, 0.025 nats; each tolerance's worth above costs one full point). Skill: the earlier score, unchanged.

Why: the skill score (minus the excess loss over the counts) pulls a row toward the counts and so toward one fixed ticket.

Evidence: the reference for how a row should predict was pick3 UnifiedLstmGruTcn: 47 different tickets over 61 draws at a validation loss about 0.02 nats above the counts.

Decided: each aim has its own Optuna studies (`<game>-<model>-variety`, `-skill`). The variety aim is not a measure of hits, which stay at chance; the tuner is not scheduled, so only hand-started runs changed.

### What a rerun continues

*Page: [operations](operations.md)*

Finding: every tuner opens its Optuna study with `load_if_exists`, so a second run of the same game and row (and aim) adds trials to the same study; *Fresh* on the deep-learning form only restarts the run's own trial count. The GPU worker, the DL360 and the server's fallback folder each keep their own study history.

## 13 November 2026

### Deep-learning variety measure extended across draws

*Page: [operations](operations.md)*

Finding: the 13 Nov pick3 LSTM run's winner gave the same ticket on 94 % of its windows on every draw, just under the line, and the same ticket on all 10 draws.

Fixed: a deep-learning candidate is also input-blind when it gives the same ticket on the line's share of its backtested draws or more.

## 12 November 2026

### All six tuners on Hyperopts; variety line a setting

*Page: [operations](operations.md)*

Decided: the page also starts the statistical, boosting, quantum, ensemble and RL tuners, on the DL360 in its backtest queue (the ensemble and RL tuners with six hours). When the DL360 does not come up before the tuner ran, the run moves to the server (exit 75), first in the queue, niced, in `localDir`.

Decided: the variety line became a setting (`config/tuning.json`, `varietyLine`, 95 % when missing) that the weekly chain reads; every tuner writes the same result file (`src/TuningResult.py`).

Evidence: on 12 Nov 2026 no served statistical, boosting, ensemble, quantum or RL lottery row gave its most common ticket on more than 34 % of its last 90 draws (pick3 PoissonMonteCarlo; most under 10 %), so at 95 % the rule only stops a candidate collapsed onto one ticket. Variety is not skill.

Finding: `worker/idle-shutdown.sh` had to be reinstalled on the DL360 to count a running hyperopt as busy.

## 11 November 2026

### Sunday controls moved to the DL360

*Page: [operations](operations.md)*

Decided: with a `cpu` entry in `config/workers.json` the control plan's four jobs run on the DL360, in their own queue beside the server's run, taking no `process.lock`.

Evidence: on 1 Nov 2026 the band held the server's queue from 10:05 to 21:52 UTC, which with the 18:05 shares run would have made the Sunday shares card hours late.

Finding: a first Sunday's band takes about 24 hours there instead of 12, so the weekly controls and the reading come on Monday afternoon, inside the plan's window. The feature control and the market rows read what the Saturday chain produced, so the plan stays after the chain.

Decided: fallback: anything failing before the control started there ends with 75, and that control and those behind it are queued on the server in order, so no week's reading is lost.

### Deep-learning hyperopts from the admin pages

*Page: [operations](operations.md)*

Decided: Admin › Models › Hyperopts starts `HyperoptDeepLearning.py` for one model on one game on the GPU worker, in the same queue as the deep-learning backtests; budget every trial at its time limit plus the gate's two references and an hour, two weeks at most.

Decided: record only: nothing a run finds is served. *Apply* writes the gate's result, or the run's best trial against the gate's verdict, into `bestParams_<game>.json` on the server; refused while a job holds `process.lock`, and it asks first when a value changed since the run started.

Finding: from then on the deep-learning studies lived on the GPU worker; a hyperopt run by hand on the server uses the server's `db.sqlite3` and no longer sees them.

### Variety rule in the tuners' gates

*Page: [operations](operations.md)*

Decided: a usable parameter set has to predict new sequences; recurring numbers are fine. A candidate that gives the same ticket at or above a line is input-blind: never the challenger, never served back, and not a score the challenger must beat. When every trial is input-blind, the served values stay.

Decided: the market rows (GARCH, Regime HMM, RL Position and every strategy on the markets) are not judged, because a market call may well repeat and RL Position sizes positions.

## 10 November 2026

### Worker machines for backtests

*Page: [operations](operations.md)*

Built: `workers.js` ran backtests on two machines beside the server: the deep-learning rows on an RTX 2080 Ti LXC container (always on, shared, TensorFlow capped to `gpuMemoryMb` of the 11 GB card), the statistical, market and boosting rows on an HP DL360 G7 (2 x Xeon E5645, 24 threads, 148 GB, no AVX). The main server stayed the only writer of `data/` and `bestParams_*.json`.

Evidence: the DL360 measured about half the server's speed: plain Python 0.84 s against 0.45 s a task, all cores 14.7 against 27.5 tasks a second, numpy the same ratio; the same 10-draw Markov backtest gave the same score on both to the last digit.

Decided: the weekend tuning chain stayed on the server (about twice as long there). The DL360 draws about 185 W idle, so it is switched on through its iLO when there is work and switches itself off after 10 idle minutes, only on a boot the worker marked.

Finding: a DL360 that does not come up usually has a USB disk plugged in; its start-up then stops at a red screen.

### Backtests on the DL360

*Page: [tuning](tuning.md)*

Built: with a CPU worker set up, the statistical, market and boosting backtests run on the DL360 beside the schedule, 12 hours at most.

Evidence: the same 10-draw Markov backtest of lotto gave 0.5472474768348052 on both machines.

### Deep-learning rows in the Backtests page

*Page: [tuning](tuning.md)*

Built: the seven deep-learning rows are backtested on the lottery games with the deep-learning tuner's evaluation on the GPU worker; `suggest_model_params` was moved out of the tuner's objective for this.

Decided: 3 draws by default (the deep-learning gate's window) and 30 at most, because each draw is a retrain.

Evidence: one LSTM draw on lotto takes about 2 minutes on the worker.

## 9 November 2026

### Lottery run moved to 06:00 Brussels

*Page: [operations](operations.md)*

Finding: the first night run of the lottery games (00:30 UTC on 9 Nov) fetched nothing: the API answered all seven games with a reply that was not JSON (`Expecting value: line 1 column 1`), every game fell back to its CSV and said *Prediction already made*; the day's predictions came from a hand run at noon. The same request answered normally that afternoon; one night does not tell whether the API has a night maintenance window.

Decided: the lottery job (*Morning predictor: lottery games*) runs at 06:00 Brussels local time, usually done by about 08:00, before the first sales close (13:00 on Sundays for Keno, Pick3 and Joker+). The plans' anchor moved to 06:00.

Fixed: `src/DataFetcher.py` tries a failed answer three times (after 30 s and 120 s) and logs each failure's HTTP status and the start of the reply.

## 8 November 2026

### The night schedule: three daily runs

*Page: [operations](operations.md)*

Decided: crypto at 00:05 UTC (five minutes after the crypto day closes, so the card is up minutes after the 00:00 UTC entry instead of about eight hours), the seven lottery games at 00:30 UTC behind it, the stocks at 18:05 Brussels on weekdays (18:30 before), caught up until 08:30.

Evidence: over 30 days the lottery API's `resultsAvailableTime` put every draw's results within about two and a half hours of the draw: 30 minutes for keno and Pick3 up to 2 h 17 min for Vikinglotto, the latest around 22:30 Brussels.

Decided: only the lottery run starts the weekly plans; their anchor moved from 09:00 to 03:00 and the catch-up nets kept their clock times. The scheduler's tests pin `TZ=Europe/Brussels`.

### Parameters page and per-row years of history

*Page: [tuning](tuning.md)*

Built: Admin › Models › Parameters shows every setting of a game's `bestParams_<game>.json`, one card per model.

Built: every deep-learning row reads its own `<prefix>_yearsOfHistory`; the LSTM rows read the bare key, the fallback for the others.

Finding: on 8 Nov every game's per-model values equalled the bare one, so no row's history changed.

### Manual overrides and fresh retrain

*Page: [tuning](tuning.md)*

Built: manual values in `config/paramOverrides.json` win over the tuners from the next predictor run until released; a changed deep-learning setting asks the row for one training from new weights.

Decided: the predictor and `TrainMetaLearner.py` apply the overrides; the tuners and the controls keep reading the tuned file.

Why: the meta-learners are trained on the base rows the daily run serves; a manual value is an experiment on the served rows, not a new tuned value.

### Backtests page

*Page: [tuning](tuning.md)*

Built: Admin › Models › Backtests runs the tuner's own evaluation for one row with the served or tuned values, as the scheduler's `backtest` job (starts with at least 15 minutes before the next scheduled run, ends 10 minutes before it, 3 hours at most).

Evidence: on a scratch copy of lotto with the draws the 7 Nov gate saw, the 90-draw backtest of Markov's served values gave 0.8460825194316038, the gate's incumbent to the last digit.

### Variant rows: settle, memorise, sampled

*Page: [models](models.md)*

Decided: every model is tracked in up to three forms (`src/RowVariants.py`): the settle row, `<row> · Memorise` and `<row> · Sampled`. Not on the market games; sampled rows on number games only.

Why: to see next to each other a model that settles on the counts, one made to chase the noise, and one that draws from its own scores.

Evidence: the check of the stored history found the models serving almost the same ticket day after day (the deep-learning rows repeated 70-95% of their numbers since Feb/Aug 2026) while the early shallow networks varied; hits per ticket were at chance either way.

Decided: `LSTM Smoothed Model` has no memorise row (it differs from LSTM Base only by the smoothing that memorise drops); CatBoost memorise rows are off unless `memoriseCatBoost` is true, because depth is its cost.

### Variant rows kept out of ranking and votes

*Page: [models](models.md)*

Decided: variants are administrator rows: ranked after every other row and marked `variant` in `modelPerformance.json`, never the best model or in the users' top five, and kept out of the vote chart, the vote rows, SubsetEnsemble and its tuner, the RL ticket (the day and its training history), the per-row chance test's family and the combination search.

Decided: they join any of these only through a gated decision (with versus without them), later.

## 7 November 2026

### Deep-learning score became excess loss over the counts

*Page: [tuning](tuning.md)*

Finding: `val_loss` carries label smoothing and the L2 penalty, both tuned knobs. The 6 Nov lotto run "won" all three studies largely by lowering the smoothing (the counts-only baseline sits at ~3.27 at smoothing 0.05 and ~3.19 at 0) while every row still read counts-only.

Decided: the score is minus the mean of the learning check's `excessLoss` (plain cross-entropy minus that of the training frequencies, no smoothing, no L2). Hits are on the record only (0.05 a hit had been ~100 standard errors of the old score), with `val_loss_mean` and `learns_days`.

Built: the new score writes to its own studies (`<game>-<model_type>-skill`), so old trials feed neither the sampler nor the pruner's median.

Decided: `-d` defaults to 3 (8 until then); the warm-started days agreed to the third decimal.

Finding: where frequencies drift, an output that only shrinks the counts toward uniform scores above 0 without reading its input; `learns_days` is the test for that.

## 6 November 2026

### Deep-learning tuner safeguards

*Page: [tuning](tuning.md)*

Built (`src/DeepLearningTuning.py`): a run marker in `data/hyperOptCache/deeplearning-run.json` so a repeated command resumes; each study's decision written and committed when the study ends; every rebuilt day's validation loss printed; a MedianPruner after `--prune-after-days` (2; first 3 trials never pruned); every training run capped at `--train-seconds` (240).

Why: until then the tuner trained up to 1,000 epochs while production stops at 240 s, so parameters were not chosen for the way production trains them.

### Deep-learning tuner put behind the gate

*Page: [tuning](tuning.md)*

Decided: this run's best trial replaces the served parameters only if it beats them and Predictor.py's defaults, trained and scored on the same days with the same budget, by one standard error; the defaults are served back when they score above the incumbent.

Finding: the standard error is taken over warm-started retrains on almost the same history, so one SE is a floor, not a test; the validation split (the last 20% of each retrain's ten years) contains the lockbox period and is not withheld. The references cost one or two extra full trials per study.

### Gate for the deep-learning tuner

*Page: [roadmap](roadmap.md)*

Built: `HyperoptDeepLearning.py` got a gate, with resume, per-study saves, pruning and production's 240 s training budget.

Finding: its validation split still contains the lockbox period; the gate records say so.

## 5 November 2026

### Operations text moved out of the README; stale points found

*Page: [operations](operations.md)*

Built: the operations text was moved out of the README word for word into `docs/operations.md`.

Finding: the check of that day found the library list incomplete and the pip lines pinning TensorFlow 2.18 against 2.16.2 in `requirements.txt`; `Dockerfile` pins differing the same way, `Dockerfile.gpu` unmentioned; `npm run build:web` missing from the Node steps.

Finding: runs are started by the server's scheduler, not cron; `LSTM.py` is `src/LSTM.py` and `GRU.py` does not exist; shares quotes come from Yahoo's chart on the Euronext listing, not Nasdaq, and a share week follows the Euronext sessions.

Finding: the production pm2 process is named `sequencePredictor`, not `predictor`; `COUNCIL_ACCESS` and `COUNCIL_API_HOST` were undocumented; `npm test` (then 15 Node tests and the Python self-checks) is the test suite and `sequenceToPredict_*.json` does not exist.

Finding: `Council_Integration.md` in the repository root describes wiring the old server-rendered Council page, which `/app/council` replaced.

### Tuning text moved out of the README

*Page: [tuning](tuning.md)*

Built: the tuning text was moved out of the README into this page word for word on 5 Nov 2026, with a list of the points a check against the code found out of date.

Finding: cron and `runHyperopt.sh` were replaced by the server's scheduler (`jobs.js`); the Saturday chain runs HyperoptStatistics, HyperoptBoost, HyperoptRLTicket, HyperoptEnsemble, HyperoptQuantum, TrainMetaLearner, keno subset tuning.

Finding: the text still said the ensemble and deep-learning tuners write without a gate (gated 15 Oct and 6 Nov).

Finding: `HyperoptBoost.py` tunes six strategies, not one; GARCH and Regime HMM cover all four market games.

Finding: "the 8 base models" of the meta-learner was outdated (Chronos, the market rows); `meta_learner.joblyb` was a typo for `.joblib`; the trainer runs after HyperoptQuantum in the chain.

### Models text moved out of the README

*Page: [models](models.md)*

Built: the models description was moved word for word from the README into docs/models.md.

Finding: a comparison with the code on that day found five groups of statements no longer true; they were listed under "Out of date" instead of being rewritten, so that nothing changed meaning unseen (see the following entries).

### Out-of-date check: main idea named only lotteries

*Page: [models](models.md)*

Finding: the page's main idea named only the lottery games. Missing were the market games (crypto, shares and their week games) and the fact that money was no longer scored on Keno and Pick3 alone: Joker+ had a payout table, and Lotto, EuroMillions and EuroDreams were scored on each draw's prize table (`src/PrizeTables.py`).

### Out-of-date check: TCN and unified rows were live

*Page: [models](models.md)*

Finding: `TCN Base Model`, `UnifiedLstmTcn Model` and `UnifiedLstmGruTcn Model` were live rows; the sentences saying TCN was never exercised and every game was hard-coded to `lstm_model` were outdated.

Finding: still true on that day: `SelfAttentionBlock` in `LSTM.py` was unused and `num_lstm_layers` had no effect on the built network.

### Out-of-date check: boosting constant name

*Page: [models](models.md)*

Finding: the page called the shared key list `BOOSTING_PARAM_REFIXES`; in `src/BoostingBase.py` it was `BOOSTING_PARAM_SUFFIXES`. Six boosting rows existed, all with `score_numbers`.

### Out-of-date check: combination section incomplete

*Page: [models](models.md)*

Finding: the row list in "How predictions are combined" was incomplete; keno served all six ticket sizes (`use_5` to `use_10`), not `use_10` alone; the meta-learner's base models also included Chronos (TimesFM opt-in) and, on the markets, the four market rows.

Finding: the sentence "Also skipped for Pick3" on the MetaLearner row had been restored from git on 5 Nov, dated from August 2026 and had not been re-checked.

### Out-of-date check: runs started by the server scheduler

*Page: [models](models.md)*

Finding: `runPredictor.sh` and cron no longer started the prediction runs; the scheduler inside the server (`jobs.js`) did.

### Vikinglotto joined the prize-table games

*Page: [scoring](scoring.md)*

Decided: Vikinglotto was scored in euro at 2.00 EUR a grid (six numbers and the viking number) once its tier names were confirmed; one correct number already paid the price back.

Built: its tables filled in on the first daily run after the deploy; until then the game ranked on hits. The Home summary showed the best row by money for the four games.

### History pages show profit for the prize-table games

*Page: [scoring](scoring.md)*

Built: `draws.js` (`PRIZE_GAMES`) mirrored the Python scoring ticket for ticket: each ticket's profit of one grid, a draw's and a month's profit over the grids the table could score; a ticket that is not a full grid, or a draw without a table, showed no profit.

### Scoring page moved out of the README; stale points listed

*Page: [scoring](scoring.md)*

Built: the scoring text was moved out of the README word for word.

Finding: the check of that day listed three stale points instead of rewriting them: "both payout games" (three fixed-table games and four prize-table games existed); "Keno/Pick3 rank on profit, the others on hits" (Joker+, Lotto, EuroMillions and EuroDreams ranked on the profit bound too); multi-pick step 4 (the prize per rank per draw was stored; only the euro value of a 7- to 9-number system play was missing).

### Research page moved out of the README; stale points listed

*Page: [research](research.md)*

Finding: the 5 Nov check of the moved text found: Chronos is a base model too, so "eight base models / seven scores" were outdated; pick3 was tuned on the slot-hit bound, not the payout; Q2 and the negative controls read as proposals but were built and ran every Sunday; the lockbox example dates were illustrative (the real file holds 2025-12-01 to 2026-02-28); `current_era` also cut lotto; keno served all six sizes; Q0 was done except the dead LSTM layer-count setting, Q3 and Q4 and the suggested project structure were not built.

### Out-of-date check of the markets page

*Page: [markets](markets.md)*

Finding: the page moved out of the README word for word still described Stooq/Nasdaq prices and the NVDA/AAPL/MSFT/ASML universe (current: Euronext download plus Yahoo back-build; Shell, ING, AB InBev, KBC, UCB), omitted the tables `meta`, `intraday_bars` and `ticket_published`, and described the pre-app pages (History page, New predictions) instead of the app's Crypto, Shares and admin workbench.

### Scheduler after cutover

*Page: [roadmap](roadmap.md)*

Finding (check of 5 Nov 2026): item 8 was done: 13 jobs (2 daily, 7 in the Saturday chain, 4 Sunday controls) and production ran with `SCHEDULER=on`.

Finding (check of 15 Nov 2026): 15 job entries: three daily predictors since 8 Nov (crypto 00:05 UTC, lottery 06:00 Brussels since 9 Nov, stocks 18:05 on weekdays), seven chain steps, four Sunday controls and an on-demand backtest job; the Sunday controls can run on the CPU worker with the server as fallback. The per-game predictor job was half done (three group jobs); no mail on failure.

### Vikinglotto in euro

*Page: [roadmap](roadmap.md)*

Built: Vikinglotto is scored on each draw's prize table like Lotto, EuroMillions and EuroDreams. The best-combination search, the tuners and the meta-learners still rank those four games on hits.

### Roadmap moved out of the README and checked

*Page: [roadmap](roadmap.md)*

Built: the roadmap items moved word for word from the README to `docs/roadmap.md`, with an "Out of date" list of what the check of that day found.

Finding (check of 15 Nov 2026): of about 60 checkable claims 16 were stale or superseded, most in items 1, 2, 7 and 8; item 4 (markets) had no section on the page; the research notice half of the "beginners' page and notice" item was built (a sticky footer on every page, 5 Nov 2026).

Decided (15 Nov 2026): the page lists open items only; decisions, measurements and findings move to the changelog.

## 4 November 2026

### Vikinglotto winner counts checked against the tier reading

*Page: [scoring](scoring.md)*

Evidence: on four draws the Belgian winner counts fitted the odds of the tier reading ('k' = k numbers without the viking number, 'k+Fav' = with it, '6/' = all six whatever the viking number).

## 3 November 2026

### RL ticket special column taken from the day's vote

*Page: [models](models.md)*

Decided: the RL ticket's special column (EuroMillions' two stars, EuroDreams' dream number, Vikinglotto's viking) is the value(s) the day's other rows put there most often (`RLTicketModel.voted_specials`, ties to the smaller value).

Why: the row's ticket had to be a grid that can be played and scored on the prize tables.

### Lottery profit read from each draw's prize table

*Page: [models](models.md)*

Finding: from 3 Nov 2026 Lotto, EuroMillions and EuroDreams were scored on each draw's prize table (`src/PrizeTables.py`).

### Lotto, EuroMillions and EuroDreams scored in euro

*Page: [scoring](scoring.md)*

Decided: the three games were scored in euro from each draw's own prize table (`src/PrizeTables.py`, `<game>-prizes.json`), since their prizes are pari-mutuel and no fixed table exists; the draw API carried the tiers back to at least 2015.

Decided: a row's profit was one grid of its own ticket at the draw's price; a tier nobody won was valued at the announced jackpot (top tier) or the tier's median over the draws where it paid; EuroDreams' annuities counted at their rule value; the raffles (Lucky Lotto, My Bonus) were not scored.

Built: the report ranked these games on the capped profit bound, so one jackpot could not carry a row; a draw without a table had no bet.

Decided: the combination search with its control, the tuners and the meta-learners stayed on hits; the combination search would have needed every ticket scored against every other draw's table.

### Old seven-value Lotto tickets trimmed to six

*Page: [scoring](scoring.md)*

Decided: the statistical Lotto rows of 8 Nov 2025 to 13 May 2026 had stored a 7th value, a guess at the bonus column that no grid plays; the one-time `TrimLottoBonusGuess.py` cut it from the day files, with a backup. Until it ran, those rows played their first six.

Finding: RL Ticket rows before 3 Nov 2026 lacked their full special column and so had no bet on those draws, hits only.

## 2 November 2026

### Order card switched on for users

*Page: [operations](operations.md)*

Decided (31 Oct 2026): `markets.ORDERS_SHOWN_FROM` was set to 2026-11-02; the user pages showed the order card from the crypto day card of Monday 2 Nov and the week starting that day, the first full week with recorded publication times.

Decided: no shadow period, because the pages had no users yet; the gate is the card's day, not the clock, and Admin showed the order book throughout.

### Shares game moved to five Euronext shares

*Page: [models](models.md)*

Finding: the shares game covered five Euronext shares from 2 Nov 2026.

### Probability order for the other number games

*Page: [scoring](scoring.md)*

Decided: EuroMillions, EuroDreams, Vikinglotto and keno rows stored their mains most probable first as well, without extras; stars, dream and viking numbers stayed in place, every keno ticket and subset was ordered, the positional games kept their slots.

Why: the order shows whether the lower-rated places hit about as often as the top ones, which is what a fair game predicts.

Built: a deep-learning row ranked by its main positions only, since a star's softmax shares labels with the mains. Day files before 2 Nov 2026 kept small-to-large tickets. The users' Home showed each game's top five tickets in the stored order.

## 1 November 2026

### Q0 band refreshed monthly

*Page: [research](research.md)*

Decided (30 Oct 2026): `NullControls.py` runs on the first Sunday of the month only; between two first Sundays the Controls drawer and the weekly reading show the latest band with its date.

Why: the band is context only since the per-row test.

### Euronext switch tool tested

*Page: [markets](markets.md)*

Built: `SwitchSharesEuronext.py`: dry run by default; `--apply --backup` refuses while `process.lock` names a live process, on code that is not the Euronext universe, or when the switch was already made; it archives files and both databases, removes the Nasdaq record, keeps `data/readings/`, installs the five shares and fetches and back-builds their bars.

Evidence: on a `cp -a` copy of production: 100 paths (283 files), 19 studies, crypto untouched, 2,773 bars a share from 2016-01-04, archive byte-identical.

## 31 October 2026

### Order rows written from the reader's side

*Page: [operations](operations.md)*

Decided: the open-positions table stayed as it was, with every cell an instruction: Buy says what to do without the coin, Sell what to do with it; the full "If you ..." branches moved to the row's hover text and "How to read this", and the Admin workbench kept every detail.

### Newcomer entries on a roll day

*Page: [operations](operations.md)*

Decided: on a day the day book rolls a position, the card shows two branches ("If you bought ... yesterday" and "If you did not buy ... yesterday"), and each such card books a newcomer entry in a separate ledger (`MarketOrders.newcomer_entry`), never added to "Following this card so far".

Why: a reader who had not bought the day before used to read "no new order today".

Decided: a newcomer position rolls exactly like the main position and shares its limit and fill scan from its roll card on.

Evidence: on the 13-30 Oct back-test (scratch copy, times estimated) the edge book rolled once and booked one newcomer entry (25 Oct BTC, -0.07 USDT against -0.57 for the position next to it); mid booked 4 (+1.22), fixed 10 (+1.88). A mechanics check, not evidence.

### Shares run moved to 18:30 on weekdays

*Page: [operations](operations.md)*

Decided: the shares got their own run at 18:30 Monday to Friday, separate from the 09:00 run of every other game; it never starts the weekly plans.

Why: Euronext opens at 09:00, before the morning run's cards could be up, and its day is complete after 18:00, so the card is up the evening before.

### First weekly chain under the per-tuner margins

*Page: [tuning](tuning.md)*

Finding: the first weekly chain under the 27 Oct margin rules ran on Saturday 31 Oct 2026.

### Shares moved to Euronext

*Page: [markets](markets.md)*

Decided: the shares game moves from NVDA, AAPL, MSFT and ASML-USD to five Euronext shares a Belgian investor can buy: Shell and ING (Amsterdam), AB InBev, KBC and UCB (Brussels); the Nasdaq record is removed, not kept.

Built: source `euronext` (about two years per answer, kept after 18:00 Brussels), a one-time Yahoo back-build since 2016 (`--back-build`), the Yahoo live quote (Euronext's live endpoints answer encrypted), the Euronext calendar checked against every session of the five shares 2016-2026, five slots, and an own evening run so the card is up the evening before its session.

Decided: money is 1,000 EUR a position in whole shares at Bolero's costs plus the 0.35% stock exchange tax, about 0.85% a leg; crypto keeps 100 USDT and 0.1%. The return-unit P&L (`FEE`) is unchanged because it scores calls, not money.

Finding: a daily round trip needs about 1.7% to break even; in the self-check's synthetic market with 1.5% daily moves even a perfect model loses on its daily book. The fee tiers above 1,000 EUR were as remembered, not re-read from the tariff sheet.

## 30 October 2026

### The null band became monthly

*Page: [operations](operations.md)*

Decided: `NullControls.py` runs only on the first Sunday of the month (Sunday day 1 to 7), from Sunday 1 Nov 2026 on; the other three controls stayed weekly. The job became *Monthly controls: null histories (Q0 band, context)* with `monthly: { since: '2026-11-01' }`.

Why: the band took about 14 hours and was context only since Q0 became the daily report's per-row test.

Built: the band became the plan's first job, so the last job, whose finish writes the week's reading, is `MarketRows.py`; the reading is written every week and carries the latest band with its own date when there is no fresh one.

Finding: on a first Sunday the three weekly controls (about 3 hours together) start after the band, so they and the reading land in the early hours of Monday; a catch-up started late on Monday could leave the reading past Tuesday 09:00.

### Crypto order book decided, intraday store added

*Page: [operations](operations.md)*

Decided: the crypto day and week cards tell a reader exactly what a paper book does: buy at market when the card appears, a sell limit at X, sell at market at a deadline or on the week's daily-call exit. `src/MarketOrders.py` booked that rule next to the existing books, none of which changed, exported as `trading.orders`.

Why: daily bars cannot support it; neither the price when a card appears nor whether the day's high came after it is in a daily bar.

Built: the store gained `intraday_bars` (Binance 1h bars from 12 Oct 2026, 1m bars in a window around every card publication) and `ticket_published` (first seen only; not stamped when the export comes more than three hours after `madeAt`).

Built: the history since 13 Oct came from a one-time `python3 -m src.MarketOrders --backfill 2026-10-13`, since an export reaches back three days only.

### Order book rules D1-D7

*Page: [operations](operations.md)*

Decided: entry at the open of the first 1m bar at or after publication (before 30 Oct `madeAt` + 15 min, flagged "estimated"). No order when X is less than 0.30 % above the base close, at or below the price, or less than 0.30 % above it. A limit fills at X only on a high of X × 1.0005.

Decided: D1 the deadline is the publication of the next card of the same game, 12:00 UTC the next day (Monday for the week) at the latest, not 00:00 UTC, which is no reader's moment in Brussels. D5 a day position called up again with a new limit at least 0.30 % above the price is rolled without fees.

Decided: D4 the week buys only when the week call and Monday's daily call are both up; D2 it sells at the first later day card that turns flat or down; D3 it never buys back that week. D7 late entries are a separate ledger, shown Tuesday to Sunday while both calls are up, the price at least 1 % below X and the deadline at least 24 h away.

Decided: money is `cash_pnl` with 0.1 % on both legs and a sized row's size. A missing bar makes an order pending, never guessed.

### Limit variants pre-registered

*Page: [operations](operations.md)*

Decided: D6 three variants of X were booked for Admin: `edge` (the predicted bin's lower edge), `mid` (the bin's representative return) and `fixed` (entry plus 1 % a day, 4 % a week, a control). `edge` was pre-registered in `ORDER_VARIANT_LOG` as the primary before the back-test was read, because a right call then fills.

Decided: a switch of variant is a logged decision at one review (60 live day orders or 31 Jan 2027, whichever is later; a paired per-order comparison whose lower bound is above 0), never a "best so far".

### Order book back-test: a mechanics check

*Page: [operations](operations.md)*

Evidence: on a scratch copy (16 day cards of 14-29 Oct, three week cards) the edge book placed 8 day orders and skipped 8, half of the skips because the price was already at the limit when the card appeared; 15 of the 16 cards were BTC bin 6 from GARCH.

Evidence: +3.75 USDT for the edge book against -6.41 for the old daily book on the same days and -58.46 for buying every coin at the card. The week book bought one coin in two closed weeks (ETH, sold the next morning on a down daily call); most week-up coins never got Monday's daily trigger.

Finding: this checked the mechanics and nothing more.

### Default still served back when challenger has no SE

*Page: [tuning](tuning.md)*

Decided: the default is served back when it beats the incumbent by the revert margin even when the challenger has no standard error.

Why: that comparison does not involve the challenger.

Finding: from 27 to 30 Oct such a row reverted nothing.

### Boosting tuner got its own CheckpointPruner

*Page: [tuning](tuning.md)*

Built: a trial is judged only at half and at three quarters of the scored days, on its partial score at that day, and only against the completed trials of this run.

Why: Optuna's pruner judged the best partial so far, which a lucky first stretch kept up after it faded; and earlier weeks' trials, fitted on another window, set the bar in a study reused every weekend.

### Row-test replay applies the since.json freeze

*Page: [research](research.md)*

Decided: the first replay flags only on or after the `since.json` freeze; an entry kept without `testFrom` is reset once by the next run.

Evidence: the replay without the freeze followed 8 rows and decided 6 *not confirmed* (keno 3, pick3 2, EuroDreams 1); production's first replay (28 Oct) ran that way. On production's 29 Oct record the 28 Oct verdicts on rows flagged before 27 Sep were withdrawn; keno MarkovBayesian followed from 27 Sep on 215 draws, pick3 LightGBM from 11 Oct, EuroDreams LaplaceMonteCarlo from 25 Oct; pick3 UnifiedLstmGruTcn, flagged on 27 Sep itself, stayed *not confirmed*.

### Dirichlet prior on the HMM transition matrix

*Page: [markets](markets.md)*

Fixed: the M-step takes the posterior mode under one pseudo-transition per cell (`TRANSITION_PSEUDOCOUNT`): (expected transitions + 1) / (row total + K), so no exit that a week game's roughly 100 rows never showed is fixed at 0. On a daily game's roughly 1,000 rows it moves a probability by about 1/rows.

## 27 October 2026

### Sunday plan: Q2 with weekly seeds, null job renamed

*Page: [operations](operations.md)*

Decided: randomness discrimination ran `-n 20 --seed weekly` for every game, and the `NullControls.py` job was named *Weekly controls: null histories (Q0 band, context)*, its band no longer a label.

Evidence: the Q2 run went from about a quarter of an hour to about 80 minutes, so the first Sunday read as over its usual time once.

### Margins in standard errors for every tuner

*Page: [tuning](tuning.md)*

Finding: a gain of 0 let noise through every week. On 24 Oct the statistical tuner replaced 30 of its 73 rows, every one by less than one standard error (median 0.23 SE, largest 0.86; four with the incumbent's very hit mean).

Decided: statistical (all strategies, GARCH, Regime HMM, `KenoSubsetTuning`), RL ticket and RL position: k = 1, revert by any gain. Ensemble: k = 1 as before, plain vote back by any gain. Quantum: k = 2.5, revert keeps the margin. Boosting unchanged (k = 1, revert at the margin). Records carry `margin_se` and `revert_margin`.

Why: the untuned default is the unselected reference; going back to it costs nothing in expectation on a row without skill.

Evidence: replay of production's 312 gate decisions of 17-18 and 24-25 Oct: statistical replacements 27 and 30 became 3 and 0 (reverts 6 and 0 became 7 and 0); RL ticket 6 and 3 became 0 and 0 (reverts 0 and 1 became 5 and 2); quantum 15 and 5 became 0 and 0 (reverts 5 became 2 on 25 Oct); boosting unchanged.

Finding: one SE is a bar on a best-of-15 selection, not a calibrated test. Over 26 Sep - 24 Oct it would have let 4 of about 242 statistical row-weeks through; GARCH and Regime HMM won by 0.06-0.47 SE; the RL default drifted 0.36 SE in a week, so part of the reverts is noise.

### Quantum gate margin raised to 2.5 standard errors

*Page: [tuning](tuning.md)*

Decided: `--gate-se-margin` 2.5 for the quantum tuner (1 until then), without a refit confirmation (zero cost).

Evidence: a single fit's score moves about 0.9 SE with its seed alone; the defaults drifted 1.28 SE in a week; the fifteen replacements of 18 Oct lost a median 1.41 SE on 25 Oct. A fair-game simulation replaces 2-5% of rows a week at 2.5, against 34-46% at 1.

### Ensemble plain vote served back by any gain

*Page: [tuning](tuning.md)*

Decided: `--gate-revert-margin` 0 for the ensemble tuner.

Why: the plain vote is a fixed reference, not picked from trials.

Evidence: pick3's served selection [LaplaceMonteCarlo, Markov] trailed the plain vote two weeks running (−2.1734 against −1.6841, −2.8680 against −2.6982) and stayed served on the challenger's 1.30 and 1.12 margin.

### Every gate fails closed without a standard error

*Page: [tuning](tuning.md)*

Fixed: with k > 0 a challenger without a standard error (None or 0) is never written ("no standard error - kept").

Finding: until then the margin fell back silently to `--gate-margin`, 0.

### RL Position sit-out measured in standard errors

*Page: [tuning](tuning.md)*

Decided: the served sizing is used only when its score (mean − 1 SE) is above max(`--gate-margin`, one standard error of its own), i.e. mean money per position more than two standard errors above 0; a sizing without a standard error sits out; no exposure floor.

Built: the record adds `sit_out_floor`, `sit_out_se_margin`, `served_mean_size`, `served_mean_nonzero_size`.

Evidence: on 17 and 24 Oct none of the eight market-weeks cleared it, where production served a sizing in 2 and 3 of the 4 markets (crypto on 24 Oct: +0.0084, SE 0.0128, mean size 0.0225, replacing an incumbent at +3e-19 that bet on one day of 40).

Finding: t > 2 on a best-of-15 sizing is a minimum bar, not a calibrated test.

### Tuning gates drawer shows margins and sit-out

*Page: [tuning](tuning.md)*

Built: the admin Tuning gates drawer shows each row's margin as "margin X = k SE", groups the chip by k, gives a margin-free revert its own line and the RL Position row a *sits out* chip with its reason (`gates.js` `describe()`; `marginSeCounts` on the Tuning page and the This week tiles).

Finding: older records lack the new keys and read null.

### Lottery games ranked on the live record

*Page: [scoring](scoring.md)*

Decided: a row's draw counted in its live record only when its ticket was saved before that draw's sales close (`src/Liveness.py`, `src/drawClose.json`); the lottery games, the users' top five and the point-in-time History ranking (`rankhistory.js`, `CACHE_VERSION` 2) moved to that record.

Why: history rebuilds and late runs had written tickets after the sales close and some tickets had been rewritten after the draw, so the all-draws record counted tickets nobody could have played.

Evidence: the closes are the draw API's `drawTime`, wagers stopping one second before it; checked over a year for Lotto, EuroDreams, keno and Pick3 and over six weeks for EuroMillions, Vikinglotto and Joker+.

Finding: ranking on the live record changed the best row of Vikinglotto (MarkovBayesian to MetaLearner, on exactly the 10 live draws the guard asks), keno (QuantumVQC to HybridStatisticalModel) and Joker+ (MetaLearner to LightGBM).

Decided: a game whose liveness could not be judged read "unchecked" and was never ranked on its live record, so a one-day git failure did not become the users' history. The point-in-time cut-off became 15:00 Brussels time or the sales close when earlier.

### Readers left on every stored ticket

*Page: [scoring](scoring.md)*

Finding: the combination search, the phase-shift analysis, `HyperoptEnsemble.py`, `HyperoptRLTicket.py` / `src/RLTicketModel.py` and `src/RLPositionModel.py` were not moved to the live record and still read every stored ticket; the market games kept their day-level check.

### Since record counts live tickets; no band on rankings

*Page: [scoring](scoring.md)*

Decided: the forward record counted the live record only, falling back to the stored rows for a game whose liveness could not be judged.

Why: not every stored day had been predicted before its draw.

Decided: from the same day no chance band labelled any ranking column; the per-row chance test became its own card.

### Per-row chance test replaces the band as a label

*Page: [research](research.md)*

Decided: every row is tested on its own live draws (`src/RowChance.py`): exact fair-draw null for lottery, block permutation for markets, Holm over the family, a row followed from its first flag and judged once on new draws (confirmed at p < 0.01). The band labels and greys no row and is context in the Controls drawer.

Why: the band says how the best of eight untuned rows looks on 120 fair draws and nothing about one row on its own record; with the rows' own tickets and fair draws it showed at least one row above in 98% of lotto weeks.

Evidence: replayed on a copy of production on 27 Oct, no row held and none was confirmed in any game; 5 rows read under 5% across eleven games against about 10 by chance. About 2 s for the first run, under 1 s a day after. Readings written before 1 Nov 2026 read *band rule*.

### Q2 verdict version 2: power rule and seed-count bounds

*Page: [research](research.md)*

Decided: one power rule in `controlRules.json` (`few_windows` 250 held-out windows, else *too few draws*); bounds from the seed count (null mean + k sd, k = t(n-1) x sqrt(1+1/n): 5.23 at 3 seeds, 2.43 at 10, 2.19 at 20), the shuffled bound the same way; fewer than 2 scored seeds reads *failed*; twenty seeds a week for every game (about 80 minutes). First weekly run 1 Nov 2026.

Why: on unchanged histories 6 of 11 labels changed between 18 and 25 Oct. Lotto's 3-seed thresholds came out 0.570, 0.572 and 0.751 against 0.686 ± 0.021 on 20 seeds, and *too few draws* came from the run itself (a band reaching 0.8).

Evidence: under the windows rule EuroDreams and Viking Lotto (150 windows), the shares week (208) and the crypto week (108) read *too few draws*. On 4,000 resampled weeks of 20 seeds: no false *above* on the seven lottery games, 0.06 label changes a week (all shares), crypto *above* in 99.7% of weeks. On the 26 Oct histories lotto read 0.573 against 0.686 ± 0.021 (*within*), crypto 0.938 against 0.622 with a shuffled bound of 0.899 (*above*). A paired test by seed was measured and left out as anti-conservative.

### Sit-out floor raised to one standard error

*Page: [markets](markets.md)*

Decided: the served sizing must beat 0 by more than one standard error of its own score (`position_sit_out`), instead of any bound above 0.

### Null band became context; later monthly

*Page: [roadmap](roadmap.md)*

Decided (27 Oct 2026): the null band of the ranking became context only, when each row got its own chance test.

Decided (1 Nov 2026): `NullControls.py` runs on the first Sunday of the month only; the other three Sunday controls stay weekly.

Finding (check of 5 Nov 2026): the "next reading" dates in the item were past; the readings ran weekly from 5 Oct 2026. By 15 Nov 2026 the reading is written by the scheduler when the Sunday plan's last job finishes (`readings.js`).

## 26 October 2026

### Cleanup of the old page code (batch 6b)

*Page: [operations](operations.md)*

Built: the old page handlers and everything only they used were removed with their tests: the server's pages and their scoring imports (`draws.js` kept the one copy of the scoring), the market workbench page and its chart client, the Council page, the account, users, activity, jobs and What's new pages with their form posts, and `help.js`.

Evidence: `server.js` went from 1761 lines to about 110 and only wires modules; `test/server.test.js` checks the routes, their order, that none of the removed ones is back and that loading the file starts nothing.

### Keno sub-selection tuning joined the weekly chain

*Page: [operations](operations.md)*

Built: `HyperoptStatistics.py -g keno -s KenoSubsetTuning` became the chain's seventh and last step, because it scores the artifacts the retrain has just written.

Built: with two steps running the same script, the job holding `process.lock` was named by script and arguments (`jobs.jobOfCmdline`, also used by the run banner), and the test matched the shell mirror on whole lines.

Built: the job carried `since: '2026-10-26'`, so *This week* expected it only from that weekend on; the admin pages counted the `KenoSubsetTuning` gate record as the keno step's decision by its row.

Finding: the job list reaches production only with a server restart, which had to come before the chain of Saturday 31 Oct was queued; otherwise that weekend would have run no keno sub-selection tuning.

### Keno subset tuning moved to the chain's last step

*Page: [tuning](tuning.md)*

Decided: a plain statistical run leaves out `KenoSubsetTuning`; it runs as the chain's own last step after `TrainMetaLearner.py`.

Why: inside the first job its day table read the keno meta-learner artifacts and the weighted vote a week old, and the retrain then replaced the artifacts it had just tuned on.

Evidence: the statistical job became 11-13 minutes shorter; the new step takes about a quarter of an hour.

Built: `HyperoptEnsemble.py` writes the keno slicing it scored with as the row's own `subsetEnsembleSubsetMode` / `subsetEnsembleSubsetTemperature` and records it as `kenoSlicing`; the next run reads past that pin, so the row follows the weighted vote a weekend behind; keys set by hand are kept.

### Boosting prune percentile corrected to 75

*Page: [tuning](tuning.md)*

Fixed: `--prune-percentile` defaults to 75 (25 until then), so only the bottom quartile is stopped.

Evidence: costs at most +0.46 to +0.65 h on the ~10.4 h boosting run, for up to 479 completed trials.

### Lockbox days withheld in every row tuner

*Page: [tuning](tuning.md)*

Decided: no score that decides what is served counts a lockbox day: trials, references, the standard error behind a margin, `modelScores` and the keno day table. Fitting still uses those days. Every gate record carries `lockbox_days_withheld`.

Finding: until then the row tuners scored the newest draws regardless, and only some printed a `LOCKBOX WARNING` when their window overlapped the period.

### Measurement: how much of the stored history was live

*Page: [scoring](scoring.md)*

Finding: on the production history of 26 Oct 2026, of 12,170 scored row-draws 7,301 were live as stored, 2,018 had been rewritten after the draw (recovered) and 2,851 were back-built or late; 502 of 1,124 draws were fully live as stored; no draw after 10 Sep 2026 was back-built or late.

### Lockbox withheld in the four row tuners

*Page: [research](research.md)*

Decided: `HyperoptStatistics.py`, `HyperoptBoost.py`, `HyperoptEnsemble.py` and `HyperoptRLTicket.py` withhold the lockbox days from selection (trials, gate re-scores, standard errors, `modelScores`, the Keno subset day table, the boosting pruner's steps) through `src/Lockbox.py`; fitting on them stays allowed. `TuningControls.py` withholds the same days and compares `lockbox_days_withheld`.

Why: until then two tuners only warned and two did not look, so every score that decided what was served counted those days and the meta-learners' lockbox evaluation rested on rows tuned inside it.

Evidence: windows reached into the period by 22 of 90 tuning draws (euromillions, lotto, eurodreams), 12-13 (vikinglotto, week games), 25-26 of the ensemble's ~100 days, 6 of vikinglotto's 40 RL ticket days. Scored days after the change: 68 of 90 (lotto's served Markov SE 0.0765 to 0.0861), 78 for vikinglotto, 78 and 77 for the week games, 74 of lotto's 99 ensemble days, 34 of 40 RL ticket days.

### Shuffled market control re-cuts its bins

*Page: [research](research.md)*

Fixed: the shuffled market history reorders the returns and re-cuts the bins by the causal rule, so bin frequencies move a little (total variation 0.006-0.035 over the Q2 window). Before, days kept the bins their returns were cut with in their original place. The series of the shuffled market control breaks at this date.

### HMM passes redone in log space

*Page: [markets](markets.md)*

Fixed: a forward or backward pass whose normaliser underflows, and a filter or validation pass that flushed a reachable regime, are redone in log space; the validation slice is scored inside one forward pass; the mixture takes the exact log of the predicted weights.

Finding: a K=4 candidate on about 100 weekly rows could end with an absorbing regime, the others 890-1,060 nats below; a later row summed the scaled step to 0, giving 0/0 warnings, a ValueError and K=4 silently dropped. Seen in the 10 Oct cryptoweek study (trial 6) and two 18 Oct null controls; another window was scored 4,539 nats too low without error.

Evidence: K=2 won in every replayed case, so no served forecast changed; trial 6's K=4 scores -4,047 nats exactly, not -79,627. On served parameters 1 forecast filter in 192 takes the log path and moves by about 1e-12.

Finding: not hardened: an E-step that flushes a state without underflowing (5 of 12,000 fuzz cases, none in about 5,000 real E-steps).

### Frequency baselines under the proper score

*Page: [markets](markets.md)*

Built: `MarketRows.py` scores `random`, `column_frequency` and `global_frequency` by the log-score (add-one smoothing) and every row carries a `vs_frequency` interval; Admin > Controls shows a Beat the frequencies column.

Why: beating uniform is not skill, because the bins are equally likely only over the whole history.

Evidence: on the 25 Oct windows the column frequencies alone beat uniform by +0.068 a day on crypto and +0.014 on shares; on sharesweek `global_frequency` beats `column_frequency` by +0.011.

### Shuffled market control re-cut causally

*Page: [markets](markets.md)*

Fixed: the shuffled control reorders whole days of returns and cuts their bins by the game's causal rule; before, shuffled days kept the bins cut in their original place.

Finding: over the last 120 game days the old bins disagreed with a causal re-cut on 14-28% of slots (agreement 0.72-0.76 crypto, 0.82-0.86 shares, 0.73-0.84 cryptoweek, 0.82-0.86 sharesweek), so GARCH and the HMM rows were scored against bins they could not have predicted.

Evidence: estimated on the 25 Oct seeds, classical suite: crypto 0.872 to 0.813, cryptoweek 0.870 to 0.801, shares 0.634 to 0.612 against a real 0.621, sharesweek 0.544 to 0.511.

Decided: the market Q0 and Q2 shuffled values are a break in the series from the first Sunday run with the change.

### Row tuners withhold the lockbox on week games

*Page: [markets](markets.md)*

Fixed: the row tuners withhold the lockbox weeks from every score; before, the tuner warned and tuned on them anyway. The week games' windows score 78 and 77 weeks until they move past February 2026.

### Items 1 to 3 outlived their pages

*Page: [roadmap](roadmap.md)*

Finding (check of 5 Nov 2026): items 1 to 3 described the old pages, deleted on 26 Oct 2026. The best-combination card moved to Admin > Models; accounts are managed at `/app/admin/people`.

Finding (check of 15 Nov 2026): every signed-in reader can change their own password on `/account` (current password required, the same lockout); there is still no reset without the current password. A user sees the slim user view; the History pages no longer exist as such.

## 25 October 2026

### One-member votes infeasible; market vote weights

*Page: [tuning](tuning.md)*

Decided: a weighted combination that one member outweighs is infeasible like a one-row subset; a served selection in that state is scored as nothing with a note naming the member. On the markets XGBoost + LightGBM alone are infeasible.

Built: the gate record gained `effectiveMembers` and `weightedEqualsFlat` (crypto, shares and sharesweek were "weighted" with every member at 1.0 that day).

Built: on the market games the vote is weighted by each candidate's own score on the run's window (`member_scores`, `subset_weights` to [1, 2]), written as `subsetEnsembleWeights` whatever the gate decides, raw values in `memberScores`.

Why: `modelScores` on the markets cover only the four statistical rows.

Evidence: simulated on that day's files the weights alone lifted crypto's served selection from −0.38 to −0.08, within one standard error of the best challenger; with a weight per member only flat pairs stay in the market search (108 infeasible combinations on crypto, against 41).

Finding: the market plain vote the gate compares with is therefore no longer exactly the served `WeightedEnsemble Model`, which keeps `modelScores`.

### LSTM Smoothed Model added as a second row

*Page: [models](models.md)*

Built: `LSTM Smoothed Model` (`useLstmSmoothed`) trains the same network on the same tuned keys with `labelSmoothing` in the loss of both heads, with its own weights and fingerprint in `data/models/lstm_smoothed_model`.

Why: the pair shows what smoothing does to one model.

Evidence: it cost a second LSTM training per game and day, about 150 s on average.

Finding: the `lstm_model` study sampled `labelSmoothing` but scored LSTM Base's unsmoothed loss, so the stored value (0.02-0.10 across the lottery games) was whatever the best trial happened to draw; the market games had no tuned LSTM keys and used the model default 0.1.

### Voter set moved to src/VoteRows.py

*Page: [models](models.md)*

Built: `POSITIONAL_VOTE_DEFAULT_ROWS` moved to `src/VoteRows.py`.

Why: the `RL Position Model` and its tuner had to read the same voter set as the vote rows.

### Market vote ties go to the middle bin

*Page: [models](models.md)*

Decided: on the market games a tied slot goes to the tied bin nearest the middle of the bins, then the lower one, wherever a market vote is taken or scored (WeightedEnsemble, SubsetEnsemble, RL Position Model, `HyperoptEnsemble.py`). The lotteries keep the lowest digit.

Why: the lowest bin is the most bearish call, so lowest-first leaned every tie towards "no long", a direction no voter gave.

Evidence: replayed read-only on the last 30 production day files, the tie-break changed the market ticket on 2-8 days per market.

### XGBoost and LightGBM count as one market voter

*Page: [models](models.md)*

Decided: on the market games `XGBoost Model` and `LightGBM Model` count as one voter, half the weight each when both vote (`MARKET_SHARED_VOTE_ROWS`). The lotteries tune each library on its own and keep two voters.

Why: on the markets both ran untuned on identical defaults (one depth-3 learner fitted twice on the same features), so every market vote counted one call twice.

Evidence: the two rows agreed on 67-77% of slots. Replayed on the last 30 production day files, the one-voter rule changed the market ticket on 8-14 days per market.

### Dominated SubsetEnsemble selection no longer served

*Page: [models](models.md)*

Decided: a weighted selection is not served on a day one member outweighs the others together (`VoteRows.effective_members`); the tuner scores such a selection as nothing. Exact equality is not a copy, a flat vote cannot collapse, and keno is exempt because its played 5-10 number selections are drawn from the vote's own scores.

Why: the dominant member wins every slot the others disagree on, so the vote is that member's ticket under this row's name.

Evidence: pick3's LaplaceMonteCarlo + Markov pair served Markov's ticket on 35 of 35 days; eurodreams' LSTM Base + LaplaceMonteCarlo pair served Laplace's.

Finding: under that day's weights those two rows paused until the next Saturday tuning replaced their selections.

Decided: on the market games the vote is weighted by the tuner's own per-row scores once written (`subsetEnsembleWeights`), and a selection of only XGBoost and LightGBM is one voter and skipped the same way.

### Warm-start guard keeps loaded weights unless beaten

*Page: [models](models.md)*

Fixed: until this date whatever a warm-started training ended on was saved and served, even when no epoch beat the weights it started from, so the rows that memorise (pick3 `UnifiedLstmGruTcn` and `LSTM Base`, the market Transformer and Autoencoder) drifted further from the held-out draws every day. `Helpers.fit_with_warm_start_guard` scores the loaded weights on the validation split before `fit` and puts them back when no epoch beats them.

Built: `last_val_loss`, which `HyperoptDeepLearning.py` reads, became the val_loss of the weights actually served.

### Weights version 2: every row trained fresh once

*Page: [models](models.md)*

Decided: the fingerprints carry `weightsVersion` 2, so every deep-learning row of every game, the tuner's cache included, trained fresh once on its first run after the change.

Why: the guard alone would have frozen weights that had drifted for two months.

Evidence: pick3 UnifiedLstmGruTcn stood at val_loss 3.52.

Finding: most rows trained to the full budget that day, and the deep-learning tickets were expected to change with it.

### Learning check added as a label only

*Page: [models](models.md)*

Built: `src/LearningCheck.py`: after training each deep-learning row is read on its validation windows (`valLoss` against `baselineLoss` minus `margin`; verdict `learns` or `counts-only`; `inputBlind` at 95% identical tickets), stored on the row and in `modelPerformance.json`, shown only on the admin Models pages as a "no better than the frequencies" tag.

Why: on a draw with nothing to learn a network ignores its input and serves the training split's most common value per position, and a memorising row does worse than those frequencies; on the pages both looked like a model at work.

Evidence: pick3's 604/504 came from four rows, sharesweek GNN's 9909 every week. Margins: 0.004 nats for pick3, 0.11-0.16 for lotto, euromillions, cryptoweek and sharesweek, where the bar errs strict.

Decided: nothing ranks, filters, hides or reweights on the label, and the users' pages do not show it.

Finding: on that day's data every deep-learning row was expected to read `counts-only`.

### Narrower Markov box on the week games

*Page: [markets](markets.md)*

Decided: the statistical tuner searches `MarkovMonteCarlo` on the week games at order 1 only, with a minimum transition count up to 3 (cryptoweek) and 5 (sharesweek) (`MARKOV_MC_WEEK_CHAIN_RANGES`); studies keep their names because the objective is unchanged.

Why: on 223 and 425 weekly draws 97-100% of the order-2 and order-3 transitions seen at all are seen once or twice, so the chain is pruned and falls back to plain frequencies; cryptoweek's study had 32 of 45 trials at order 3 and served order 3 / minimum 20, the fallback on every week.

Evidence: the new maxima keep nine of ten columns live over the 90-week window (cryptoweek 92% at 3, 69% at 4; sharesweek 99% at 5, 85% at 6). On cryptoweek the served fallback scored 0.655, the untuned default 0.625 and 26 points of the new box 0.44-0.57.

Finding: the `Markov Model` row's own chain is just as pruned (cryptoweek live on 4% of the window's columns, sharesweek on none) and was left as it is.

### RL Position votes over the ensemble's voters

*Page: [markets](markets.md)*

Fixed: the row's ticket and its two voter features are computed over the `WeightedEnsemble Model`'s voters by the same rule, XGBoost and LightGBM as one voter; before, the stuck deep-learning tickets, the constant OrderStatistics Baseline, the three Regime HMM rows and the aggregates outvoted them.

Evidence: the ticket matched the WeightedEnsemble's on only 2-5 of the last 30 days per market; on sharesweek it replayed GNN's constant [9,9,0,9] on 9 runs.

Built: policy files carry `featureVersion` 2; a file without it is not warm-started from.

### ZeroMean regimes separated again

*Page: [markets](markets.md)*

Finding: from 10 Oct, when the tuner set `regimeHmmFeatures` to `returns`, the ZeroMean row never separated its regimes on crypto, shares and sharesweek (cryptoweek until 24 Oct): with every mean at zero each state began at the same covariance and stayed a copy (weights exactly [0.5, 0.5]). Its readings of 10-25 Oct are of the collapsed model.

Fixed: in that case only (`GaussianHMM._variance_only`) each restart spreads the regimes' starting covariances; crypto's weights became [0.949, 0.05, 0.001]. Every other row and feature set is bit-for-bit unchanged.

### HMM tickets name the median bin

*Page: [markets](markets.md)*

Decided: the three Regime HMM rows' ticket is the bin that holds the mixture's predictive median; GARCH keeps the most likely bin. Probabilities and the proper score are unchanged.

Why: the bins are quantiles of fat-tailed returns, narrow in the middle and wide outside, so a near-zero-mean forecast put most mass in bin 1 or 8 and the side was set by the edges' asymmetry (ZeroMean went long on BTC, ETH and XRP every crypto day and never in shares).

Evidence: on the last 30 scored days exact hits went from 17, 8, 6 of 150 slots (full, ZeroMean, Single) to 28, 29, 27 on crypto and from 22, 19, 9 to 24, 22, 22 on cryptoweek, moving by five or fewer of 120 on the shares games; crypto's full row went from -0.38 to -0.06 summed log return after fees. One month, not a result.

Finding: nearly every HMM ticket moved to bin 4 or 5; the RL Position Model's `regime` feature mixes the two rules until older days leave its window.

### Bins are not equiprobable in every stretch

*Page: [markets](markets.md)*

Finding: the bins are equally likely over the history each day is cut from, not within every stretch: crypto's outer bins hold about 6% of days each and its middle bins 12.6%, shares' outer bins 12.5% and middle bins 8.5%; crypto's two middle bins each held about 19% of the 30 days up to 23 Oct. A row that keeps naming an over-filled bin beats 10% without skill.

Decided: the expanding window stays; the chance levels (10%, 28%, 50%) are documented as the design target.

### OrderStatistics Baseline on positional games: per-column modes

*Page: [roadmap](roadmap.md)*

Fixed: on the positional games (pick3, Joker+, the market games) the baseline's ticket became each column's own mode with repeats kept in their slots (`Baselines.positional_column_frequency_ticket`, ties to the lowest value, one copy shared with the Backtester's positional baseline).

Finding: until then the ticket went through the set-game construction, whose dedupe filled a slot sharing its mode with an earlier slot from the pooled ranking: pick3 served 804 for the per-slot mode 800 and crypto [4,5,3,6,2] for [4,4,5,4,5].

Decided: the set games were unchanged and the row was not restarted; its record and 'since' ran on across the change.

## 24 October 2026

### The switch to the app (batch 6)

*Page: [operations](operations.md)*

Built: `switchover.js` answered every old page address with a 302 to its app page (`Cache-Control: no-store`), and installed the last handlers (404, 500; JSON under `/api/` and `/council/api/`), with the error stack only in the log.

Why: Express's own error handler prints the stack whenever `NODE_ENV` is not `production`.

Built: the server kept rendering only the sign-in, "page not found" and "something went wrong" pages (`shell.js`).

Evidence: every element of the old pages was mapped to its app home before the switch (`web/PARITY.md`, sections A-D); the gaps found were closed or listed there.

### Boosting pruner had stopped most trials

*Page: [tuning](tuning.md)*

Finding: Optuna's percentile is the share of trials kept, so the default of 25 stopped all but the top quarter. On 24 Oct 224 trials were pruned against 314 completed of 540, and 7 of the 36 studies finished three or fewer.

Finding: with a refit cadence above 1 all blocks fit concurrently, so pruning saved little compute (0.79 trial-hours on 24 Oct); it is kept because it is free and protects the cadence-1 path.

## 21 October 2026

### Decision to replace the old pages fully

*Page: [operations](operations.md)*

Decided: the app would replace the server-rendered pages fully; at the switch the old pages retire at once for every reader, with no grace period and no `/legacy`.

Decided: a user gets the slim view (Home, Lottery with today's top 5 and each past draw's top 5 as ranked before it, Crypto, Shares, the Council's ask box and own sessions); the administrator gets everything else under the Admin menu.

## 20 October 2026

### User pages count what the page said each week

*Page: [operations](operations.md)*

Decided: the user pages counted the plain week book for every week closing before `markets.GUARD_SHOWN_FROM` (26 Oct 2026, the first full week the page showed the guard) and the guarded book from then on; the week of 19 Oct showed the guard as information with the sell at the week's end.

Evidence: production served the guard rows from the evening of 20 Oct.

Evidence: on 20 Oct the two live crypto weeks gave guard -25.02, plain -41.00, market -57.84 USDT; shares -3.92, -9.72, -19.05 USD, mostly from staying out of a falling market.

## 19 October 2026

### Earlier guard weeks marked as booked after the fact

*Page: [operations](operations.md)*

Finding: the rule was chosen on 16 Oct, when all weeks but the newest had closed, and that week's two sales closed before the rule existed.

Built: `MarketSettle.GUARD_FORWARD_FROM` (exported as `forward_from`) recorded the first week the guard ran forward; earlier weeks were marked "booked after the fact" on the workbench card.

## 17 October 2026

### The guarded week rule

*Page: [operations](operations.md)*

Decided: the user page's week positions followed the week game with a daily guard (`MarketSettle.guarded_week`): the week's followed row names the instruments in play, the daily followed row's call decides the held days. Bought at the entry of the first day with an up daily call, sold at the close of a day whose call is flat or down, bought again on the next up call, sold at the week's close at the latest.

Decided: one deviation from "buy at the week's start": a flat or down call on the week's first day buys nothing, so the page never says buy and sell on the same down day.

Evidence: a week held on every day equalled the plain week book to the cent on production's 11 Oct week (BTC -3.38, BNB -5.37, XRP -7.45, SOL -8.62 either way).

Evidence: on the records of 17 Oct the guard changed the one live crypto week from -24.82 to -21.69 USDT and the two live share weeks from -9.72 to -3.92 USD; in the week to 16 Oct NVDA's daily call was flat every day, so the guard never bought the position the plain week lost -6.58 on. In the crypto week of 12-18 Oct the guard sold ETH on the 13th, SOL on the 14th and never bought BNB.

Finding: these figures said nothing yet about skill.

## 16 October 2026

### One headline figure and no "game" wording

*Page: [operations](operations.md)*

Decided: the market pages' headline figures counted what following the whole table made, the daily and weekly positions together, a week's result on the day it settled; the words a reader sees never say "game". The two parts stayed in the record as `status.dailyTotal` and `status.weeklyTotal`.

Evidence: a review of the batch by three readers produced 30 findings, all applied; it made these rules explicit.

## 15 October 2026

### React app started under /app

*Page: [operations](operations.md)*

Decided: the interface was rebuilt as a React application in `web/` (Vite, TypeScript, ECharts), with FreqUI as the model: positions, a trades log, profit per day and per instrument, little text.

Built: the market pages read `GET /api/me`, `GET /api/markets/:market` and `GET /api/quotes/:market`; a signed-out client got a JSON 401 on `/api/*`, never the login page.

Built: from 15 Oct 2026 the per-instrument money of the followed row travelled in the followed-plan points (`MarketSettle.followed_plan`, the ninth element); an older record gave the days but no positions until the next run rewrote it.

### Ensemble tuner put behind the gate

*Page: [tuning](tuning.md)*

Finding: until 15 Oct 2026 the run's best combination was written as it was.

Decided: the best combination is written only if it beats the served selection and the plain vote (every candidate, weighted) by one standard error of its own score; the plain vote is served back when it beats both.

Fixed: a served selection that cannot be served (a dead member, nothing written yet) is scored as nothing, so the row is revived. A first draft scored the selection minus the missing member, a proxy the challenger could never beat by the margin.

Evidence: on keno and pick3 the standard error of a selection's score exceeded the spread between any two selections (0.67 and 1.06 against 0.42 and 0.15 on 15 Oct), so there the gate keeps what is served unless it cannot be served or the plain vote beats it.

### Quantum rows' keno subset keys joined the subset tuning

*Page: [tuning](tuning.md)*

Built: `KenoSubsetTuning` searches the quantum and control rows' subset mode and temperature keys with the classical meta rows'.

Finding: until then they kept the softmax/0.5 defaults.

### Ensemble tuner gated

*Page: [roadmap](roadmap.md)*

Decided: the ensemble tuner runs through the gate; it was the last tuner of the Saturday chain that wrote its result without a comparison. `HyperoptDeepLearning.py` was not in the chain and stayed ungated at that point (gated on 6 Nov 2026).

### Keno subset tuning covers the quantum rows; stale flags pruned

*Page: [roadmap](roadmap.md)*

Built: the Keno subset tuning (`KenoSubsetTuning` in `HyperoptStatistics.py`, `KENO_SUBSET_META_ROWS`) also searches the sub-selection mode and temperature of `QuantumMetaLearner Model`, `QuantumVQC Model` and `ClassicalSVM Model` (`quantumMetaLearnerSubset*`, `quantumVqcSubset*`, `classicalSvmSubset*`), from their artifacts on the same cached day table.

Why: until then those three rows played Keno with the untuned softmax / 0.5 defaults while the classical meta rows were tuned.

Fixed: the 84 per-model `<model>_use_<n>` flags in `bestParams_keno.json`, written by the tuners until September 2026 and read by nothing since, are dropped the next time the statistical or boosting tuner writes the file (`TuningGate.prune_keno_flags`).

### Feature-control card shows each game's column count

*Page: [roadmap](roadmap.md)*

Fixed: the card shows each game's base-model column count instead of saying eight, and gives the chance of one column clearing a two-sigma band by luck for the smallest and the largest count.

Evidence: on the 11 Oct records 6 columns for pick3 and Joker+, 9 for the lotto games and keno, 10 for the market games.

## 14 October 2026

### Quantum tuner put behind the gate

*Page: [tuning](tuning.md)*

Finding: until 14 Oct 2026 the quantum tuner wrote its run's best trial as it was.

Decided: a run's best trial is written only if it beats the served parameters and `QUANTUM_DEFAULTS` (0.25 standing in for scikit-learn's `"scale"` gamma), both re-scored on the run's holdout, by one standard error of the challenger's score; each run judges its own trials only.

Why: cross-week trial scores were measured on different holdout windows and are not comparable.

### Pick3 and Joker+ quantum objective became the slot bound

*Page: [tuning](tuning.md)*

Finding: the old objective was the mean real payout of the argmax ticket (+0.01 × top-1 accuracy); one pair (+46 against −4 on a miss) moved production's 56-day holdout mean by 0.9 a day.

Decided: the objective is the statistical tuner's positional score (slot-hit lower bound plus 0.001 × capped-profit bound), in fresh studies `<game>-<variant>_slot_bound`; the raw payout and top-1 accuracy stay on the record. The market games keep their P&L objective.

### Quantum tuner gated; pick3 on the slot-hit bound

*Page: [research](research.md)*

Decided: `HyperoptQuantum.py` scores pick3 and Joker+ on the slot-hit lower bound, not the raw payout, and every study goes through the champion/challenger gate with a one-standard-error margin. Why: the payout sits at minus the stake on almost every day with rare spikes, so one lucky ticket in a 15-75 day holdout decided a study.

Built: the feature control gained the `inconclusive` verdict (ratio exceeded, pooled importance not two standard errors above zero); Q2 gained *too few draws*.

### Quantum tuner gated on the slot-hit bound

*Page: [roadmap](roadmap.md)*

Decided: `HyperoptQuantum.py` runs its three studies through the gate; the positional games' objective became the common slot-hit bound, and every quantum challenger must win by one standard error.

Why: the review of the gate showed that on pick3 one pair decided all three decisions.

### Regime HMM feature set travels with the results

*Page: [roadmap](roadmap.md)*

Decided: the three Regime HMM rows keep sharing one tuned feature set, because the ablations stop being ablations otherwise.

Built: the `Regime HMM Model` gate record carries `features`; every entry in `data/markets/<market>-regimes.json` carries the row's `features`; the Sunday report (`MarketRows.py`) carries `hmm_features`; the market pages say it in the regime card and in *how rows are judged*.

Finding: the 10 Oct studies chose `returns` alone in all four markets, unannounced. On `returns` alone the *Single* row is a Gaussian with a constant mean and variance per instrument, a random walk with drift, so "not better than *Single*" says the returns hold no regimes in mean or variance; with volatility and momentum in the set it is the linear regression of the returns on them.

### Q2 verdict "too few draws"

*Page: [roadmap](roadmap.md)*

Built: `RandomnessDiscrimination.compute_verdict` marks a record `too_few_draws` when the null band's ceiling (mean + 2 sd of the best-of-suite AUC on fair draws) is 0.80 or more; `--reanalyze` rewrites the saved verdicts and the page reads older records by the same rule.

Why: fair draws alone reach that ceiling when the held-out windows are few, so such a record can show nothing either way.

Evidence: on the 11 Oct records 496 windows gave ceilings of 0.60-0.75, the shares week's 208 gave 0.83, eurodreams' and vikinglotto's 148 gave 0.89 and 1.00, the crypto week's 106 gave 0.97.

Superseded on 27 Oct 2026 by the windows rule in `controlRules.json`.

### Feature control verdict "inconclusive"

*Page: [roadmap](roadmap.md)*

Built: `FeatureControl.summarise` has a third verdict, *inconclusive*: the noise ratio is exceeded but the pooled noise importance is not two standard errors above zero (the repeats disagree).

Why: the card had shown that case as *noise ignored* in green.

### Null report without the special-number rows

*Page: [roadmap](roadmap.md)*

Fixed: `NullControls.selection_report` no longer counts the special-number rows (`<row>_special`: the bonus, the stars, the dream number, scored on another column and scale) among the rows compared; they stay in `per_row_mean`.

Evidence: 16 rows instead of 8 were compared on euromillions, eurodreams and vikinglotto, 10 instead of 5 on Joker+, which pulled the row mean down (by a sixth on euromillions, two fifths on eurodreams and vikinglotto, a third on Joker+) and inflated the selection gain (euromillions synthetic: 0.164 reported, 0.095 over the main rows). The band's ceiling did not move: the special rows never won a seed.

### "Best since the freeze" without back-built market days

*Page: [roadmap](roadmap.md)*

Fixed: for a market game the column counts only the days whose ticket was out before the day's close (`Helpers._days_predicted_before_draw`, on `MarketSettle.ticket_times`).

Why: the back-built history the market games started with on 1 and 5 Oct sat after the 27 Sep freeze date and was being counted as the frozen design's record.

## 13 October 2026

### Explanations moved behind a "?"

*Page: [operations](operations.md)*

Decided: every page showed its data and kept the explaining prose (how to read, definitions, method, caveats, history notes) behind a small round "?" that opens in place (a native `<details>`, no hover).

Decided: titles, summary lines, data, headline results, the plan's orders and a one-line version of any warning a reader must see before acting stayed visible.

### Within-draw Markov chain got a start state

*Page: [tuning](tuning.md)*

Finding: the within chain as built on 5 Oct never played number 1; only the frequency fill gave it a score (about a tenth of its fair share under the `log` blend). The smallest number of a sorted draw is nobody's successor.

Fixed: every draw is read as a chain from a start state (`Markov.WITHIN_START`) to its smallest number and on; the ticket draws its first pick from the start state's table and stops at the draw's size.

Decided: the lotto and euromillions within rows tuned on 10 Oct are re-scored like for like by the next gate.

### Boosting challenger must win by one standard error

*Page: [tuning](tuning.md)*

Decided: the boosting tuner's challenger must also win by one standard error of its own score (`--gate-se-margin` 1); `TuningScore` reports `score_se` and the gate uses the larger of the two margins.

Evidence: the boosting winners of 3 Oct outside keno lost 0.07 on average when re-scored a week later; 11 of the 20 replacements outside keno on 10 Oct had won by less than that.

Why: a boosting fit moves with every new draw through its refit blocks.

### Gate reverts on a tie and on a beaten incumbent

*Page: [tuning](tuning.md)*

Fixed: an exact tie of the run's best trial with the default now reverts. Before, it kept an incumbent worse than both: three rows on each of 3 and 10 Oct 2026 (in five of six cases parameters far from the default that score exactly what it scores; once, vikinglotto LaplaceMonteCarlo, the sampler re-drew the default's 900 simulations).

Fixed: the defaults are also written back when the challenger beats the incumbent by the margin but not the defaults. Without it the incumbent, the lowest of the three, stayed: euromillions XGBoost and LightGBMMultiLabel and pick3 LightGBM on the 10 Oct scores.

### RL ticket tuner gated on the capped bound

*Page: [tuning](tuning.md)*

Finding: until 13 Oct 2026 the RL ticket trials were scored by the plain uncapped mean per day and the study's all-time best was written without a gate; for keno that was a 13 Sep trial on the 1-80 game with one ticket size, for pick3 one 500 EUR straight.

Decided: scored by the shared capped lower bound (keno subsets against a fair ticket of each size, pick3 by slot hits, hits elsewhere) and written only through the gate, in fresh studies `<game>-rl_ticket_gated`.

### Keno day table refits its boosting row weekly

*Page: [tuning](tuning.md)*

Finding: with the XGBoost the 3 Oct boosting gate served (one keno fit measured at 430 s) the daily refit took the keno day table past its hour after 61 of 90 days on 10 Oct 2026; `KenoSubsetTuning` kept its 3 Oct settings for a week.

Fixed: the table's boosting row refits once per week of the window (`KENO_DAY_TABLE_REFIT_EVERY`); the Backtester cuts its worker blocks on the same refit slices.

### Money games ranked on the capped profit bound

*Page: [scoring](scoring.md)*

Decided: keno, Pick3 and Joker+ were ranked on the profit bound: per draw the profit per bet with every bet capped at 20 EUR (a keno bet against a fair ticket of its size), mean minus one standard error; the raw profit per bet kept its own column.

Why: the raw profit per bet was decided by one winning ticket.

Evidence: on 13 Oct 2026 a 65-bet keno row with one 9-number ticket at 7 of 9 led a 513-bet row, and Pick3's best row was one straight.

### Fresh control seeds every week

*Page: [research](research.md)*

Decided: the Sunday plan runs on `--seed weekly`: (ISO year x 100 + ISO week of the weekend's Saturday) x 10000 + 1 on, for Q0, Q2 and the feature control's noise columns; five histories instead of three for the four market games (`--market-repetitions 5`). A hand run keeps seed 1; `NullControls.py` removes a game's control folders of earlier seeds.

Why: seeds 1-3 rebuilt the same synthetic histories every week, appending only the newest draws, so a second week re-ran the first week's test.

Built: `current_era` also drops lotto before 1 October 2011 (6 of 42 until then), 1747 draws, by date.

### Sit-out floor after the position gate

*Page: [markets](markets.md)*

Built: `tune_position` sets `rlPositionSitOut` when what the gate serves does not beat 0 by the gate's margin; the row is then served at size 0 everywhere with the vote's ticket, its tuned knobs kept for the next comparison. The record carries the trial's mean size so a sit-out can be told from a sizing. The lottery ticket studies went under the same gate that day.

### Following the plan record

*Page: [markets](markets.md)*

Built: `MarketSettle.followed_plan`: per settled day the row the page's rule would have followed, picked from the record before that day, its money under the plan's rule against buying everything; a day is live only when its ticket was made before the plan's entry (`entry_instant`). `Predictor.py` stamps `madeAt` into day files (`stampTicket`).

Fixed: `update_matching_numbers` keeps a day file's modification time when a missed-day recovery re-links files; rewriting it would have turned every past live day into a back-built one.

Evidence: first reading on a copy of production: the daily crypto plan, live since 1 Oct, made -10.21 USDT over 12 days against -48.39 for buying every coin every day; shares -1.50 USD over 8 sessions against -13.41; the week cards one live week each, -24.82 against -31.98 and -3.14 against -3.85. Days, not evidence.

### Sized row booked wrongly in day records

*Page: [markets](markets.md)*

Fixed: the Day by day records and the results CSVs (`pnl_cash`) read a position through `position_of`; they had booked the RL Position Model by the bin rule at full stake, so the page showed it trading on days it sat out.

Evidence: over 30 weeks per week game: crypto +14.79 USDT shown against a sized -47.05, shares +43.74 USD against a sized +10.36.

### Plan cards open with labelled lines

*Page: [markets](markets.md)*

Finding: the plan card's headline (+23.17 USDT from 81 positions) was read as what following the card had made, and the Models calling up column as the models deciding the orders; neither was so. A first draft said "one of the 28 models" next to a column reading "of 29".

Decided: plan cards open with three labelled lines: One model's calls (the model's record, not a reader's), Following this card so far (the live total of the followed record, paper money, against buying everything) and the sell at the close. Counts match the adjacent table, the ahead/behind phrase is computed from the figures as printed, the Models calling up column is labelled context, and a settled week is named by its end.

### Second control reading: runtimes, keno artefacts closed

*Page: [roadmap](roadmap.md)*

Evidence: every job of the 10-11 Oct weekend exited 0: statistical tuner 5.6 h against 4.2 (the keno day table alone about an hour, its 3600 s budget), boosting 8.2 h, quantum 3.5 h against 7.2 (all of it the keno score table), Q0 12.0 h.

Finding: the keno artefacts of 5 Oct closed in all three controls on the 1-70 game: the null control's random ticket hit 5.74 against a fair 5.71 and no keno row cleared the band (6.30); Q2's keno AUC fell from 1.000 to 0.607, *no evidence*; the feature control's keno flags went with the 80-number table (held-out AUC 0.49-0.52).

### Second control reading: nothing above a band week after week

*Page: [roadmap](roadmap.md)*

Finding: vikinglotto's `MarkovBayesian Model` was above for a second week on one new draw (46 draws, inside a band scaled to its record); crypto's five rows above 0.878 rose with the whole table on a mostly back-built record; lotto's seven were short-record rows (three deep-learning rows, LightGBM and three aggregates); shares' Transformer fell back inside.

Finding: Q2 repeated crypto's *above both controls* (0.909 against a shuffled 0.863). Shares turned *not temporal*: its real AUC (0.631, was 0.628) cleared a null threshold that fell from 0.639 to 0.627, but stayed below its redrawn shuffled control (0.652, was 0.613). The week games', eurodreams' and vikinglotto's Q2 had too few draws for a verdict (thresholds 0.83-1.0).

Finding: none of the first week's feature-control flags repeated; lotto's quantum-kernel `Chronos Model` column was the one above its band twice.

### Second control reading: market rows

*Page: [roadmap](roadmap.md)*

Finding: the `Chronos Model` row was back (second in crypto, behind GARCH). GARCH kept +0.092 nats over uniform in crypto and +0.032 in shares, most of it the period's bin frequencies rather than timing (a constant forecast at its own average scored +0.073 in crypto). The full `Regime HMM Model` row beat uniform after its first tuning (crypto +0.049, shares +0.017, the shares part inside the tuning window); its two ablation rows were worse than uniform in crypto (-0.065 each) and no different in shares.

### Second control reading: defects fixed the same day

*Page: [roadmap](roadmap.md)*

Fixed: the position gate's missing zero floor; the full-stake booking of the sized row; the gate's tie case; the keno day table's daily XGBoost refit; the within-draw Markov chain's unreachable number 1.

### Second control reading: design defects listed for decision

*Page: [roadmap](roadmap.md)*

Finding: the RL ticket and quantum tuners wrote without a gate (keno served an RL ticket trial of 13 Sep from the 1-80 game with one ticket size; pick3's new knobs came from one 500-EUR straight on an uncapped mean).

Finding: boosting replacements were mostly selection luck: outside keno the previous week's winners lost 0.07 on average when re-scored, and 11 of the week's 20 replacements outside keno (26 in all) cleared the gate's bar by less than that.

Finding: the Regime HMM's two ablation rows share the full row's tuned knobs, so its returns-only choice turned the *Single* ablation into the random walk with drift.

Finding: the control histories reused seeds 1-3 every week (a synthetic history never resampled; Q2's null band was byte-identical for the long games), three seeds made the shuffled band jumpy, and lotto's shuffled control mixed the 6/42 and 6/45 eras.

Finding: the ranking named a keno best model by profit per bet from ten draws on, which one winning ticket decided (`QuantumVQC Model` on 65 bets).

Finding: two input columns were degenerate: eurodreams' XGBoost scored 1-50 in a 1-40 game and vikinglotto's was constant.

### Lotto's 6-of-42 era excluded

*Page: [roadmap](roadmap.md)*

Decided: `Helpers.GAME_ERA_START` gained `lotto: 2011-10-01`, the day Belgian Lotto moved from 42 to 45 balls.

Why: 1747 of 3315 draws, half the history the whole-history lotto rows (statistical, boosting, meta-learner) and the controls read, came from a game in which 43, 44 and 45 could not be drawn (no main number above 42 is drawn before that date in the files).

Fixed: `era_start` matches a whole folder name; by substring "vikinglotto" would have inherited lotto's cut.

The deep-learning rows read ten years and lost nothing; the cached meta-learner tables key on the row count and were recollected by the next chain.

### Gates, margins, fresh seeds and the profit-bound ranking

*Page: [roadmap](roadmap.md)*

Decided: the RL ticket studies are gated and scored by the shared capped bound.

Decided: boosting replacements must win by one standard error.

Decided: the controls draw fresh seeds every week, five histories for the market games (`jobs.js`, `runHyperopt.sh`).

Decided: the payout games are ranked by the profit bound.

Evidence: on the 13 Oct data keno's best stayed `QuantumVQC Model`, now on 40 draws of a size-10 record well above a fair ticket rather than on its one 9-ticket; pick3's moved from the row with one straight to `LightGBM Model`.

### Two degenerate XGBoost input columns: cause

*Page: [roadmap](roadmap.md)*

Finding: nothing failed and no fallback ran; XGBoost's own `min_child_weight` rule produced them. Under `multi:softprob` every class starts at 1/K, so each training row adds 2 x (1/K) x (1 - 1/K) to a node's Hessian sum (0.041 at K = 48); a leaf whose sum stays under `min_child_weight` gets weight 0, and with too few rows every tree is a single zero leaf and `predict_proba` is exactly 1/K.

Evidence: vikinglotto's served value (8.49 at subsample 0.53) needed about 396 training rows; its 306 draws gave at most 262, so the row scored every number 1/48 on every day of the meta-learner table and played the same degenerate ticket ([1, 45, 46, 47, 48] or close) on six of its eight day files since 19 Aug.

Finding: eurodreams hit the same wall on the table's oldest days only, because `HyperoptQuantum.py` built its 300-day tables from row 6 of a 306-draw game with no history floor; its column also spanned 1-50, the default of `Helpers.get_unique_labels`, which has no eurodreams branch (the deep-learning fingerprints carry its class count).

Finding: the degenerate tickets came from the refill: uniform probabilities put number 1 first in every position and collisions were refilled from the `top_k` candidates only; with `top_k` 4 that pool cannot fill six slots. The same short-ticket bug gave keno's and eurodreams' LightGBM rows (`top_k` 1) tickets of 17-19 and 5 numbers in September.

### Two degenerate XGBoost input columns: fix

*Page: [roadmap](roadmap.md)*

Fixed: `XGBoostBackend._fit_classifier` caps `min_child_weight` at half the root's expected Hessian sum (rows by weight x subsample x 2(1/K)(1-1/K)), which never binds on the long games.

Fixed: the per-position refill takes the `top_k` candidates and then every other number by inclusion probability.

Fixed: `BoostingBase.load_numbers` uses `Helpers.main_number_labels` (eurodreams 1-40); `Predictor.addRLTicketPrediction` and `HyperoptRLTicket.tune_ticket` keep their own copies.

Fixed: `TrainMetaLearner.meta_table_start_index` gives the quantum tuner the trainer's history floor, and cached tables carry a format (`META_TABLE_FORMAT` 2) so every table was recollected once.

Evidence: on a copy of production's data with the served settings, vikinglotto at 1, 164 and 300 draws back and eurodreams at 1 and 165 back scored 0.01-0.08 rather than a constant, with six distinct numbers in range.

The served vikinglotto `xgBoost*` values had been tuned while the row was constant; the next boosting gate re-scored them like for like. The constant tickets stay in the history.

## 10 October 2026

### First chain comparing the two Markov modes

*Page: [tuning](tuning.md)*

Finding: the Saturday chain of 10 Oct 2026 was the first fair comparison. It chose `within` for lotto and euromillions on margins of 0.4 and 0.6 standard errors (lotto: best column trial 0.8140 against within's 0.8148), kept column for eurodreams and vikinglotto, and for keno sampled within only, every trial below both column references.

### First position gate served sizings below zero

*Page: [markets](markets.md)*

Finding: the 10 Oct gate, which compared only with the served and the untuned knobs, served sizings scoring below 0 on crypto (-0.015 at a mean size of 0.03) and shares (-0.126). A policy that sits out everywhere scores exactly 0, which is the ceiling of a market whose calls lose money net of fees.

## 8 October 2026

### Keno serves all six ticket sizes

*Page: [models](models.md)*

Finding: from 8 Oct 2026 keno served all six ticket sizes (`use_5` to `use_10`), not `use_10` alone.

### Deep-learning ranking changed to summed inclusion probability

*Page: [scoring](scoring.md)*

Fixed: the deep-learning rows' number ranking had been the per-number softmax maximum over the positions; it was changed to the probability summed over the main positions.

Why: the maximum favoured numbers at the edges of the range (keno subsets such as 1, 4, 63, 66, 70).

### Per-position rows' keno subsets picked the range edges

*Page: [research](research.md)*

Finding: the first morning with all six sizes served, four of seven deep-learning rows gave the 5-subset [1, 4, 63, 66, 70], the others within two numbers, and the per-position boosting rows were close ([1, 3, 4, 67, 70], [2, 3, 68, 69, 70]). `Helpers.score_numbers_from_prediction` scored a number by its highest probability at any one position; sorted positions are order statistics (slot 1 gives number 1 a probability of 2/7, slot 20 gives 70 the same), so a row that learned only the marginals always ranked the edges first.

Fixed: the score is the inclusion probability summed over the positions (flat 20/70 on a fair table); the boosting rows slice keno subsets from `BoostingBase._inclusion_probabilities`. This also reordered the lotto multi-pick of the deep-learning rows. `python3 -m src.Helpers` checks the property on the exact order-statistic table.

Evidence: the stored LSTM softmax spanned 0.25-0.33 and its 5-subset under the sum was [1, 16, 32, 42, 60]. Keno boosting values of 3 Oct (xgBoostTopK 16, lightGbmTopK 3, catBoostTopK 29) were selected through the old read-out. Subsets recorded before this date were edge tickets.

### Edge numbers remaining in two keno rows

*Page: [research](research.md)*

Finding: `WeightedEnsemble Model` votes over every row's twenty-number ticket; the thirteen per-position tickets all carry 70 and nearly all carry 1, so the vote kept 1 and 70 on every day of 1-8 Oct. The foundation rows (`Chronos` [1, 3, 25, 68, 70], `TimesFM` [1, 5, 44, 65, 70]) pool the positions, but their quantile read-out clips out-of-range mass onto the first and last label. Neither was fixed by the inclusion-probability change.

Finding: the deep-learning rows' twenty-number tickets are the order-statistic modes to within near-ties (five of seven rows within one number of the same ticket on 8 Oct): a per-position model of a fair draw has nothing else to learn.

### Keno draws test fair at number, pair and overlap level

*Page: [research](research.md)*

Evidence: on the 6202 draws of the 1-70 era, number chi-square 42 against 48 ± 9 in twenty simulated fair histories, pair chi-square 2103 against 2185 ± 164, consecutive-draw overlap 5.714 against 5.714.

Decided: the nested read-out stays; there is no joint structure a size-specific model could use that a per-number ranking does not carry.

### M4 part 3: position tuner

*Page: [markets](markets.md)*

Built: `HyperoptRLTicket.py` opens a `<market>-rl_position` study for the four market games; each trial walks forward over the newest 40 scored days (30 weeks for a week game) and scores the lower confidence bound of the per-day capped money per position; the study goes through the gate and records the plain rule on the same days; the row never enters `modelScores`.

Evidence: on a scratch copy of production a four-trial crypto study took 88 s on 36 days, a cryptoweek study 64 s on 30 weeks.

Finding: the 20 cap bites often on a week game, where a quarter of crypto's instrument-weeks move more than ten percent; `rlPositionTrainDays` cannot move a score until a market has more than 60 scored days.

### Week games in tuners and controls

*Page: [markets](markets.md)*

Built: `cryptoweek` and `sharesweek` joined the statistical tuner, the meta-learner retrain, the ensemble tuner, `TuningControls.py` and the Sunday plan's controls; `MarketRows.py` and `TrainMetaLearner.py` keep a history floor of 100 rows for a game shorter than their window.

Fixed: `ControlHistories.control_market` cuts a week game's synthetic history after `MIN_WEEK_HISTORY` weeks; with the daily cut-off only the newest 71 of cryptoweek's 221 weeks and 272 of sharesweek's 422 would have had a cut and the control would have stopped.

Decided: the quantum tuner skips the week games unless named (three studies per game on a seven-hour step, for a few hundred weekly draws); the boosting tuner tunes no market game.

Finding: the week games' 90-draw tuning window holds 12 (cryptoweek) and 13 (sharesweek) lockbox weeks.

## 7 October 2026

### Keno is a 1-70 game everywhere

*Page: [research](research.md)*

Decided: after assessing a proposal for one model and one hyperopt per subset size, no per-size models. Why: no base row's objective sees a subset; the fair expected loss is 46-48 cents per euro at every size; for per-number probabilities around chance the top-k by probability is the expected-value-optimal k-subset (checked by enumeration), so the sizes are slices of one ranking. Six models per row would cost about nine more hours of Saturday tuning and six gates per strategy.

Fixed: `Helpers.get_unique_labels` says 1-70 for keno (ten dead softmax classes went), and `Helpers.load_data` drops draws of an earlier game version (`GAME_ERA_START`). The deep-learning keno rows retrained once; the six control folders of 4 Oct held 1-80 draws until the Sunday plan rebuilt them.

### Keno subsets follow each row's own probabilities

*Page: [research](research.md)*

Fixed: `MarkovBayesian` and `MarkovBayesianEnhanched` cut their subset with an ascending sort and a constant Markov term, serving the ticket's coldest numbers; `PoissonMonteCarlo`, `PoissonMarkov` and `LaplaceMonteCarlo` drew a weighted sample tilted to the ticket's lowest numbers; `Markov` and `MarkovMonteCarlo` ranked by global frequency while the tuned `markovSubsetSelectionMode` was never read; the deep-learning rows sampled at temperature 0.5 and could return a repeated number twice. Every row now ranks its ticket by its own per-number mass and takes the top k (`Helpers.generate_subset_from_scores`, mode `top`).

Decided: `markovSubsetSelectionMode` retired, not revived: the stored softmax would have sampled almost uniformly (masses span 0.27-0.30).

### Keno bets scored against the fair capped value

*Page: [research](research.md)*

Built: `TuningScore.fair_capped_ev`, what a fair k-ticket is worth per euro under the 20 EUR cap (-0.64 at k=5, -0.52 at 6, -0.67 at 7, -0.64 at 8, -0.68 at 9, -0.70 at 10; raw expectation -0.46 to -0.48), subtracted from every keno bet's capped profit so a fair ticket scores 0 at every size. The ranking shows profit per bet and bets per size (`profit_by_size`).

Evidence: within one run the baseline is a constant and changes no gate decision; the bound's penalty still differs under the null (about -0.17 for a fair 6-ticket over 90 days, -0.10 for a 10-ticket). Earlier keno gate scores and trial values are on the old scale.

Decided: served sizes stayed `use_10` alone for the moment (last written 23 July 2026); serving all six was left as a choice, and all six were served from 8 Oct 2026. A per-size hyperopt, if ever, belongs on the slicing read-out (`KenoSubsetTuning`: 15 trials in 28 s after a 22-minute day-table precompute).

## 5 October 2026

### Markov transition mode became a tuned knob

*Page: [tuning](tuning.md)*

Decided: `markovTransitionMode` offers `column` (default) and `within` for the set games, scored on the same days under the gate; the positional games keep the column path only.

Why: the tuner, not a table of old hit rates, should answer which definition is better.

Evidence: stored day files, within-draw era (12 Nov 2025 - 2 Feb 2026) against column era (4 Feb - 5 Oct 2026): keno 6.04 hits over 82 draws against 5.53 over 243 (random 5.71; the column tickets carried 55 numbers above 70 and six duplicates, fair baseline 5.63); lotto 0.65 over 23 against 0.79 over 70 (random 0.80); euromillions 0.48 against 0.46 (0.50); eurodreams 0.88 over 24 against 0.63 over 27 (0.90); vikinglotto 0.50 over 10 against 0.83 over 35 (0.75).

Finding: the keno figure was 1.7 standard errors above chance; the best of eight rows scores 6.14 on a shuffled keno history; the two rows still on the within-draw table sat at 5.84 and 5.54 over 330 draws, chance.

Built: the `within` path differs from 2025 in three things: today's log-space softmax, bonus and star columns left out, and (from 13 Oct) a start state.

### Keno control drew the wrong game (1-80)

*Page: [research](research.md)*

Finding: keno's range in `HyperoptStatistics.GAME_CONFIG` and the boosting tuner's mirror stood at 1-80, while the game has drawn 20 of 1-70 since 9 March 2008 (the 58 draws of 2 January to 8 March 2008 are the earlier game, 56 with numbers above 70). The synthetic control drew the wrong game; Q2's 4 Oct keno verdict (AUC 1.000 on real and shuffled alike) was that artefact; the random-ticket baseline scored 4.96 and 5.03 hits where a 1-70 ticket scores 5.71; the keno meta-learner tables carried ten never-drawn numbers as free negatives. The live `Markov Model` and `LaplaceMonteCarlo Model` keno rows had picked fourteen numbers above 70 in the 120 days before.

Fixed: range 1-70 in both tables; `ControlHistories.read_real_rows` drops draws of an earlier version of a game (`current_era`); the literal 1-80 fallback in `Markov.generate_best_subset` took the model's range.

Evidence: keno tuning and meta-learner numbers recorded before this date were measured over 80 numbers and are not comparable in the last decimals.

### Chronos row missing from the Sunday market report

*Page: [research](research.md)*

Fixed: `MarketRows.py` never precomputed the foundation rows before the Backtester forked, so the Chronos row was unavailable in every worker and silently absent from `data/controls/markets/<market>-rows.json`. It now calls `ModelFactory.prepare_foundation_scores`, records `foundation_days`, and writes a row that scored no day to `errors` with the reason.

### First control reading: runtimes and null band

*Page: [roadmap](roadmap.md)*

Evidence: every Sunday job finished: Q2 11 min, feature control 38 min, market rows 10 min, Q0 13.8 h (keno's six histories alone took half a day).

Finding: in the lottery games the only ranked row well above its band was vikinglotto's `MarkovBayesian Model` (1.044 against 0.961 over 45 draws). The others above were young or marginal: lotto's four ten-draw deep-learning rows, eurodreams' `LightGBM Model` by 0.004 over 13 draws, Joker+'s `TimesFM Model` by 0.01 over 13. Pick3's six rows above its band were read on average set hits while the game ranks on profit, so its band was information only.

Finding: shares' `Transformer Model` (0.75 against 0.675 over 32 draws) was the one to watch; crypto's four rows above its band (LaplaceMonteCarlo, Markov, Autoencoder, the OrderStatistics baseline) were a 34-day record held against a band measured on 120-day histories.

### First control reading: Q2 and the feature control

*Page: [roadmap](roadmap.md)*

Finding: Q2 read *no evidence* for every lottery game and for shares. Crypto read *above both controls* (real 0.909, shuffled 0.858, null threshold 0.677): the shuffled part is the cross-coin correlation the per-coin random walk lacks and the rest volatility clustering, both direction-free and both what the GARCH row models, so expected rather than evidence. Keno's verdict was an artefact of the 1-80 range defect.

Finding: feature control: 34 of the 45 variant fits *fit noise* and held-out AUC sat at 0.46-0.60 in every game. Keno's quantum kernel and VQC put all nine base rows above the noise band, which the 80-number table may explain (the ten numbers never drawn make every real column zero exactly where the label is always zero). Elsewhere one or two columns in one or two variants (lotto, euromillions, crypto): first-week flags.

### First control reading: market rows under the proper score

*Page: [roadmap](roadmap.md)*

Finding: GARCH was better than uniform in both markets (+0.093 nats over 217 crypto days, +0.029 over 189 shares days). Against uniform the Regime HMM rows were no different in shares; in crypto only ZeroMean was no different and the full and Single rows were worse; the base rows were worse or no different. Against GARCH every other row was worse in both markets.

## 4 October 2026

### Ensemble tuner extended to the market games

*Page: [tuning](tuning.md)*

Built: since 4 Oct 2026 the market games are tuned too, with the fixed rule in bin space as the day's score (`Helpers.market_ticket_profit`).

### Hold-while-up book

*Page: [markets](markets.md)*

Built: `MarketSettle.hold_book`: a position opened on an up call is kept while the next call is up and sold at the close of the last up day, so a run costs one fee pair and the stake compounds; benchmark buy-and-hold from the first settled day to the last.

### Long-and-short twins of the books

*Page: [markets](markets.md)*

Built: every book has a long-and-short twin where a lower-half bin is a short, with the same stake and fees; the sized RL row never shorts. Each view draws buying everything and selling everything.

Finding: every long-and-short book was in the red on both markets: against a rising month a short pays the drift and both fees (a long and a short of one instrument-day cost four fee legs, 0.40 on the stake), and 42% of the crypto shorts were bin-4 calls, which cannot cover two fee legs. LightGBM's down calls still beat random short days.

Decided: shorts are paper only; the plan names a paper short on a down call and gives no order.

### Open-to-close book for shares

*Page: [markets](markets.md)*

Built: a third rule for shares: the stake bought at the session's open and sold at its close (`ledger` `open`, `OPEN_RULE_MARKETS`).

Why: it is the one rule a reader can follow, since the ticket is up before the session opens; crypto's UTC day opens at the previous close, so its open book would repeat the daily one.

### Today's plan card and honest fill wording

*Page: [markets](markets.md)*

Built: the pages open with a Today's plan card that turns the best book's calls into two orders (for a share a market order before the open and one before the close; for a coin buy now, sell at the close of the UTC day); the followed book needs ten positions (`MIN_PLAN_TRADES`), else the best exact rate. Supporting cards moved into one collapsed Background card.

Finding: the daily book buys at the previous close, for crypto the 00:00 UTC price eight to ten hours before the ticket was on the page (the daily run started at 09:00 Belgian time and finished between 10:06 and 10:38 on most days), so the book is the model's call scored on the daily candle, not what a reader could have made. The pages say so and name when a ticket was written and when it is judged.

### M5: the week game

*Page: [markets](markets.md)*

Built: `cryptoweek` and `sharesweek`, one draw per week (`MarketGame.weekly_returns`), ten quantile bins of the instrument's past weeks with 100 weeks minimum, own instruments in the store reading the base market's bars, the Week ahead card.

Why: a hold planned at the open needs a multi-day forecast, and every row predicts the next draw after the newest one, so a rolling five-day return would predict a window already half over.

Decided: a week counts only once it is over and its last bar is in the store (`week_closed_by`), because a lagging source would otherwise make a six-day week dated a day early, re-cut the next morning, and orphan its ticket.

Finding: GARCH, Regime HMM and RL Position run on weekly returns with their daily knobs, which is crude.

### M4 part 2: RL Position Model

*Page: [markets](markets.md)*

Built: `src/RLPositionModel.py`: per instrument and day a size from 0, 0.5, 1, 1.5 or 2 times the stake, linear softmax policy trained by REINFORCE on the stored day files, reward money − `rlPositionRiskAversion` × money² (default 0.1); its ticket is the vote's, its contribution the sizes that every book applies.

Why: expected money is linear in the size, so a risk-neutral learner would only choose 0 or the maximum; the quadratic term makes a middle size right for an uncertain call.

Built: the weekly retrain trained the positional meta-learners of both markets without code changes; `HyperoptEnsemble.py` included the two markets, scored with the fixed rule in bin space.

## 3 October 2026

### Lotto rows in probability order with three extra numbers

*Page: [scoring](scoring.md)*

Decided: every Lotto row kept six numbers but listed them most probable first instead of small to large, and named the three next most probable unplayed numbers (`multiPick`, `src/MultiPick.py`) for a system play of 7, 8 or 9 numbers (7, 28, 84 grids at 1.50 EUR).

Why: hits of a set game are set hits, so the scoring did not move; the extras test whether the ranking's tail carries signal.

Built: the RL Ticket Model's rank feature was changed to read every ticket sorted ascending, so its stored policy kept the meaning it was trained on. Day files kept the six as the ticket, so existing readers were untouched. Without a meta-learner artifact the daily run computed the rankings itself (about ten seconds a day).

### Multi-pick read against chance levels, not raw hits

*Page: [scoring](scoring.md)*

Decided: the extras were scored against exact hypergeometric levels: 3 or more main hits 2.38% with six numbers, 3.94% with seven, 5.94% with eight, 8.39% with nine; a single extra number is drawn 13.3% of the time.

Why: more numbers win more by arithmetic alone (nine numbers: 3.5 times the chance at 84 times the price), so "do nine hit more than six" had a known answer before any data.

Decided: the reading plan (roadmap item 10): after about fifty Lotto draws (spring 2027) read the extra-number hit rates with a binomial interval against 13.3% and the nine-number win rate against 8.39%, per row and pooled; only then reconsider how the first six are built (an argmax-by-ranking ticket, tested on the backtest first); a euro value of the system play needed the prize per rank per draw.

### M4 part 1: market row knobs in the tuner

*Page: [markets](markets.md)*

Built: `HyperoptStatistics.py` strategies `Garch` and `RegimeHmm` for the market games, through the champion/challenger gate, scored by the lower confidence bound of the mean log-score (`TuningScore`, `proper=True`).

Why: a near-one-hot forecast wins hit rates and loses the log-score the moment it is wrong.

Decided: the three Regime HMM rows read the same `regimeHmm*` keys (tuned apart they would stop being ablations); the served score is not written to `modelScores`, because a log-score of about -2.3 next to slot-hit scores of about 0.7 would pin itself to the lowest vote weight.

### First chain under a valid lockbox; reading the bands

*Page: [roadmap](roadmap.md)*

Built: Saturday 3 Oct 2026 was the first chain under a valid lockbox, with the third quantum-tuner study per game and the retrain that wrote the `ClassicalSVM Model` artifact and the corrected VQC circuit; Sunday 4 Oct brought the first `ClassicalSVM Model` prediction and the first Sunday plan with all three controls.

Decided: with eight base models tested against a two-sigma band, one clears it by chance about one week in five, so only a column that stays above the band week after week counts.

### Roadmap item 10: lotto multi-pick step 1

*Page: [roadmap](roadmap.md)*

Built: three extra numbers per lotto row for a 7-, 8- or 9-number system play, the rows ordered by probability, and the scoring against the exact chance levels.

Decided: the reading after about fifty draws belongs with the weekly control readings: a 7th-9th number that hits more than 13.3% over a few weeks is what the null band says about the best of thirty rows, noise unless it stays.

Finding (check of 5 Nov 2026): the prize tables step 4 needs exist since 3 Nov 2026; the system-play euro value does not.

## 2 October 2026

### Paper trading with a fixed stake

*Page: [markets](markets.md)*

Built: every long call is paper-traded with a fixed stake of 100 (USDT or USD) bought at the previous game day's close and sold at the day's close, 0.1% fee on each leg, summed per model into a book next to a market benchmark that buys every instrument every day (`MarketSettle.ledger`, `cash_pnl`, `pnl_cash` in the CSVs).

Why: a fixed stake rather than one coin, because one BTC and one XRP are not comparable positions; a book has to beat the benchmark before it means anything.

Decided: a model with fewer than ten settled days ranks after the others (`MIN_RANK_DAYS`).

## 1 October 2026

### Three sections: Lottery games, Crypto, Shares

*Page: [operations](operations.md)*

Decided: Home became the choice between Lottery games, Crypto and Shares, each with its headline, and the menu repeated them, followed by the Council and the Admin menu.

Why: the market "games" were hard to read next to the lottery ones.

Decided: a market was explained in market terms first, with its draws as digits (the game view) second. The workbench's *Background: how a day becomes a draw* kept three steps written for a newcomer: two closes give one number (the day's move); the move falls into one of ten bins cut from the instrument's own past moves, about one day in ten each; the bins side by side are the draw.

Finding: the first version of that explanation was too abstract; a newcomer persona and a trader persona reviewed the rewording.

### A flat call drawn as a price: no lag

*Page: [operations](operations.md)*

Finding: a call drawn as a price is the previous game day's close moved by the bin's representative return, so a flat call sits on the previous close and a line of them looks like the close a day late.

Decided: the price view stayed the default and the page said that this is not a lag; the moves view showed the real move against each call's interval.

### GARCH and Regime HMM tuned on the log-score

*Page: [tuning](tuning.md)*

Built: the `Garch` and `RegimeHmm` strategies tune the market rows through the same studies, gate and record, scored by the lower bound of the per-day mean log-score (floored at `1e-4`; uniform −2.303).

Why: hit rate is not a proper score and a near-one-hot forecast games it.

Decided: the three Regime HMM rows share the `regimeHmm*` keys, the template knobs are not searched, and the served score stays out of `modelScores` (another scale than the slot-hit scores the vote weights are mapped from).

### Home page as three kinds of sequence

*Page: [markets](markets.md)*

Decided: the home page is a choice between Lottery games, Crypto and Shares, and the top bar repeats it; the former home page became the lottery section's New predictions. The market pages open with a worked example of how a day becomes a draw and a day-by-day list of settled days.

## 30 September 2026

### Market rows joined the Sunday plan

*Page: [operations](operations.md)*

Built: `MarketRows.py` was added to the Sunday plan after the feature control (roadmap item 4, M3).

### Server time zone set to Europe/Brussels

*Page: [operations](operations.md)*

Decided: the machine's local time became Europe/Brussels (it was UTC), so a 09:00 slot meant 09:00 Belgian time and followed summer and winter time.

Finding: a `pm2 restart` was needed once for the server process to read the new zone, because Node fixes its zone at start.

### Market games joined every control

*Page: [research](research.md)*

Built: crypto and shares joined the controls with their own fair history (a geometric random walk with the real volatility) and their own shuffled history. Their week games joined on 8 Oct 2026.

### Markets track: SQLite store and daily bars

*Page: [markets](markets.md)*

Decided: the markets track is a predictor with tracked predictions and honest scoring, not a trading bot. Storage is SQLite inside the repo (`data/markets.sqlite3`; the repo already carried `db.sqlite3` for Optuna, so no new service) and the cadence is daily bars, one draw per trading day.

Why: daily bars keep parity with the lottery framing and the daily cron rhythm.

### Design: returns as a positional game

*Page: [markets](markets.md)*

Decided: each instrument's next-day return is cut into K quantile bins fitted on the training window only; a day's draw is the vector of bins across instruments, a positional game (instrument = position, bin = digit) run by the pick3/Joker+ machinery. Scoring is per-position hits (exact, adjacent) plus the paper P&L of a fixed rule (long when the predicted bin is above the median, else flat, minus a fee).

Why: models stay untouched because they only see integer sequences; the bar is the efficient-market prior (random-walk control, shuffled control, untouched holdout) before any accuracy above 1/K is read as predictability.

Decided: settlement is per instrument at its next available bar, so a Friday share prediction settles on Monday and no artificial holiday rows exist.

### Universe frozen at kickoff

*Page: [markets](markets.md)*

Decided: start small and fixed. Crypto: BTC, ETH, BNB, XRP, SOL, the five largest non-stablecoins on CoinGecko that day, quoted in USDT; Tether (#3) and USDC (#6) skipped because a coin pegged to the quote currency has no return to predict. Shares: NVDA, AAPL, MSFT and ASML's Nasdaq listing, so the four shared one calendar and one currency.

Why: re-selecting "the top 5" every day would smuggle survivorship and look-ahead into the results; the list changes only between evaluation windows.

Finding: SOL (listed 2020-08-11) bounded the joint crypto history at about six years.

### M1 data layer built

*Page: [markets](markets.md)*

Built: `src/MarketData.py` (store, sources, incremental update, checks), `FetchMarkets.py` (`--init`, `--update`, `--check`, `--status`, `--set-source`) and the self-check `python3 -m src.MarketData`. `install_universe` adds what is missing and never changes a row. The fetch takes no `process.lock` and was not yet scheduled; the daily fetch arrived with the market predictor in M2.

Evidence: the first Binance fetch stored 15,224 crypto bars in twelve seconds with no gaps (paginated by 1,000, the running day's candle skipped until it closes).

### Source probe: Stooq dead, Yahoo 429, Nasdaq works

*Page: [markets](markets.md)*

Finding: Stooq's CSV URL served a JavaScript browser check whatever the client claimed to be. Yahoo answered 429 to everything from this address, including the cookie endpoint; its client was implemented and fixture-tested but kept as an alternate. Nasdaq's keyless endpoint (`api.nasdaq.com/api/quote/<symbol>/historical` with browser-like headers) worked: ten years in one answer, 2,513 rows for NVDA and ASML.

Decided: Nasdaq became the shares source; Alpha Vantage (25 requests/day) and Twelve Data were noted as keyed alternatives.

### Split-adjusted history refetched whole

*Page: [markets](markets.md)*

Finding: Nasdaq and Yahoo serve the whole history adjusted to the share basis of the day of the call (NVDA's 2024 bars at a tenth of the traded price, AAPL's 2020 bars at a quarter). A three-day incremental window would have spliced two price scales at the next split and produced a fake return of minus the log of the split ratio, with no check firing.

Fixed: for adjusted sources every update refetches the full answer and rewrites the history in place; a uniform rescale of overlapping bars is logged as `HISTORY RESCALED`; bars older than the source's window are put on the new basis with the same factor.

Why: a uniform factor moves no log return and no bin; dropping the older bars instead would have re-anchored the expanding window and re-cut a fifth of the shares' history (found in a second review).

Built: `--check` counts jumps beyond 35% close to close (a 2:1 seam is 69%); `--update --refetch-all` is the manual repair.

### Review hardening of the data layer

*Page: [markets](markets.md)*

Fixed: the update loop survives any exception per instrument (a truncated response, an odd payload, the 29 February on which a ten-year window has no date). A source change clears an instrument's bars because price bases differ. A source symbol with an exchange suffix is refused for a USD instrument (ASML.AS is the EUR listing, not the same instrument).

Fixed: an instrument added in a later window carries its own date, and an entry whose slot is held by another symbol is reported as a conflict instead of silently ignored.

### Store gitignored, text exports tracked

*Page: [markets](markets.md)*

Decided: `data/markets.sqlite3` is gitignored, revising the first decision to track it.

Why: the daily commit is `git add -A`, and a binary file that changes every day would add about a megabyte a day for data one call re-fetches. The tracked record is text exports per market and year.

### M2 built: the game, settlement and pages

*Page: [markets](markets.md)*

Built: `src/MarketGame.py` (ten quantile bins on an expanding window of at least 250 returns, yearly game files in the lottery shape, `game_days` with each day's edges), `crypto` and `shares` registered as positional games, `src/MarketSettle.py` (exact, adjacent and direction hits, fixed-rule P&L with a 0.1% fee, results CSVs, the page JSON), the Crypto and Shares pages, and `MarketsDaily.py` as the hand-run refresh under `process.lock`.

Decided: three market rules: a hit is the bin in the right slot, never a set intersection; the RL ticket row is skipped; where the positional machinery asks for a profit it gets the fixed rule in bin space (`Helpers.market_ticket_profit`).

Built: a new market back-builds 31 game days, with deep-learning rows on the newest day only, because a deep-learning child costs ten to twenty minutes a day.

Fixed: found in review: the weekly statistical tuner now names market games to the Backtester so tickets stay in drawn order and skips the lottery fetch for them; the History page's portfolio control scores a market ticket slot by slot; the fetcher refuses an empty game name, because the all-games endpoint would have appended every lottery's draws to the market files.

### Market pages as their own sections

*Page: [markets](markets.md)*

Decided: two sections, Crypto and Shares, in the top navigation next to the lottery predictor, instead of a Markets tab on the History page. Each shows per instrument the actual price course with the predicted course and band, the daily accuracy and the paper profit per model.

### M3 built: GARCH and Regime HMM rows

*Page: [markets](markets.md)*

Built: `<market>-returns.tsv` next to the game files (not `.csv`, because every `.csv` in a game folder is read as draws) and `src/MarketModels.py`: `GARCH Model` (GARCH(1,1) per instrument, constant mean, Student-t innovations, newest 2,000 returns) and `Regime HMM Model` with the ablations `Regime HMM ZeroMean Model` and `Regime HMM Single Model`; persistent regime templates by 2-Wasserstein distance; the regime reading file.

Decided: EM written out in numpy and scipy instead of `hmmlearn` and `arch`.

Why: neither package is on the pipeline interpreter, and hmmlearn's defaults (diagonal covariances, ten iterations) would have changed the model silently.

Evidence: GARCH fits in about a second a day and finds persistence 0.97-0.999 and 3-6 degrees of freedom; the regime count selected two for crypto and three for shares on the 30 Sept histories.

### Proper score judges the market rows first

*Page: [markets](markets.md)*

Decided: the Regime HMM row is judged by the mean log-score of its bin probabilities with a 95% paired bootstrap interval against GARCH before hits or paper P&L are read (`MarketRows.py`, Sundays, 250 days).

Why: regimes add information beyond GARCH only if regime-conditional means differ; tuning or judging on the fixed rule's P&L would turn the row back into an allocator.

Evidence: first hand run over the newest 40 crypto and 60 share days, not a result: GARCH on the uniform forecast on both markets (-2.298 against -2.303), the HMM rows a little under it, the full and Single rows worse on crypto; the lottery rows' near-one-hot scores punished hard (Markov -5.4, MarkovMonteCarlo -3.4) while their hit rates looked no worse.

### Market controls and the lockbox

*Page: [markets](markets.md)*

Built: the synthetic control of a market is a geometric random walk (independent Gaussian log returns with each instrument's own mean and standard deviation, cut by the game's causal quantile rule); the shuffled control reorders the game days. `NullControls.py`, `RandomnessDiscrimination.py` and `IrrelevantFeatureControl.py` took `crypto,shares` in the Sunday plan.

Decided: the lockbox period (Dec 2025 - Feb 2026) lies inside both market histories; the trainer, the quantum tuner, the feature control and the market rows report withhold its days.

### BLAS oversubscription in backtests

*Page: [markets](markets.md)*

Finding: the Backtester's forked workers each ran numpy with every core's worth of OpenBLAS threads; fifteen of them fitting the HMM put the load average at 230.

Fixed: the report pins one BLAS thread per process before numpy loads, and the Backtester's workers limit their thread pools to one.

### Regime paper read as a design, not a result

*Page: [markets](markets.md)*

Finding: arXiv 2603.04441 is a single-author preprint (no venue, code or data statement) with no numeric hyperparameters, tickers, test dates or significance test. Its Sharpe 2.18 against 1.59 and drawdown -5.43% against -14.62% are gross of costs over about 680 trading days (roughly June 2023 to February 2026), a window without 2008, 2020 or 2022 and coinciding with a gold rally; its own figure shows the S&P 500 falling about 19% where its table prints -14.62%.

Evidence: with 2.7 years of data the standard error of an annualised Sharpe is about 0.6, so the reported gap is about one standard error.

Decided: the regime machinery is imported as a design and validated against GARCH and the controls; the paper's mean-variance allocator is not built.

### Sentiment feeds: indicators only, not built

*Page: [markets](markets.md)*

Decided: news and social-sentiment feeds (CryptoPanic, LunarCrush, QuantMuse) can only be an indicator, never a predictor: a sentiment score could enter as one more meta-learner feature column and would be held to the irrelevant-feature control. Nothing was built.

### Roadmap item 9: day-file growth measured, raw arrays kept

*Page: [roadmap](roadmap.md)*

Evidence (production, 30 Sep 2026): day files 10 MB in 1,029 files, 1.2 MB added in June and 4.8 MB in September, about 10 MB a month with the markets; training CSVs 0.9 MB, a few KB a year per game; git pack 438 MB, about 1 MB a month from the day files, the rest legacy Keras blobs of 65-70 MB each from before model artifacts were ignored.

Finding: the fourfold growth since June came from the rows per day (7 then, about 30), not from the days. Three quarters of a day file were the two raw probability arrays, which nothing reads after the predictor copies them forward.

Decided: the raw arrays stay for now; dropping them would have shrunk every new day file about four times without a migration and remains the cheap first step.

Decided: the migration to SQLite waits for a trigger that is not size (10 MB a month is 120 MB a year on disk and about 12 MB a year in git): queries the files cannot give cheaply, or the report's scan over every file (about 3,600 files per year of nine games then) costing seconds in the daily run.

## 28 September 2026

### Irrelevant-feature control joined the Sunday plan

*Page: [operations](operations.md)*

Built: `IrrelevantFeatureControl.py` was added to the Sunday plan, between randomness discrimination and the null controls.

### Meta-learner calibration metrics and library versions

*Page: [tuning](tuning.md)*

Built: the trainer reports per-number calibration and ranking metrics (`src/Calibration.py`: Brier and log-loss against base rate, AUC, ten reliability bins with ECE, top-`draw_size` hits per day against chance; per-slot top-1 accuracy and argmax-ticket profit for positional games) instead of a printed accuracy line.

Built: every artifact stores `versions` (Python, numpy, scikit-learn, joblib); the same in `meta_learner_metrics.json`.

Why: a pickled model is only as loadable as the library that wrote it, and a metric that moved between two weeks reads differently when the library moved too. The metrics file is the durable record of the comparison between the meta-learner variants.

### Phase Q1 implemented and its gaps closed

*Page: [research](research.md)*

Built: `QuantumMetaLearner Model` (quantum-kernel SVC), `QuantumVQC Model` (variational classifier) and the classical control `ClassicalSVM Model` (same pipeline, RBF kernel, own tuned C and sample cap) as tracked rows, trained by `TrainMetaLearner.py` from the one-pass backtest table and tuned by `HyperoptQuantum.py`. One module, `src/QuantumModels.py`, holds a pure-numpy batched statevector simulator, the feature map and the classifiers; the proposed package layout was not built.

Decided: the per-game flags default on and act as off-switches. Why: the opt-in was proposed when heavy simulation was assumed; the 4-qubit numpy implementation trains in seconds to about a minute per game.

Decided: the control has its own study per row, tuned by the same objective on the same day split. Why: a C chosen for the fidelity kernel is not the right C for an RBF, and an untuned control would be handicapped.

### Encoding-scale search widened to 0.1-2.0

*Page: [research](research.md)*

Finding: `encodingScale` behaves as the kernel bandwidth and matters more than any other knob. Hand tests favoured about 0.3-0.5 over 1.0, and the first tuned values per game sat at 0.5-1.0, the bottom of the 0.5-2.0 grid.

Decided: search 0.1-2.0 in steps of 0.05. Every old grid value lies on the new grid, and Optuna continued the persisted studies.

### VQC readout moved to the last qubit

*Page: [research](research.md)*

Fixed: the readout was qubit 0, the entangling ring's final target; Z on a CNOT target pulls back to an observable without that qubit, so the last layer's two rotations on qubit 0 could never change the output, and a finite-difference check passed them as 0 = 0. The readout is now the last qubit, the ring's final control, whose Z pulls back to all four qubits.

Built: the self-check compares every RZ and RY angle and both readout parameters with central finite differences (agreement to 1e-10 over 18 parameters, shown to catch a 1% error in one) and asserts every angle has a live gradient.

Evidence: served artifacts carry their own readout sign vector, so earlier VQC rows kept serving as trained until the next weekly retrain.

### Pick3 quantum rows through the positional formulation

*Page: [research](research.md)*

Built: pick3 was first excluded because a per-number ranking does not express digit order; with the positional meta-learner it got quantum rows, one classifier per digit position. `HyperoptQuantum.py` tuned them on the real pick3 payout of the argmax ticket (mean `pick3_ticket_profit` per held-out day, per-position top-1 accuracy as tie-breaker).

### Lockbox declared in lockbox.json

*Page: [research](research.md)*

Built: a hand-written `lockbox.json` at the repository root (dates inclusive, no file meaning no lockbox), honoured by `TrainMetaLearner.py` and `HyperoptQuantum.py`, which drop the period from their tables before splitting and fitting, and evaluated once with `TrainMetaLearner.py --lockbox-report`.

Why: repeatedly inspecting the same holdout and changing the model afterwards turns the holdout into development data.

### TuningControls: the selection effect of tuning

*Page: [research](research.md)*

Built: `TuningControls.py` runs the statistical tuner's own objectives for `-t` trials on a control history and compares the best trial with the untuned defaults, next to the gate's real-history gain. `-t` and `-d` default to the weekly tuner's values; the gate record gained `run_trials`, `study_trials` and `window_days`, and a record made with other values gets no verdict.

Why: the maximum of n trials grows with n, so only like-with-like comparisons are read. The band is a floor: the weekly study is warm, a control study starts cold.

### Irrelevant-feature control built

*Page: [research](research.md)*

Built: `IrrelevantFeatureControl.py` and `src/FeatureControl.py`: every served meta-learner variant refitted on the cached table (lockbox days removed) with three noise columns, three repeats, the trainer's 80/20 day split and permutation importance on both sides. It runs in the Sunday plan after Q2; records in `data/controls/features/<game>.json`.

Fixed (in review): importance is the drop in |AUC - 0.5|, not the signed AUC drop. A Platt-scaled SVM fitted on noise often has probabilities inverted against the labels, where a signed drop read a column the model leans on as ignored.

Evidence: about an hour for the seven games, almost all of it the four VQC refits per game (a keno-sized VQC fit at 120 epochs about 160 s). With three noise columns among eleven, a model that cannot tell noise from signal lands near a 27% share; with eight real columns against a two-sigma band one clears it by chance about one week in five.

### Q0 completed: tuning on a control history, calibration metrics

*Page: [roadmap](roadmap.md)*

Built: the hyperopt on a control history (`TuningControls.py`), which measures the selection effect of tuning on top of the selection over rows; per-number calibration and ranking metrics persisted by `fit_meta_model` (`src/Calibration.py`, `meta_learner_metrics.json`).

Finding: still open in Q0: the dead `num_lstm_layers` setting (stored and written to metadata, never used by `create_model`).

### Q1 gaps closed

*Page: [roadmap](roadmap.md)*

Built: the classical-kernel (RBF) SVM comparison as its own tracked row, `ClassicalSVM Model` (`RbfKernelClassifier` in `src/QuantumModels.py`: the quantum-kernel row with only the kernel swapped), trained by `TrainMetaLearner.py` and tuned by `HyperoptQuantum.py`; `data/models/<game>/meta_learner_metrics.json` as the durable record of every variant's held-out metrics; package versions in every artifact under `versions`, next to `trained_at` and `params`; a finite-difference check of the parameter-shift gradient in `python3 -m src.QuantumModels`; the `encodingScale` grid 0.1-2.0 in steps of 0.05.

Evidence: the `runHyperopt.sh` pass of 26 Sep 2026 delivered tuned quantum keys for pick3 and Joker+.

Finding: what the phase leaves is the comparison itself: whether the quantum kernel ever separates from its classical control, and either from the null band, is a question for the ranking over months, not for more code.

### Negative-control suite complete; state of Q3 and Q4

*Page: [roadmap](roadmap.md)*

Built: the irrelevant-feature control (`IrrelevantFeatureControl.py`): shuffled-copy noise columns appended to the meta-learner table must not gain stable importance, and the base models are held to the noise columns' held-out band. With the shuffled and synthetic histories and the lockbox the suite was complete.

Finding: Q3 existed only in the weak sense (the Q1 rows emit a per-number quantum-assisted probability from base-model scores); Q4 was missing entirely and has the lowest value for the research question (a cost study, not a predictability study).

Decided: order Q0 and controls, then the Q1 gaps, Q2, Q3, with Q4 only if the earlier phases leave a reason; a fuller layout (`src/quantum/`, `experiments/quantum/`) only when Q2-Q4 code arrives.

## 27 September 2026

### Sunday control plan added to the schedule

*Page: [operations](operations.md)*

Built: `RandomnessDiscrimination.py` and `NullControls.py` became a second weekly plan, started when Sunday's predictor finished, with the same catch-up net as the chain (23:00 of its day until 09:00 two days later).

Why: the controls were measurements of the rows, never inputs to them, and hours long; at the tail of the Saturday chain they would have pushed Sunday's predictions behind them.

### Forward record from a design freeze date

*Page: [scoring](scoring.md)*

Built: a `since.json` at the repository root (`2026-09-27`, "September 2026 design freeze", `src/Since.py`) made the report add, per game, the same ranking over the days on or after that date, shown next to the all-history best row.

Why: the date is a filter that turns the record after a freeze into that design's own out-of-sample track record, the forward counterpart of the lockbox; nothing is rebuilt.

Decided: no chance band was applied to the since column, because the best of thirty rows over a handful of draws clears a band measured over long histories by luck alone.

### Controls run weekly for every game

*Page: [research](research.md)*

Built: both control scripts joined the scheduler's Sunday plan for every game.

### First lockbox declared

*Page: [roadmap](roadmap.md)*

Decided: the first lockbox was declared on 27 Sep 2026 in `lockbox.json`: 1 Dec 2025 to 28 Feb 2026 (`src/Lockbox.py`). `TrainMetaLearner.py`'s 80/20 and `HyperoptQuantum.py`'s 75/25 splits never see it.

## 26 September 2026

### First gated production run reverted three rows

*Page: [tuning](tuning.md)*

Finding: the first production run under the gate (26 Sept 2026) found three rows whose served parameters scored below the code defaults, e.g. lotto LaplaceMonteCarlo at 0.77 against the default's 0.82; the defaults were written back.

## 21 September 2026

### Null controls built: synthetic and shuffled histories

*Page: [research](research.md)*

Built: `src/ControlHistories.py` (synthetic and shuffled histories on the real calendar), `NullControls.py` (Q0) and `RandomnessDiscrimination.py` (Q2). The admin pages read their records through `controls.js`: the Best model per game card got a Q0 band column (mean over control histories of the best row plus two standard deviations, over the higher of the two controls) and greyed every hits-ranked row at or under it; for the payout games the band was shown as information only. The Randomness watch carried Q2's verdict next to its tripwires.

Why: the best row of a ranking is a maximum over rows and sits above the mean even when every row is worthless.

### First Q0 measurement on lotto

*Page: [research](research.md)*

Evidence: lotto, 120 days x 3 seeds, eight rows on untuned defaults, hits per draw. Synthetic: average row 0.781, best row 0.883, selection gain +0.102, best-to-worst spread 0.219, random baseline 0.794. Shuffled: average 0.821, best 0.964, gain +0.143, spread 0.289, baseline 0.794. Chance is 6x6/45 = 0.8.

Finding: a different row won almost every run (PoissonMonteCarlo, XGBoost, Markov on synthetic; PoissonMonteCarlo and PoissonMarkov on shuffled). The shuffled control was the harder bar, because it keeps the real number frequencies. On lotto a tracked row was not interesting below about 0.96 hits per draw.

Finding: the same winning row reached exactly the same score in two control histories (PoissonMonteCarlo 0.9417 on synthetic seed 1 and shuffled seed 1; PoissonMarkov 0.975 on two shuffled seeds). Not a reused dataset: hits over 120 days are multiples of 1/120 and every other row differed.

### First Q2 result: nothing separates real from fair

*Page: [research](research.md)*

Evidence: 1000 draws, 5 repetitions, best AUC of the suite per repetition. Lotto: real 0.588 ± 0.049, null band 0.565 ± 0.076, shuffled 0.582. Euromillions: 0.551 ± 0.050, 0.544 ± 0.030, 0.536. Pick3: 0.495 ± 0.043, 0.540 ± 0.045, 0.534. All within the null band.

Finding: no classifier, classical or quantum, separated real draws from correctly simulated fair ones beyond what the same pipeline scores on two histories identical in law.

### Shuffled control cut before the shuffle

*Page: [research](research.md)*

Fixed: permuting the whole 1995-2026 history and then taking its newest 600 draws sampled every rule era. Per-number counts had a standard deviation of 13.5 against the real 9.1 (chi-square p < 0.0001), so the control no longer shared the marginals it exists to preserve and "separated" better than the real history. The window is now cut before the shuffle and the two match exactly.

Finding: the control's own null band caught the defect.

### Q2 verdict compares best-of-suite with best-of-suite

*Page: [research](research.md)*

Fixed: comparing the best AUC over seven classifiers and five repetitions against the mean of the null comparison's individual runs made euromillions read as evidence (0.637 against a 0.598 threshold). Held against the null's own best-of-seven it is 0.551 ± 0.050 against 0.544 ± 0.030: nothing.

Built: `--reanalyze` recomputes the verdicts of saved records under the current rule without rerunning.

### Foundation models measured on CPU

*Page: [roadmap](roadmap.md)*

Evidence (Chronos-2 / TimesFM-3): first run downloads about 460 MB / about 1.2 GB in 6 min; warm load 1.4 s / 4.2 s; six lotto positions 0.16 s / 0.41 s; a further point in time 0.28 s / 0.40 s; resident memory 0.84 GB / 2.84 GB; quantile grid 21 levels (0.01-0.99) / 9 (the deciles).

Finding: both are cheap enough to run for every game every day without the GPU or the deep-learning budget.

### Foundation rows on the live day only

*Page: [roadmap](roadmap.md)*

Decided: the foundation rows run on the live day only (`skipRows == 0`), never inside a history rebuild.

Why: the rebuild runs the statistical step in a pool of up to eight spawned children, and a worker in each would add 0.8-2.8 GB apiece on top of their TensorFlow imports; and backfilling a new row into days that are already scored gives it a track record it never earned out of sample, the rule the tuned `SubsetEnsemble Model` follows too.

### OrderStatistics Baseline as the foundation rows' yardstick

*Page: [roadmap](roadmap.md)*

Built: the `OrderStatistics Baseline` row (`Baselines.ColumnFrequencyBaseline`), the historical mode of each position from the same data through the same interface.

Why: for the sorted set games the per-position series are order statistics (position 1 of lotto is the minimum of six draws from 1-45), so a forecast of "about 5" is arithmetic; the row is meaningless against chance and a per-position model is interesting only once it beats this baseline.

Evidence: Chronos's six lotto positions peaked at 2, 11, 18, 25, 35, 44.

### Chronos on pick3 tracks recency noise

*Page: [roadmap](roadmap.md)*

Finding: on the positional games the forecasts are near-uniform and repeated-digit tickets (`[9, 9, 9]` on 20 Sep, `[8, 8, 8]` on 21 Sep 2026) are that result, not a fault; the model acts as a recency-weighted frequency estimator, in the same class as the baseline. The scores are the output worth reading, not the ticket.

Evidence: Chronos gave digit 8 a mass of 0.121 / 0.126 / 0.128 across the three positions against 0.083-0.089 for the least likely digit (ratio 1.4-1.5). Over the last 100 draws 8 was the most frequent digit in positions 1 and 3 (0.160 and 0.180) and 9 in position 2; a digit's frequency over 100 draws has a standard deviation of 0.030, so these are 2.0 and 2.7 deviations, the largest of 30 digit-by-position cells where the expected largest is about 2.2. Over the full 512 draws the digit sat at 0.104 / 0.100 / 0.113, inside one deviation of uniform (0.013).

Evidence: the quantile grid is not involved: 21 native levels, 199 levels inside the trained range and 199 reaching past it gave the same distribution to three decimals.

Decided: the worker asks inside the trained span only.

### Quantile tails anchored at the game's support

*Page: [roadmap](roadmap.md)*

Finding: TimesFM-3's first lotto forecast returned 1 as the most likely value for four of six positions, each at exactly 0.100. The shared step that turns a quantile curve into label probabilities read the curve as a CDF clamped at its outermost levels and folded the outside mass onto the first and last label; with Chronos-2's 0.01/0.99 levels that moved 0.005 per side and changed no argmax, with TimesFM-3's deciles it put a flat 0.100 on the lowest label of every series.

Fixed: the curve is anchored at the bounds of the game's own support (the points (first label - 0.5, 0) and (last label + 0.5, 1) are added before interpolating), so each tail is spread uniformly over the stretch it occupies (`src/foundation/quantile_labels.py`).

Evidence: Chronos's published numbers were unchanged to three decimals. The second implementation found the defect within an hour.

### The two foundation models side by side

*Page: [roadmap](roadmap.md)*

Evidence: lotto, the same 512 draws: Chronos 2, 11, 18, 25, 35, 44 against TimesFM 2, 11, 21, 25, 40, 44: agreement on four of six positions and the same shape.

Evidence: pick3, the same 512 draws: Chronos `[8, 8, 8]`, TimesFM `[9, 3, 9]`; on the first day both rows were served (21 Sep 2026) Chronos `[8, 1, 8]` against TimesFM `[9, 9, 9]`. TimesFM's spread was wider (ratios 1.5-2.6 against 1.4-1.5), which on nine knots is as much resolution as signal.

Finding: two models from different labs giving different most likely digits from the same history says there is nothing there to find.

### Foundation models as meta-learner columns

*Page: [roadmap](roadmap.md)*

Built: `Chronos Model` became a base-model column (`BASE_MODEL_NAMES` in `src/ModelFactory.py`), appended last so older artifacts, which store their own `feature_names`, keep working. `TimesFM Model` is available behind `useTimesFmFeature`.

Decided: the TimesFM column is off by default.

Why: 2.8 GB for a second reading of the same kind of signal is a choice, not a default.

Built: the score-table cache is invalidated by the base-model set, not only by their parameters, so `TrainMetaLearner.py` recollects automatically.

### Foundation forecasts precomputed before the Backtester forks

*Page: [roadmap](roadmap.md)*

Built: `ModelFactory.prepare_foundation_scores` forecasts every day the backtest will ask for in the parent with one warm worker, closes it and switches the model to cache-only; forked children read the inherited dictionary and a day nobody precomputed returns "no signal". `python3 src/FoundationModel.py <game>` is the self-check (precompute, fork four children, assert zero workers started).

Why: a forked child must neither talk to the parent's worker (two conversations in one pipe) nor start its own (fifteen children x 0.8-2.8 GB).

Evidence: 0.51 s per lotto day (forecast plus data reload), about 2.5 minutes for a 300-day window. On pick3 the collected table carried `Chronos Model_position_scores`, the trained artifact's features were Markov, MarkovMonteCarlo, PoissonMonteCarlo, LaplaceMonteCarlo, XGBoost, Chronos, and a process scan saw a peak of one forecasting worker.

Fixed: the first attempt recognised the models with `isinstance`, which returned False because `src/` modules are imported both as `src.X` and as `X` (two copies of the class); the precompute did nothing and the pool spawned a worker per child. Identity became an attribute (`IS_FOUNDATION_MODEL`), and `_start_worker` refuses when the model is used in a process other than the one that built it.

### Adversarial review of the foundation work: three defects that mattered

*Page: [roadmap](roadmap.md)*

Fixed: the column was trained on real scores and served as 0.0: `Predictor.py` fills 0.0 for any feature it has no instance for, and the foundation models were missing from that instance map. Evidence: pick3 served with and without the instance gave `MetaLearnerV2` `[1, 5, 8]` against `[0, 7, 8]`.

Fixed: one unusable day killed a whole game: `precompute()` had no per-key guard; with the default 300-day window eurodreams (exactly 300 stored draws) asked for a slice with no rows and `Helpers.load_data` raised. Now a per-key `except` and a `try/finally` that releases the worker (a mid-run failure had left 0.8-2.8 GB resident).

Fixed: `_unavailable` was set on any worker failure and never cleared, while `Predictor` closes the worker between games, so one transient timeout on the first game dropped the row for every later game. `close()` re-arms; permanent refusals (missing libraries, a model carried across a fork, cache-only) are re-checked on every call.

Finding: the convention "absent feature means 0.0" turns a wiring mistake into a wrong answer instead of a crash.

### Adversarial review of the foundation work: three smaller defects

*Page: [roadmap](roadmap.md)*

Fixed: the Chronos worker fed possibly-crossing quantiles into an interpolation that needs a monotone curve (now sorted after the clip).

Fixed: `foundationContext` was not part of the score-table cache fingerprint, so a changed context could have been trained against a stale table.

Fixed: exact argmax ties, common because a piecewise-linear CDF is flat across stretches of labels, always resolved to the lowest label, a measured bias of about 0.8 of a label; ties now break toward the slot's own mean.

Evidence: none of the documented numbers changed; they were re-measured after every fix.

### Foundation flags and their defaults

*Page: [roadmap](roadmap.md)*

Decided: in `bestParams_<game>.json` the three rows are on by default (`useChronos`, `useTimesFm`, `useOrderStatisticsBaseline`), the Chronos meta-learner column is on (`useChronosFeature`), the TimesFM one off (`useTimesFmFeature`), and `foundationContext` defaults to 512 past draws. On a machine without the libraries every one degrades to nothing with a single log line.

Decided: nothing stayed open in the item; fine-tuning and a third checkpoint were left for later, if the tracked results justify them.

### Q0: null controls built

*Page: [roadmap](roadmap.md)*

Built: `NullControls.py` on `src/ControlHistories.py` runs the production evaluation path against two control histories per game (shuffled and synthetic fair) and reports what the best row scores when there is provably nothing to find.

Built earlier: `TCN.py` became tunable and runs as `TCN Base Model`; the daily run had `--ai` on again, so TCN accrued a production record.

From 27 Sep 2026 the control ran weekly for every game and fed the null band of the ranking.

### Q2: randomness discrimination built, first answer no

*Page: [roadmap](roadmap.md)*

Built: `RandomnessDiscrimination.py` turns windows of draws into multi-hot counts (per slot for the positional games) and trains logistic, RBF SVM, random forest, gradient boosting, small NN, quantum kernel and VQC classifiers on a chronological holdout, repeated over independent synthetic histories.

Finding: across lotto, pick3 and euromillions (1000 draws, 10-draw windows, 5 repetitions) the real-versus-synthetic separation sat inside the band the same pipeline produces on two histories identical in law.

From 27 Sep 2026 it ran weekly for every game with its verdict as a column on the Randomness watch card. Left open: other window lengths and feature families, and a rule-change map.

## 20 September 2026

### Council API became a supervised service

*Page: [operations](operations.md)*

Built: `services.js` started the Council API (`src/llmCouncil/api.py`) with the server, restarted it with a growing backoff, marked it failed after five exits within ten minutes, and adopted an instance already answering on its port.

Why: before 20 Sep 2026 the Council API had to be started by hand, so the Council was a dead link whenever nobody had.

Decided: a service stayed an ordinary child of the server (cheap to restart, must simply be up), while pipeline jobs were launched detached because a twelve-hour tuning run must survive a deploy.

### Roadmap item 5: foundation models, zero-shot

*Page: [roadmap](roadmap.md)*

Decided: zero-shot, no fine-tuning; fine-tuning stays ruled out until a zero-shot row proves interesting.

Built (20-21 Sep 2026): every drawn position is one univariate series; `amazon/chronos-2` is asked for the next value's distribution, which serves as the `Chronos Model` row (`src/ChronosModel.py`) with `run()`, `score_positions()` and `score_numbers()`. TimesFM-3 (`google/timesfm-3.0-pytorch`, `src/TimesFmModel.py`) followed through the identical protocol; both are thin subclasses of `src/FoundationModel.py`, which owns the worker, the cache and the ticket construction.

Why two: they are different models from different labs trained on different corpora, so what they agree on is a property of the data and what only one sees is a property of that model.

### Foundation libraries run outside the pipeline's interpreter

*Page: [roadmap](roadmap.md)*

Decided: the libraries live in their own directory (`/root/.foundation-libs`, `FOUNDATION_LIBS` to move it) that only a small worker process puts on its path (JSON lines over stdin/stdout).

Why: the foundation stack pulls numpy 2.x while the pipeline runs TensorFlow 2.16 on numpy 1.26; installing it into the system interpreter would have upgraded numpy underneath the daily predictor. A plain target directory, not a virtualenv, because `python3-venv` was not present on the machine.

Built: the worker is kept warm for the life of the model object and is launched with `PR_SET_PDEATHSIG` and an `atexit` close, because the history rebuild runs the statistical step in a pool of spawned processes and eight orphaned 0.4 GB workers would exhaust a 16 GB machine. Missing libraries, a crash or a timeout disable the row with one log line.

Evidence: install about 630 MB plus a 457 MB model cache; installing TimesFM pulled the CUDA build's dependencies, and removing the `nvidia` and `triton` folders reclaimed 2.8 GB without affecting forecasting. `timesfm` needs `--no-deps` only because its torch extra would re-resolve the CPU wheel.

### Scheduler built in two halves, shipped in dry mode

*Page: [roadmap](roadmap.md)*

Built (20-21 Sep 2026): first the services (`services.js`): the Council API supervised and shown on the admin Jobs page. Then the schedule (`jobs.js`): seven jobs (the daily predictor and the six of the weekly chain), a single-worker FIFO queue that waits on `process.lock`, the chain trigger, catch-up, `setsid --fork` launches verified to come up with PPID 1, adoption of a job that outlived a server restart, rotation of `log/*.log` at 10 MB, and a run record with the git SHA. `test/jobs.test.js` covers the schedule arithmetic and fails if `jobs.js` and the shell scripts disagree.

Evidence: a restarted server picked its running job back up and still recorded the right duration and exit code.

Decided: it shipped in `dry` mode (works out what it would start, records it, starts nothing), which made it safe next to the live crontab. Cutover: watch a weekend, remove the crontab entries, set `SCHEDULER=on` and restart; fallback `touch config/scheduler.disabled` and re-add the crontab lines.

Left open: a per-game predictor job and a mail to the administrator when a job exits non-zero.

## 19 September 2026

### The schedule moved from crontab into the server

*Page: [operations](operations.md)*

Decided: the daily predictor, the weekly tuning chain and the weekly controls became the server's own work (`jobs.js`, roadmap item 8), with `runPredictor.sh` and `runHyperopt.sh` kept as the hand-run path and `test/jobs.test.js` failing when the two drift apart.

Why: under cron the chain started at 14:00 whatever the predictor did, and a job that found the pipeline busy exited, which could skip a week of tuning in silence.

Evidence: on 2026-09-19 the chain sat idle for 3.5 hours between the predictor's end and 14:00.

Built: the chain started when Saturday's predictor finished; a busy pipeline queued the job (single worker, first in first out); a missed slot was caught up for 20 hours and the chain until Monday 09:00; jobs were launched with `setsid --fork` so a deploy did not end them; every run was recorded with reason, duration, exit code and git SHA.

Built: `SCHEDULER=dry` was made the default so the scheduler could be deployed next to a live crontab; the first ever start recorded the slots in its window as handled, because they belonged to the crontab era.

### Roadmap item 8: why cron was replaced

*Page: [roadmap](roadmap.md)*

Finding: on 19 Sep 2026 the predictor finished at 10:30 and the weekly chain sat idle until its 14:00 cron slot, then ran 16.5 h to 06:29, while `process.lock` already guaranteed one Python job at a time. A tuner that found the lock held simply exited, so a whole week's tuning could be skipped in silence.

Decided: the Node server owns the schedule (`jobs.js`, required by `server.js`, so the web process stays the one thing pm2 supervises): a single-worker queue mirrors the lock, the weekly chain is triggered by Saturday's predictor finishing, a run missed while the machine was down is caught up, and every job is launched through `setsid --fork`.

Evidence: pm2 runs with `treekill`, so a job spawned as an ordinary child of the server was killed 1.6 s into any `pm2 restart`; after `setsid --fork` the job's parent is PID 1, so a deploy, crash or restart cannot touch a running 12-hour tuning job.

### Scheduler decisions: queueing, crontab, access

*Page: [roadmap](roadmap.md)*

Decided: a Saturday chain that overruns into Sunday's predictor slot (09:00 then) queues that predictor behind it rather than pre-empting the tuner.

Decided: the three crontab entries are removed at cutover; a `config/scheduler.disabled` flag file falls back without a restart.

Decided: the UI keeps its public Tailscale funnel (onboarding a user is a link, not a VPN setup), so every trigger route is admin-gated, POST-only and CSRF-checked.

Decided: the Jobs page is admin-only, but a small "prediction running" indicator is visible to every user.

## 17 September 2026

### Time-boxed deep learning back in the daily run

*Page: [models](models.md)*

Built: every model's training run got a wall-clock budget (`--dl-model-seconds`, default 240 s, `src/TimeBudget.py`) that stops training the way an early stop does, and the per-day child a hard deadline (`--dl-timeout`, automatic by default) with a checkpoint after every model; rows finished before a kill are kept. The cheap research rows run first, then the LSTM, then TCN and the unified rows.

Why: this let the heavy rows back into the daily run; training quality may degrade at the budget but never the row.

Evidence: the first batch of a freshly compiled graph cost up to 50 s on the TCN in testing. The automatic deadline came to 37 min for seven models (42 min for eight after 25 Oct 2026), so up to about 4.9 h on a day every lottery game has a draw (typically 3-5 games, 2-3.5 h), multiplied by the days rebuilt after a gap.

Decided: `runHyperopt.sh` waits for the predictor's lock instead of skipping the week.

### Joker+ lost its deep-learning step to a KeyError

*Page: [models](models.md)*

Finding: the first `--ai` run lost Joker+'s whole deep-learning step to a `KeyError` on `batchSize`; Joker+ had never been DL-tuned.

Fixed: `LSTM Base Model` falls back to `LSTMModel`'s own defaults for any tuning key missing from `bestParams_<game>.json` and logs which keys are untuned.

### Roadmap item 3: login form with roles

*Page: [roadmap](roadmap.md)*

Decided: a login form with basic user management instead of HTTP Basic Auth.

Why: a login page with roles needs a session and a logout, which Basic cannot express.

Built: `auth.js`. The administrator is `WEB_USER` / `WEB_PASSWORD` from the environment (never on disk; unset = open access for local development); further accounts live in the gitignored `config/users.json` as salted `scrypt` hashes, managed by the administrator, no self-service reset. Two roles. Sessions are a signed `HttpOnly` `SameSite=Strict` cookie (secret from `WEB_SESSION_SECRET`, minimum 32 bytes, or generated once into `config/session.secret`), `Secure` over HTTPS, 24 h with renewal and a 7-day absolute limit, revoked by a password change or deletion; five failed sign-ins per user name and client address lock for 15 minutes; forms carry a CSRF token; user names cannot equal the administrator's.

No npm dependency was added (Node's `crypto` hashes and signs). HTTPS stayed the deployment's job (reverse proxy).

### Security review of the first login version

*Page: [roadmap](roadmap.md)*

Finding: an independent review of the first version found nine defects.

Fixed: a backslash open redirect in the post-login target; the lockout keying on the proxy's address; sessions surviving password changes; the CSRF token rotating on renewal; user-name enumeration through sign-in timing; no minimum secret length; an unbounded failure map; write access to the user page in open mode; a bare 401 on logout with an expired session.

### Optuna dashboard removed from the UI

*Page: [roadmap](roadmap.md)*

Decided: the Optuna dashboard was removed from the UI entirely (the navbar link and the `optuna-dashboard` process the server used to start) instead of being proxied behind the login; it is started by hand when needed.

### Roadmap item 7: a vote per position for pick3

*Page: [roadmap](roadmap.md)*

Decided: a weighted vote per slot: every model's digit in a slot casts a vote weighted like the ensemble (`modelScores`, min-max mapped to 1-2), the slot's digit is the argmax, ties go to the lowest digit, duplicates across slots are allowed, drawn order is kept, and the row carries a per-slot confidence.

Why: the row did not exist for the positional games (`addWeightedEnsemblePrediction` returned early) and removing the guard would have been wrong: the set-game path takes the three most voted distinct digits and sorts them, which destroys what pick3 pays on (straight, front pair and back pair are exact-order, the consolation depends on the last slot, and 28% of draws repeat a digit).

Built (phase 1): the per-slot hard vote on the models' final tickets (`Helpers.count_position_votes` / `build_positional_vote_predictions`, new code beside the untouched pooled counters, which Keno subset generation and tuning depend on), served as `WeightedEnsemble Model` over the rows that own a model (`POSITIONAL_VOTE_DEFAULT_ROWS`) and as `SubsetEnsemble Model` over the subset `HyperoptEnsemble.py`'s positional branch selects (exhaustive include flags, scored with `pick3_ticket_profit`, lower confidence bound plus a 0.01 x per-slot accuracy tie-breaker), with `positionConfidence` on the row and `positionFrequency` in the day JSON.

Decided: the tuner's own score is never written into `modelScores` (it would feed back into its own weights); it runs after `HyperoptBoost.py` and `HyperoptRLTicket.py`.

Decided: Joker+ is mechanically covered (six digit slots plus the sign as slot 7) but gated off behind `useJokerplusEnsemble`, because the player cannot choose the digits.

Decided: deferred: phase 2 (the soft vote, which needs a `positionScores` sidecar) and excluding the ensemble row from the RL row's pick3 vote features.

### Item 7 pitfalls: weights, double counting, pick3 scoring

*Page: [roadmap](roadmap.md)*

Finding: the `modelScores` weights were, until September 2026, a min-max over a 31-day profit per bet: lucky-strike dominated and sign-blind (for Joker+ three negative boosting scores mapped to weights 2.0 / 1.75 / 1.0, so XGBoost counted double). The capped lower-bound objective replaced it.

Finding: the three deep-learning research rows emitted identical tickets day after day and the four meta rows are aggregates of the base rows, so an all-rows vote double-counts.

Decided: the default include set is the rows that own a model (statistical and per-position boosting rows); meta and deep-learning rows are opt-in through the tuned selection.

Fixed: `server.js` scored a pick3 ticket first-match-wins without deducting the stake while the backtest used the cumulative four-bet model (an exact `[1,2,3]` was 500 in the UI and 676 in the backtest), so every pick3 row's shown profit disagreed with its tuning profit; the server now uses the same model as `Helpers.pick3_ticket_profit`. The scoring moved to `draws.js` on 22 Oct 2026.

## 16 September 2026

### Subset selections for five games; candidacy rule tightened

*Page: [roadmap](roadmap.md)*

Finding: the first production run (15 Sep 2026, exhaustive over 256-512 combinations per game) gave selections for all five games; the lotto and eurodreams selections included `LSTM Base Model`, a row the cron had stopped emitting in mid-August (`--ai` off), so those two served rows were skipped daily.

Decided: a candidate row must additionally be present on three of the last five scoreable days.

Why: so a selection can no longer include a row the pipeline has stopped emitting.

The deep-learning rows ran again in every scheduled run (`-a true`) from 16 Sep 2026, after which the two rows were served without a re-run.

## 14 September 2026

### No combination clears its shuffled control

*Page: [scoring](scoring.md)*

Finding: on 14 Sep 2026 no game's combination p-value was anywhere near small; the card showed the winner of a lottery among combinations, which is what the control was built to make visible.

### Roadmap item 2: portfolio of rows, ranked per draw

*Page: [roadmap](roadmap.md)*

Built: `Helpers._build_combination_report` (pairs, triples, greedy build-up, the `minDrawsForRanking` floor on shared draws, the number of combinations evaluated, a shuffled-history control with 100 shuffles and a p-value), shown as the "Best combination per game" card.

Decided: payout games rank portfolios by profit per draw (a set wins only if at least two rows are positive on the shared draws, and the card states whether it beats the best single row); hit-scored games by the best line held per draw, pairs and triples ranked within their size, no greedy build-up.

Why: the original plan ranked by summed payout over summed stakes, profit per bet; a portfolio's profit per bet is the stake-weighted mean of its members', so it could never beat the best member. On hits every extra line raises the best line, so a greedy build-up says nothing.

Finding: on the first reading of the real data no game's p-value was anywhere near small.

### Roadmap item 2: searched subset-ensemble row

*Page: [roadmap](roadmap.md)*

Built: `SubsetEnsemble Model`, selected by the new `HyperoptEnsemble.py` and served by `Predictor.addSubsetEnsemblePrediction`.

Decided: the vote recipe moved into `Helpers.build_vote_ensemble_predictions`.

Why: so the served row and the tuner cannot drift apart.

Pick3 joined on 17 Sep 2026 with item 7's per-slot vote (Joker+ behind `useJokerplusEnsemble`).

## 13 September 2026

### Roadmap item 1: Settings dropdown removed

*Page: [roadmap](roadmap.md)*

Built: the top-right Settings dropdown was removed, with the navbar status text, the `/playedModel` and `/playedNumbers` endpoints and the body-parsing middleware only they used, the `selectedModel`/`filterDataByModel` plumbing and the related CSS.

Why: the global model filter's option list was hardcoded to seven statistical models while the pipeline tracked about 25 rows per game, so the filter silently hid most rows; the "Keno played numbers" form was only echoed in the navbar and never used in a calculation.

Evidence: every page (home, History, the seven game pages, oldest and newest day page per game) was rendered with the previous and the new `server.js`; outside the removed navbar block the HTML was byte-identical and the two removed routes answered 404.

The `.nav-btn` style that survived this change went with the old pages on 26 Oct 2026.

## 1 September 2026

### Lower-bound, capped objective replaced raw profit

*Page: [tuning](tuning.md)*

Finding: a trial's value was its raw profit per bet over 31 days, decided by whether one top payout fell in the window. The keno LightGBM parameters served from 10 Sept scored 5.0 per bet: one draw (a 6/6 with its nested 5/5, +348 EUR) was everything, the other 60 bets lost 38 EUR. The pick3 XGBoost and CatBoost winners were single straights (+756 and +676 EUR); the pick3 statistical bests (21 and 42 per bet) were one and two straights.

Evidence: the per-day intermediate values of these trials were negative for most of the window and jumped on one day.

Decided: trials are scored by a lower confidence bound over the days (mean − std/√days, the estimator `HyperoptEnsemble.py` already used); keno bets capped at the lucky-strike threshold of 20 EUR net; positional games by slot hits with the capped-profit bound as a 0.001 tie-breaker; hits bound for games without a payout table.

Why: almost every pick3 window is a flat loss once a straight is capped, so profit alone would rank noise; the Backtester's own pick3 `_hits` column is a set count that scores a box like a straight, so `TuningScore` counts slot hits from the drawn-order ticket.

Finding: on the September data every keno and pick3 trial's bound was negative, as a fair ticket's would be.

### Default tuning window raised from 31 to 90 draws

*Page: [tuning](tuning.md)*

Decided: `-d` defaults to 90 draws for the statistical and boosting tuners (31 before).

Why: 31 days were decided by single payouts.

Finding: the boosting refit cadence scales with the window (five blocks at 29 days or more), so a boosting trial's cost, its 20-minute budget and the parallelism stayed unchanged; the run grew only by up to two reference backtests per strategy and game. Statistical trials and the keno precompute took roughly twice as long as at 31 days.

### Champion/challenger gate replaced writing the study's best

*Page: [tuning](tuning.md)*

Finding: every run wrote `study.best_params`, the best trial the study had ever seen, which under the old objective was the trial that caught the biggest payout (the keno 5.0 was never approached again: 2.13, 1.26).

Decided: this run's best completed trial is the challenger and replaces the served parameters only if it beats the re-scored incumbent and the untuned defaults by `--gate-margin`; the defaults are written back when they beat the incumbent; a trial without a finite score is never a challenger; a reference that cannot be scored cannot block.

Built: references scored through the strategy's own objective with an `optuna.trial.FixedTrial`, each in a forked child (`HyperoptRunner.run_isolated`); every decision recorded under `tuningGate`; `modelScores` carries the score of what is served, and an unscored row loses its old raw-profit entry.

Why: the coordinator must never run a fit itself before forking the next study's trials; a weight measured on another scale must not stay.

### Per-model keno subset flags were never read

*Page: [tuning](tuning.md)*

Finding: the per-model keno flags the tuners searched (`lightgbm_use_5`, `markov_use_6`, ...) were written to the file but never read at serve time; `Predictor.getKenoSubsetSizes` uses the global `use_<n>` flags for every row. A trial's value depended on ticket sizes that never reached production, and the gate would have compared unlike tickets.

Fixed: trials and references bet the served sizes (`served_keno_subsets`); the stale per-model keys are ignored.

### Keno subset precompute bounded at three trial budgets

*Page: [tuning](tuning.md)*

Decided: the keno subset-tuning precompute, the one step no trial budget reaches, is bounded at three trial budgets (an hour); the row is skipped for the run when it does not finish.

### MarkovMonteCarlo pick3 ticket made per-slot

*Page: [models](models.md)*

Fixed: `MarkovMonteCarlo Model`'s pick3 ticket was the sorted set of its top-voted unique digits (never a repeated digit, order meaningless), so every straight/pair payout it was tracked on was scored against a scrambled ticket. It became the per-slot mode of its simulated tickets in drawn order, the same tallies its `score_positions()` gives the positional meta-learner.

Finding: the row's tracked pick3 history before this fix is not comparable with what follows.

### Markov fallback range derived from the data

*Page: [models](models.md)*

Fixed: nothing ever called `setGameRange`, so an unseen-context fallback in `Markov` used the 1-80 default for every game and could emit an impossible digit. The range is now derived from the data the model is built on (pick3 0-9, lotto 1-45, a special-only pass its special range).

## 25 August 2026

### RL ticket rows stored without special columns

*Page: [models](models.md)*

Finding: every `RL Ticket Model` ticket since late August 2026 was stored mains-only in EuroMillions, EuroDreams and Vikinglotto, because the policy builds main numbers only. Those tickets cannot be completed afterwards, so they count on hits only and carry no profit on the prize tables.

## 22 August 2026

### Second out-of-memory kill; training moved to a child

*Page: [models](models.md)*

Finding: the Predictor was OOM-killed a second time, at 10.3 GB RSS.

Built: all deep-learning training runs in a one-shot spawned child process per day. An OOM in the child surfaces as a caught `BrokenProcessPool`: that day loses only its deep-learning rows (the half-built file is repaired on the next run), everything else still runs.

Why: recycling the process per day releases the memory that caused the kills.

## 20 August 2026

### Predictor killed by the memory limit

*Page: [models](models.md)*

Finding: the container's 16 GB memory cgroup OOM-killed the Predictor at 16.7 GB RSS: a single long-lived process accumulated TF/Keras allocations across about 24 model trainings and was killed without a trace.

## 15 August 2026

### Heavy deep-learning rows switched off in the daily run

*Page: [models](models.md)*

Decided: the heavy deep-learning rows (LSTM, TCN, Unified) were switched off in the daily run from mid-August 2026.

Why: their unbounded training blocked the prediction flow for hours.

Finding: the rows had no tracked history for about a month as a result.

## 1 August 2026

### A hung run held the pipeline lock for two days

*Page: [operations](operations.md)*

Finding: in August 2026 a run hung alive while holding `process.lock` for two days; a single log line was the only evidence, and no predictions were produced meanwhile.

Decided: a blocked queue has to stay noisy. Waiting on the lock was re-logged every half hour, and after six hours (the point where `runHyperopt.sh` used to give up) the wait was recorded as a blocked entry on the Jobs & services page.

### Ensemble candidates must be recently present

*Page: [tuning](tuning.md)*

Finding: the `LSTM Base Model` row stopped appearing in mid-August 2026, still covered 79 of 87 lotto days, got selected, and the served SubsetEnsemble row was then skipped daily for a missing member.

Decided: a candidate must also be present on at least three of the last five days, next to the 80% coverage of the window.

## 3 February 2026

### Markov row switched to per-column transitions

*Page: [tuning](tuning.md)*

Built: on 3 Feb 2026 ("updated markov to learn transition t+1") the `Markov Model` became one table per column, from the number in a column of the previous draw(s) to the number in the same column of the next draw.

Finding: the Backtester, the tuner and the meta-learners scored this definition from then on; `MarkovBayesian Model` and `MarkovBayesianEnhanched Model` kept the within-draw table.

### Label smoothing left the LSTM loss

*Page: [models](models.md)*

Finding: label smoothing left `LSTM Base Model`'s loss on 3 Feb 2026 (commit 49bcb2f5). From then on the row trained on plain categorical cross-entropy; the tuned `labelSmoothing` was stored but not applied.

## 1 August 2025

### Mean-minus-std repeat objective tried and reverted

*Page: [tuning](tuning.md)*

Built: a mean-minus-std variant over repeated evaluations per trial was added in August 2025 (the original `Hyperopt.py` had three repeats per trial).

Decided: reverted a week later.

Finding: repeats could not have helped; the Backtester is seeded per day, so a repeat returns the same number. The luck is in the draws of the window, not in the model's randomness.

## 13 February 2025

### Markov row learned a within-draw table

*Page: [tuning](tuning.md)*

Built: from 13 Feb 2025 the `Markov Model` learned one table over the whole game, from each number to the next number inside the same sorted draw (the shape of a ticket, not time).

Finding: the bonus and star columns rode along in that table as the last step of every draw.

## Undated

### Optuna dashboard no longer started by the server

*Page: [operations](operations.md)*

Decided: the server stopped starting and linking the Optuna dashboard; it is started by hand (`optuna-dashboard sqlite:///db.sqlite3`).

### Login and audit design reasons

*Page: [operations](operations.md)*

Decided: the administrator's credential lives only in the environment and is not editable from the web, since it is the one that can change everyone else's.

Decided: accounts, secrets and state live in the git-ignored `config/` folder because the daily predictor commits everything under `data/`; what an account has seen is kept apart from the password hashes.

Decided: a failure to write the audit trail is logged and swallowed: a trail that can break a sign-in is worse than one that misses a line. With only one of `WEB_USER` and `WEB_PASSWORD` set the server refuses to start rather than silently running open.

Decided: the run banner is shown to every reader, so a reader who wonders why today's draw is missing gets the answer instead of an empty card.

### Process-per-trial runner and trial budgets

*Page: [tuning](tuning.md)*

Built: `src/HyperoptRunner.py`, one coordinator for every tuner with expensive trials: forked process per trial, parallel within a memory gate, crashed trials marked failed, teardown on Ctrl+C or kill, stale `RUNNING` trials failed at the next start, `constant_liar` sampling.

Built: `HyperoptStatistics.py` got the wall-clock budget (1200 s) and prewarms the keno precompute in the parent; `HyperoptRLTicket.py` runs one trial per core with a scratch policy directory and BLAS pinned to one thread; `HyperoptQuantum.py` got an 1800 s deadline; the deep-learning child got a 2 h budget.

Evidence: the keno precompute took 43 min in production when rebuilt; a hung deep-learning trial once had to be found and killed by hand.

Built: `runHyperopt.sh` waits up to 6 h for `process.lock` instead of letting each tuner exit on a live lock.

Why: with the deep-learning rows back in the daily run, a long Saturday run would otherwise silently skip the week's tuning.

### Ensemble search: enumeration replaced TPE

*Page: [tuning](tuning.md)*

Finding: sampling boolean flags independently, TPE collapsed onto its mode: a production run spent 162 trials on 37 distinct subsets, one of them 62 times, and the process-per-trial runner spent ~2 s launching each 5 ms evaluation.

Decided: the space is enumerated exhaustively when it fits the budget, else a deterministic hill-climb; each combination evaluated once (2-20 ms in-process; writing a trial to SQLite ~70 ms).

### Ensemble subsets need 60% joint coverage

*Page: [tuning](tuning.md)*

Finding: with a looser floor the search found a six-row subset that broke even on the ten days its members coexisted and ranked it above every subset that lost keno's house edge over 300 days.

Decided: a subset counts only when its members coexist on at least 60% of the window.

Why: every scored subset then sits on nearly the same draws; with thousands of subsets such a fluke is always there to be found.

Finding: rows only weeks old are not candidates at the default window; they join as they age into it.

### CatBoost depth capped at 7 in the search

*Page: [tuning](tuning.md)*

Decided: CatBoost's tree depth is searched over 1-7; XGBoost and LightGBM keep 1-10.

Evidence: median minutes per trial by depth: 1-6 → 1-4, 8 → 24, 10 → 64, single trials up to 12 hours (symmetric trees cost 2^depth).

### Boosting cost gate for predicted timeouts

*Page: [tuning](tuning.md)*

Finding: TPE proposes its first ten trials at random, and on keno roughly half the search space cannot finish one refit block inside the budget; keno LightGBM burned six 20-minute timeouts in its first seven trials.

Decided: a configuration at least as expensive (fit-cost proxy) as one that already timed out under the same budget tag is skipped and not counted toward `--trials`.

### Boosting refit cadence

*Page: [tuning](tuning.md)*

Decided: one fit per block of days (`--refit-every`, default the window over five blocks), block-aligned on the data before the block, identical for every trial.

Evidence: eurodreams CatBoostMultiLabel (2 trials × 21 days): 7.5 min at cadence 1, 66 s at cadence 7, objective values in the same range.

### Boosting threads per library and parallel trials

*Page: [tuning](tuning.md)*

Evidence: CatBoost scales (one fit: 4 threads = 3.1x); giving XGBoost 5 threads made a study 5x slower; an inherited 16-thread BLAS pool cost ~10 cores of spin-waiting per fit.

Decided: CatBoost gets the idle cores; XGBoost and LightGBM fit single-threaded with BLAS/OpenMP pinned to one thread, and their trials run in parallel (3 at a time with 5 refit blocks on 15 cores) within a memory reserve.

### CatBoost border count exposed, default kept

*Page: [tuning](tuning.md)*

Finding: on this data (integer features with at most 50 distinct values) 32 against 254 made no difference in fit time (1014 s vs 1016 s for a 50-classifier multi-label fit) or ranking (Spearman 1.0).

Decided: `--catboost-border-count` stays at 254; the knob is for experimentation only.

### Boosting tuner rebuilt as a mirror of the statistical one

*Page: [tuning](tuning.md)*

Finding: the legacy boosting tuner could not run: it imported `src/LSTM_ARIMA_Model.py`, `src/RefinemePrediction.py` and `src/TopPrediction.py`, none of which existed any more.

Built: it evaluates through `src/Backtester.py`, scores with the shared function instead of raw total profit, bets the served keno sizes and writes `modelScores` so the boosting rows are weighted in the vote.

### Quantum score table collected once and cached

*Page: [tuning](tuning.md)*

Built: the backtest score table is persisted to `data/hyperOptCache/meta_score_table_<game>.joblib` and reused by `TrainMetaLearner.py` in the same weekly run (any new draw or changed base-model parameter recollects).

Why: the pipeline's most expensive stage runs once, not twice.

Finding: kernel and RBF-control trials run in under a second, VQC trials in seconds to half a minute.

### Special-column ranges determined from the data

*Page: [tuning](tuning.md)*

Finding: no special range is hardcoded; `determine_special_range` found Euromillions stars 1-12, EuroDreams dream number 1-5, VikingLotto super viking 1-8.

### Pick3 meta-learners use a positional formulation

*Page: [tuning](tuning.md)*

Decided: pick3 is trained on the 30 (position, digit) pairs per day with one classifier per position, instead of being skipped.

Why: a per-number ranking cannot express digit order or repeated digits.

Built: base models expose `score_positions()`, normalised per slot so the learned weights do not depend on tuned simulation counts; this gave pick3 the classical meta-learner rows for the first time.

### Real LSTM/TCN fusion replaced the Unified prototype

*Page: [models](models.md)*

Built: `UnifiedLstmTcn` and `UnifiedLstmGruTcn` concatenate their branches before a shared attention, pooling and output head, and run as two rows with their own tuner studies and prefixed keys (`unifiedLstmTcn_*`, `unifiedLstmGruTcn_*`), so they do not overwrite LSTM Base's tuned values.

Why: the old `src/Unified.py` prototype trained LSTM, GRU and TCN separately and only averaged their output probabilities afterwards: an ensemble, not a fusion, and never wired into anything.

Decided: no Markov blending in either fused model (out of scope at the time).

### Model dispatch through MODEL_REGISTRY

*Page: [models](models.md)*

Fixed: `Predictor.py` and `HyperoptDeepLearning.py` selected the model with `modelToUse = tcn if "lstm_model" not in model_type else lstm` and then called a hard-coded, LSTM-only setter block. `TCNModel` had none of those setters, so any model type other than `lstm_model` would have crashed immediately (none was ever used until the fused models).

Built: `HyperoptDeepLearning.py` dispatches through `MODEL_REGISTRY`, mapping each model type to its own instance and `configure` function.

Finding: at that time `tcn_model` was still not exercised by either script's sweep, and every dataset was hard-coded to `lstm_model`; the TCN row became live later.

### External feedforward network compared, not ported

*Page: [models](models.md)*

Finding: sminerport/SequencePredictionANN, suggested as a comparison point, uses a single-hidden-layer sigmoid feedforward network trained with MSE, without recurrence, convolution or attention; its own README lists LSTM/GRU as a future improvement.

Decided: not ported; it is a step below what `LSTM.py` and `TCN.py` already do, kept as a baseline data point only.

### Six boosting rows on one shared implementation

*Page: [models](models.md)*

Built: three libraries (XGBoost, LightGBM, CatBoost) in two formulations (per-position multiclass, multi-label) as six tracked rows sharing `src/BoostingBase.py`; XGBoost gained `score_numbers()` with this alignment.

Why: a difference between rows had to be attributable to the library or the formulation and not to incidental plumbing. Multi-hot features answer a membership question; raw sorted values encode order-statistic structure.

Decided: `XGBoost Model` keeps the earlier `xgBoost` prefix and `useBoost` flag so its tracked history stays continuous.

Decided: the multi-label formulation is skipped on Pick3, where set membership can represent neither digit order nor repeated digits.

### Keno subset sizes unified across all rows

*Page: [models](models.md)*

Fixed: the deep-learning rows hard-coded `range(5, 11)` and so bet at all six keno sizes while every statistical model played only the tuned ones (at the time just `use_10`), which made their tracked results non-comparable. Every row now asks `getKenoSubsetSizes` for the tuned `use_5`..`use_10` flags.

Built: `HyperoptDeepLearning.py` reads the same choice through `tuned_keno_subset_sizes`, falling back to all six only when a game has no tuned flags, so its profit signal never silently becomes zero.

### Per-slot vote for the positional games

*Page: [models](models.md)*

Built: on pick3 and Joker+ the `WeightedEnsemble Model` became a vote per slot (`Helpers.build_positional_vote_predictions`) over the rows that own a per-position model; argmax per slot, ties to the lowest digit, duplicates and drawn order kept; the row carries `positionConfidence`, the day file `positionFrequency`.

Why: a set-style top-3 sorts the digits and can never express a repeated digit.

Evidence: 28% of pick3 draws contain a repeated digit.

Decided: the meta rows and the deep-learning research rows do not vote: the former are aggregates of the voters, the latter emitted identical tickets, so an all-rows vote would double-count.

### SubsetEnsemble row served from a tuned selection

*Page: [models](models.md)*

Built: `SubsetEnsemble Model`: the WeightedEnsemble vote restricted to the rows `HyperoptEnsemble.py` selected (`subsetEnsembleModels`, `subsetEnsembleWeighted`), served through the shared recipe `Helpers.build_vote_ensemble_predictions` so the served ticket is what the tuner scored.

Decided: not served until a selection exists, and skipped on a day one selected row did not run; a vote among the remaining rows would be a different ensemble under this row's name.

Decided: its own result is never written into `modelScores`, so it cannot feed back into its members' weights.

### Meta-learner saw only the special column

*Page: [models](models.md)*

Fixed: the meta-learner's `collect_scores` data only ever saw the special column, never the main numbers. A second, separately trained model in the same artifact now ranks the special column(s) independently, mirroring the WeightedEnsemble split.

Decided: `HybridStatisticalModel` is excluded from the base models, since it is itself a vote over several of them and would be circular.

### MetaLearnerV2 added as its own row

*Page: [models](models.md)*

Decided: a second model class per game (`GradientBoostingClassifier`, `meta_learner_v2.joblib`) was added as its own tracked row instead of replacing `MetaLearner Model`.

Why: a tree-based model can pick up nonlinear interactions between base-model scores that a logistic regression cannot, and its errors need not correlate with the linear model's, so real-life tracking can show which is worth keeping.

Built: both rows share the base-model scores (cached by feature-name set, so nothing is scored twice).

### Rebuild flag did nothing

*Page: [models](models.md)*

Fixed: `-r/--rebuild_history` existed but did nothing; it now force-regenerates (overwrites) the last `-d` draws, existing files included.

Built: gap recovery anchors on the newest existing prediction file and builds every missing draw after it plus interior holes within the last `-d` draws, without overwriting existing files.

### NaN training runs contained

*Page: [models](models.md)*

Fixed: exploding gradients on some hyperparameter combinations (most often keno's 20 x 80 output space) drove the loss to NaN. Training stops at the first NaN batch (`TerminateOnNaN`) instead of burning the early-stop patience on dead epochs; a healthy earlier epoch's checkpoint is restored; if the very first batch is NaN the model raises and the day's row is skipped.

Why: a prediction from NaN weights is an argmax over garbage and would pollute the tracked history and the WeightedEnsemble vote.

### Research rows implemented

*Page: [models](models.md)*

Built: `GNN Model`, `Transformer Model`, `Autoencoder Model` (with the `anomalyWatch` z-score monitor), `RL Ticket Model` and the entropy and divergence analysis run in the daily pipeline as tracked rows or monitoring layers.

Decided: the three light rows run regardless of `--ai`, inside the same spawned child, each with its own `use…` flag.

Evidence: at two smoke-test epochs on CPU the three light rows together cost less than a single `UnifiedLstmGruTcn` training.

Decided: the Autoencoder's label smoothing defaults to 0 so its negative log-likelihood stays an honest likelihood.

### Hits counted per pool, not as one intersection

*Page: [scoring](scoring.md)*

Decided: main numbers scored only against the drawn mains, special columns (stars, dream number, viking number) only against the drawn specials; shown as `N (M)`. Lotto followed the real tiers: six played numbers against the six mains, the bonus only supplementing a partial match and matched against the played numbers.

Why: the pools are separate drawings, so a predicted main equal to a star's numeric value was not a hit; a full Lotto main match makes a bonus match impossible.

Built: the split was carried through the day views, `matchingNumbers` in the day files, `draws.js`, the report's `avg_hits` / `avg_special_hits`, the lag analysis, the randomness watch and `src/Backtester.py`, whose baselines were changed to bet main-sized tickets.

### Games classified once as positional or set-based

*Page: [scoring](scoring.md)*

Decided: one predicate (`Helpers.is_positional_game`) decided every positional branch instead of tests on a game name; at the time it covered Pick3 and Joker+.

Why: positional games keep the drawn order, repeat digits, use per-position models and have no set-style rows.

### Joker+ modelled as six digits plus a categorical sign

*Page: [scoring](scoring.md)*

Decided: Joker+ was modelled as a 6-slot positional game with the zodiac sign as one categorical special column, encoded 0-11 at every CSV parse site. Hits were tracked as `L/R (Z)` after the regulation (Reglement Joker+, Sept 2023): runs of consecutive matches from the left and the right end, plus the sign.

Decided: `RL Ticket Model` was left out for Joker+ and the per-slot vote rows stayed off unless `useJokerplusEnsemble` was set.

Why: the player cannot choose the digits (combinations are system-generated, only the sign is selectable), so digit ticket construction would be fiction; digit predictability is an integrity signal, and the sign's own tier only refunds the stake.

Finding: 86% of Joker+ draws repeated a digit.

### Payout rules for Pick3 and Joker+ taken from the regulations

*Page: [scoring](scoring.md)*

Built: `pick3_ticket_profit` followed the Belgian Pick-3 rules (Reglement Pick-3, July 2024): all four bet types on one ticket (4 EUR, triples 3 EUR), prizes cumulating, an exact hit netting +676 or +756. `jokerplus_ticket_profit` followed Art. 27 of the Joker+ regulation: stake 1.50 EUR, 20,000 EUR for six digits, 200,000 EUR with the sign, otherwise the left and right run prizes plus 1.50 EUR for the sign.

### Keno profit made net of the stake

*Page: [scoring](scoring.md)*

Fixed: `keno_ticket_profit` had returned gross wins and double-counted the stake on losses; it was changed to payout minus the 1 EUR stake.

Why: profit per bet had to be comparable across games.

### Short records sorted below the ranked rows

*Page: [scoring](scoring.md)*

Decided: models with fewer scored draws than `minDrawsForRanking` (10, or the largest count available if lower) were listed but marked as a short record and sorted below the ranked ones.

Why: a two-day-old model could otherwise claim "best" off one lucky draw. Averages per bet and per draw were used rather than totals because models joined the tracking at different times.

### Next draw's date derived from the stored history

*Page: [scoring](scoring.md)*

Decided: the draw a prediction applies to was derived from the weekdays of the game's last 40 stored draws rather than from a hardcoded calendar, so a schedule change is picked up and an older page is read with the schedule of its own time.

Evidence: checked against the full stored history, the rule reproduced the draw that actually followed in 765 of 767 cases; the two misses were a missing 31 December file for the two daily games, a gap in the stored data rather than a wrong schedule.

Finding: a coming-draw date already in the past appeared when the predictor had not run since, for example while the weekly hyperopt held `process.lock`.

### Phase-shift persistence check automated

*Page: [scoring](scoring.md)*

Built: each run kept only its single best lag peak per model and appended it to `data/database/lagPeakHistory.json` (one entry per run date, last 60 runs); the page showed the peak with a z-score, the most frequent peak across runs (highlighted from 3 runs with 50% agreement) and a run-length-encoded trail.

Why: with under about 100 scored draws per lag the best lag moved around by chance; only the same peak lag held over time would point to a real phase shift.

### Combination search with a shuffled-history control

*Page: [scoring](scoring.md)*

Built: per game every pair and triple of ranked rows was scored on shared draws (roadmap item 2): net profit per draw for keno, Pick3 and Joker+ with a greedy build-up, best line held per draw for the hit-scored games, and a control that re-ran the search 100 times with each draw's result moved to another day.

Why: hundreds of combinations were scored, so a best one existed by luck alone; profit per bet of a set is a stake-weighted mean and can never beat the best member, so it was shown, not ranked on.

Evidence: scores were computed once as matrix products, and the section added under two seconds to the daily run.

### Quantum track proposed as a research layer

*Page: [research](research.md)*

Decided: quantum models were added as an adversarial test of the draw process, not on the assumption that a random draw can be predicted. The first integration was to be a quantum meta-learner on the existing meta-learner table, as its own tracked row, never replacing a classical meta-learner.

Why: the quantum-kernel model gives a clean comparison with a classical RBF support-vector machine; the variational classifier was to follow once data flow, persistence and evaluation were stable.

Decided: phases Q0 (classical controls first), Q1 (quantum meta-learning), Q2 (randomness discrimination), Q3 (direct quantum scoring), Q4 (Grover candidate search). The original proposal assumed seven base-model scores reduced to four features on four qubits, an opt-in flag, and a `src/quantum/` package with an `experiments/quantum/` folder.
