# Research Findings & Plan: Expected-Value Analysis of Lottery-Style Games

> Part of the Sequence Predictor project. Companion to `inspiration/IDEAS.md`.
> Status: draft. Last updated: 2026-10-08.

---

## 1. Research question

**Can lottery-style games become profitable (positive expected value) through their rules and money flows, rather than through predicting the drawn numbers?**

Focus areas:

- Prize structures where lower tiers (e.g. 2+ matching numbers) receive redistributed money.
- Roll-down rules, jackpot caps, must-be-won draws, superdraws and promotions.
- Prize sharing: how the number and behaviour of other players affects the payout.
- Scratch-card games where the value of the remaining tickets can exceed their price.
- Whether advancing technology (ML, quantum computing) changes any of the above.

---

## 2. Key findings so far

### 2.1 Predicting fair draws is not a viable route

- Published research consistently finds no exploitable patterns in properly run draws. A 2005 study tested three lotteries with statistical randomness tests, ARIMA and neural networks and found no patterns or suspicious data.
- Better ML or quantum computing does **not** change the probabilities of a fair draw. Physical ball draws have nothing to compute in advance, and properly designed random generators remain unpredictable.
- Conclusion: the prediction models in this project should be kept as a **falsification track** (see Step 9), not as the main route to profitability.

### 2.2 Where real advantages have historically come from

The advantage is in **expected value (EV)**, driven by the rules, not by prediction.

| Case | Mechanism | Lesson |
|---|---|---|
| **Stefan Mandel** (14 wins, 1960s–1992) | Bought (almost) all combinations when the jackpot exceeded the total cost of all combinations, funded by a syndicate. Most famous: Virginia 1992, ~7M combinations, ~$27M jackpot. | EV > cost can arise from jackpot size alone. Rules were changed afterwards to block bulk buying. *(Details still to verify, see 3.4.)* |
| **Massachusetts Cash WinFall** (2004–2012) ✅ verified | Jackpot capped at ~$2M. If not won, the money "rolled down" to the 5-, 4- and 3-match tiers, making those prizes 5–10× bigger. During a roll-down there was at least $2.30 in prize money per $2.00 ticket sold. An MIT group found that buying at least $600,000 of tickets per roll-down gave an average profit above 15%. The Selbees ran a similar operation. Per the Inspector General's report, syndicates spent about $40M over seven years and won an estimated $48M. | Lower-tier "loopholes" are real, but only when accumulated money is redistributed. Risk: a syndicate once triggered a premature roll-down by flooding ticket sales, and prize sharing grew as more syndicates joined. |
| **Unpopular combinations** (UK, NZ studies) | Popular numbers/patterns lead to more prize sharing. Choosing unpopular combinations raises the expected payout. | Player behaviour is measurable from published winner counts and directly affects EV. |
| **Powerball EV window** (Kim & Skiena) | With collisions (shared wins) modelled, Powerball tickets had positive EV for pool sizes between ~$775M and ~$1.67B under the sales model of the time. | Prize sharing must be modelled explicitly; very large jackpots attract so many sales that EV can turn negative again. |

### 2.3 Why fixed low-tier prizes alone are never profitable

Operators set prize levels so the total payout (typically ~50% of revenue) stays well below the ticket price. A positive EV only arises when **accumulated money is redistributed**:

- Roll-down rules (unwon jackpot money flows to lower tiers), as in Cash WinFall.
- Jackpot caps / must-be-won draws (e.g. EuroMillions cap; surplus goes to the next tier with winners; rules have changed in recent years, verify the current version).
- Superdraws, promotions and guaranteed prizes.
- Tax treatment (in Belgium, lottery prizes are in principle tax-free for the winner; relevant for comparing countries, not a source of positive EV by itself).

### 2.4 Where ML is genuinely useful

ML does not predict the draw, but it can model the **other inputs of the EV calculation**:

- **Ticket sales** as a function of jackpot size, weekday and season (sales spike with big jackpots, increasing prize sharing).
- **Number/combination popularity**, estimated from published winner counts per tier.
- **Risk**: variance is enormous even with positive EV, so Kelly criterion and risk-of-ruin analysis decide whether a strategy is usable in practice.

### 2.5 Scratch cards: a separate loophole type

For scratch-off games, some operators publish how many tickets were printed per prize level and which prizes are still unclaimed. The EV of the remaining tickets is then the value of all remaining prizes divided by the cost of all remaining tickets. When most tickets of a game are sold but top prizes remain unclaimed, that EV rises and can exceed the ticket price. Whether the Nationale Loterij publishes "prizes remaining" data needs to be checked (see Step 10).

### 2.6 Covering designs for low tiers

Cushing & Stewart showed that 27 specific tickets guarantee at least two matching numbers on UK Lotto. Combinatorial covering designs like this guarantee low-tier wins, but a guaranteed win is not a positive EV: the expected return of such a set still has to be compared against its cost (in their worked example the return was worse than a single ticket). Relevant for evaluating "earnings from 2 numbers" rules.

### 2.7 Lesson learned: verify every source

A local 20B model (gpt-oss via opencode) produced a table of three convincing but **fabricated** sources (a non-existent Springer book, a non-existent journal claiming 92.4% accuracy, and a GitHub repo returning 404). Rules for this project:

- Every reference needs a DOI, a Google Scholar/arXiv hit, or a working link before it is added.
- Ask local models to open every link they cite (opencode `webfetch`); this catches dead links but not invented book/journal titles.
- Claims of high prediction accuracy on lottery draws are a red flag by default.
- Code repositories listed below were found via search and have not been reviewed for quality or maintenance status yet. Treat them as examples, not production libraries.

---

## 3. References

### 3.1 Literature (verified via web search)

**Prize sharing & player behaviour**

- Cox, Nicole, Takeda et al. (1998). *Using Maximum Entropy to Double One's Expected Winnings in the UK National Lottery.* University of Southampton. Estimated the popularity of all ~14M UK tickets from winner counts in the 3/4/5-match tiers. <https://eprints.soton.ac.uk/250895>
- Crack & Whigham (University of Otago). Research on New Zealand Lotto number choices: randomly generated tickets have a higher expected payoff than self-selected ones because they avoid popular patterns. Coverage: <https://www.1news.co.nz/2026/07/08/lotto-whats-better-a-lucky-dip-or-your-own-numbers/>
- Penrice, S. *Machine Learning for Understanding Lottery Players' Preferences.* Uses published prize amounts per tier in parimutuel games to infer player preferences. <https://nycdatascience.com/blog/?p=8474>
- *Patterns in manually selected numbers in the Israeli lottery.* Judgment and Decision Making. Shows the ticket's geometric layout affects number popularity. <https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F7167C1DD46E4876DAFCDDD6CE8F238C/S193029750000807Xa.pdf/patterns-in-manually-selected-numbers-in-the-israeli-lottery.pdf>

**Expected value, syndicates & collisions**

- Kim, A. & Skiena, S. *Maximizing the Expected Value of a Lottery Ticket: How to Sell and When to Buy* (arXiv:2101.04242). Models collisions (shared winning tickets) and derives the jackpot range with positive EV for Powerball. Key reference for Step 3. <https://arxiv.org/abs/2101.04242>
- *A Method for Winning at Lotteries* (arXiv:1801.02958). Shows a coalition of players gets a higher expected value than individuals playing independently; summarises Chernoff (unpopular numbers in Massachusetts) and Cook & Clotfelter (1993, syndicate EV formula). <https://arxiv.org/pdf/1801.02958>
- Scientific American, *Is the Lottery Ever a Good Bet?* Accessible explanation of EV including jackpot splitting; notes that sales figures are not published before draws, which makes real-time EV hard to determine. <https://www.scientificamerican.com/article/is-the-lottery-ever-a-good-bet/>

**Massachusetts Cash WinFall**

- MIT Alumni Association, *A Calculated Approach to Winning the Lottery.* MIT group (Random Strategies), James Harvey, $600k+ per roll-down for >15% average profit. <https://alum.mit.edu/node/1378>
- The Tech (MIT newspaper, 2012). Background on Harvey's analysis and the group's wagers. <https://thetech.com/2012/08/01/cashwinfall-mit-v132-n30>
- The Atlantic, *How a Group of MIT Students Gamed the Massachusetts State Lottery* (archived copy). $2.30 prize money per $2 ticket in roll-downs; premature roll-down triggered by bulk buying. <https://sites.oxy.edu/lengyel/m150/Winfall%20lottery/How%20a%20Group%20of%20MIT%20Students%20Gamed%20the%20Massachusetts%20State%20Lottery%20-%20The%20Atlantic.htm>
- Freakonomics, *A Lottery for People Who Are Good at Math* (2009). Early worked EV example for roll-down conditions. <https://freakonomics.com/2009/01/a-lottery-for-people-who-are-good-at-math/>
- Massachusetts Inspector General, Cash WinFall letter/report (July 2012). Primary source; old URL, may have moved. <http://www.mass.gov/ig/publications/reports-and-recommendations/2012/lottery-cash-winfall-letter-july-2012.pdf>

**Randomness & prediction (falsification track)**

- *Analysis of the results of lotteries using statistical methods and artificial neural networks* (2005). Randomness tests, ARIMA and ANN on three lotteries; no patterns found. <https://redi.cedia.edu.ec/document/23527>
- *Predicting Pseudo-Random and Quantum Random Number Sequences using Hybrid Deep Learning Models* (2023, CEUR-WS Vol. 3426). Good template for comparing models against a random benchmark. <https://ceur-ws.org/Vol-3426/paper7.pdf>

### 3.2 Code examples (found via search, not yet reviewed)

| Plan step | Repository / resource | What it offers |
|---|---|---|
| Step 2 – Data | [daowa89/lottery-archive](https://github.com/daowa89/lottery-archive) | Automated CSV archive (GitHub Actions) including EuroMillions from 2004. Check whether it has prize amounts and winner counts per tier. |
| Step 3 – EV | [OliverSherouse/powerball_post](https://nbviewer.jupyter.org/github/OliverSherouse/powerball_post/blob/master/Powerball.ipynb) | Jupyter notebook: EV contribution per prize tier and simulated winnings (pandas, matplotlib). Template for the `ev` module. |
| Step 3 – EV | [DQYDJ Powerball analysis](https://dqydj.com/?p=5088) | Worked example of the Poisson method for jackpot splitting (article, not a repo). |
| Step 3 / 8 – EV, monitoring | [hamshro/powerball-EV-tracker](https://github.com/hamshro/powerball-EV-tracker) | Simple EV tracker from odds, price and current jackpot (Python/HTML). No prize sharing. |
| Step 3 / 7 – Low tiers | [prowlett/lottery-guaranteed-win](https://github.com/prowlett/lottery-guaranteed-win) | Code for the 27-ticket covering design guaranteeing 2+ matches on UK Lotto, including expected return calculation. |
| Step 10 – Scratch cards | [Owoga/lottery_data_scraper](https://git.owoga.com/Owoga/lottery_data_scraper) | Scrapes US scratch-off "prizes remaining" data and computes EV per game. |
| Step 9 – Randomness | [sts-pylib](https://pypi.org/project/sts-pylib/) | Python interface to the NIST Statistical Test Suite (`pip install sts-pylib`). |
| Step 9 – Randomness | [dj-on-github/sp800_22_tests](https://github.com/dj-on-github/sp800_22_tests) | Pure-Python SP800-22 Rev 1a implementation, one file per test. |
| Step 9 – Randomness | [stevenang/randomness_testsuite](https://github.com/stevenang/randomness_testsuite) | Python 3 NIST suite with GUI and batch testing (Section 4 aggregate statistics). |
| Step 9 – Randomness | [terrillmoore/NIST-Statistical-Test-Suite](https://github.com/terrillmoore/NIST-Statistical-Test-Suite) | Original NIST STS C code with GitHub packaging. |

### 3.3 Verified items (moved from "to verify")

- [x] Massachusetts Cash WinFall: roll-down mechanism, MIT group and Selbees, profitability figures, IG report (see 3.1).

### 3.4 Still to verify (from general knowledge)

- [ ] Stefan Mandel: number of wins, Virginia 1992 figures, subsequent rule changes.
- [ ] Cook & Clotfelter (1993): original paper on lottery syndicate expected value.
- [ ] Current EuroMillions jackpot cap and roll-down rules.
- [ ] Current Belgian tax treatment of lottery prizes.
- [ ] TestU01 and dieharder (documentation links).
- [ ] Time-series foundation models for baselines: Chronos, TimesFM, Moirai.
- [ ] Whether the Nationale Loterij publishes scratch-card "prizes remaining" data.

---

## 4. Step-by-step plan

Work through the steps in order; each step has a clear deliverable and "done" criterion. Steps 9 and 10 can run in parallel with the main track.

### Step 1: Collect the game rules

**Goal:** a structured description of every game in scope.

- Start with **EuroMillions**; add Belgian Lotto (and others) afterwards.
- Per game, record: prize tiers and odds, % of the prize pool per tier, fixed vs. pool-dependent prizes, jackpot cap, roll-down / must-be-won rules, superdraw and promotion rules, ticket price, rule-change history with dates.
- **Deliverable:** `data/rules/<game>.yaml` (or similar), one file per game, including the date range each rule set applies to.
- **Done when:** every prize tier of the selected game can be computed from the file.

### Step 2: Collect historical data

**Goal:** a clean dataset per draw.

- Per draw: date, drawn numbers, jackpot amount, carryover, number of winners per tier, prize amount per tier, ticket sales (if published).
- Identify official sources (operator websites, published result archives) and record where each field comes from. Evaluate `daowa89/lottery-archive` as a starting point.
- **Deliverable:** `data/draws/<game>.csv` plus a short data-source note in the README.
- **Done when:** the dataset covers the period of the current rule set without gaps.

### Step 3: Build the baseline EV calculator

**Goal:** compute the expected value of one ticket for a given draw.

```
EV(ticket) = Σ over tiers [ P(tier) × expected prize(tier) ] − ticket price
```

- Start simple: assume uniform number choice by other players (no popularity effects).
- Expected prize per tier = pool for that tier divided by the expected number of winners (Poisson model based on ticket sales, see DQYDJ example and Kim & Skiena).
- **Deliverable:** an `ev` module with unit tests against hand-calculated examples.
- **Done when:** the calculator reproduces the operator's published odds and average payout percentage, and reproduces the Cash WinFall roll-down EV (> ticket price) as a sanity check.

### Step 4: Model ticket sales

**Goal:** predict sales for a draw, since sales drive prize sharing.

- Features: jackpot size, carryover count, weekday, season, holidays, special events.
- Start with linear/GBM models; validate with walk-forward backtesting.
- **Deliverable:** a sales model plus an evaluation report.
- **Done when:** the model clearly beats a naive baseline (e.g. previous draw's sales).

### Step 5: Model number and combination popularity

**Goal:** estimate how often each number/combination is played.

- Use winner counts per tier versus expected winners under uniform choice (Southampton maximum-entropy approach).
- Feed popularity into the EV calculator to get EV per specific ticket instead of per average ticket.
- **Deliverable:** a popularity model plus EV per combination.
- **Done when:** the model explains winner-count variation better than the uniform assumption.

### Step 6: Historical EV scan

**Goal:** answer the core question with data.

- Run the full EV calculator over every historical draw.
- Flag draws where EV > ticket price, and analyse which rule or situation caused it (roll-down, cap reached, promotion, low sales).
- **Deliverable:** a report with the EV distribution over time and a list of positive-EV draws with their cause.
- **Done when:** the question "has this game ever had positive EV, and why?" is answered.

### Step 7: Risk and strategy analysis

**Goal:** determine whether positive-EV situations are practically usable.

- Variance per strategy, Kelly criterion for stake sizing, risk of ruin, capital required.
- Syndicate scenarios (coalition EV, see arXiv:1801.02958) and the effect of other syndicates joining (Cash WinFall lesson).
- Covering designs for guaranteed low-tier wins (see `prowlett/lottery-guaranteed-win`): compare cost versus expected return.
- Practical constraints: purchase limits, logistics, rules on bulk buying.
- Extend the existing **RL Ticket Model** to optimise expected payout rather than "predicted numbers".
- **Deliverable:** a strategy report per game.

### Step 8: Monitoring

**Goal:** detect future positive-EV draws automatically.

- Scheduled job that fetches the upcoming draw's jackpot and rules state, predicts sales, computes EV and alerts when a threshold is crossed.
- Track rule changes by operators (these can open or close loopholes).
- **Deliverable:** a monitoring job plus alerting.

### Step 9: Falsification track for prediction models (parallel)

**Goal:** keep the existing prediction work honest.

- **Synthetic control:** train every model on draws generated by a cryptographically secure RNG. Any "pattern" found there is overfitting.
- **Randomness tests on draw data:** chi-square per number, gap and serial tests, and comparison against the hypergeometric distribution.
- **NIST SP 800-22 suites** (sts-pylib, sp800_22_tests, randomness_testsuite): designed for long bit streams, so mainly use them on the synthetic control data. A few thousand draws is small and not naturally bit-based; converting draws to bits can skew results.
- **Hard baselines:** compare against uniform random picks using the known hypergeometric match distribution; walk-forward evaluation only.
- **Multiple-comparisons correction** (Bonferroni / permutation tests) when testing many models.
- **Physical process:** if operators publish machine/ball-set per draw, test for conditional bias.
- **Deliverable:** an evaluation report per model.

### Step 10: Scratch-card track (parallel)

**Goal:** check whether scratch-card games offer positive-EV moments.

- Check whether the Nationale Loterij (or other operators in scope) publishes tickets printed per prize level and prizes remaining.
- If so, build a scraper and compute EV of the remaining tickets per game over time (see `Owoga/lottery_data_scraper` as an example).
- **Deliverable:** a data-availability note and, if data exists, an EV report per game.
- **Done when:** it is clear whether this track is feasible for the games in scope.

### Step 11: Documentation and write-up

- Keep `README.md` updated after every step (data sources, how to run the EV calculator, current findings).
- Update `inspiration/IDEAS.md` with the new directions and verified references.
- Final write-up: method, results per game, limitations.

---

## 5. Open questions

- Which games are in scope besides EuroMillions (Belgian Lotto, scratch cards, others)?
- Which historical data does the operator publish (winner counts per tier, ticket sales)?
- Does the operator publish drawing machine / ball set per draw?
- What are the current EuroMillions cap and roll-down rules, and when did they last change?
- Does the Nationale Loterij publish scratch-card prizes-remaining data?
