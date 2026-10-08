# Keno (Belgium)

## Source

| Item | Value |
|---|---|
| Regulation | Royal decree 18.01.2008, last amended 14.10.2019 (BS 03.12.2019) |
| PDF (December 2019, same link on NL and FR index) | <https://www.loterie-nationale.be/content/dam/opp/draw-games/keno/brand-assets/documents/reglement-keno-december2019.pdf> |
| Payout tables per draw | <https://www.nationale-loterij.be/uitslagen-trekkingen> (Keno section) |
| Retrieved | 2026-10-08 |

> ⚠️ The most recent version linked is from 2019. Check for newer amendments.

## Game structure

| Item | Value |
|---|---|
| Type | National draw lottery, **fixed odds** |
| Matrix | Pick 2–10 numbers from 1–70; 20 numbers are drawn |
| Stake | Base €1 per grid; stake multiplier €1, 2, 3, 4, 5 or 10 per grid (prizes scale linearly) |
| Grids | Even number of grids (2, 4, 6, 8 on paper; up to 20 via Quick-Pick); "Keno Pack" = 4 grids with 10, 9, 5 and 2 numbers |
| Draws | Daily; drum (70 balls, 20 drawn) or rotating plate, or electronic |

## Prize table (per €1 base stake)

| Numbers played | Matches → prize (€) |
|---|---|
| 10 | 10 → 250,000 · 9 → 2,000 · 8 → 200 · 7 → 10 · 6 → 4 · 5 → 1 · **0 → 3** |
| 9 | 9 → 50,000 · 8 → 500 · 7 → 50 · 6 → 5 · 5 → 2 · **0 → 3** |
| 8 | 8 → 10,000 · 7 → 100 · 6 → 10 · 5 → 4 · **0 → 3** |
| 7 | 7 → 3,000 · 6 → 30 · 5 → 3 · **0 → 3** |
| 6 | 6 → 200 · 5 → 20 · 4 → 4 · 3 → 1 |
| 5 | 5 → 150 · 4 → 5 · 3 → 2 |
| 4 | 4 → 30 · 3 → 2 · 2 → 1 |
| 3 | 3 → 16 · 2 → 1 |
| 2 | 2 → 6.50 |

- 35 prize classes in total; each grid wins in one class only (highest number of matches).
- **Cap per class per draw:** total payout per class is limited to €3,000,000 (class 1, 10 of 10: €5,000,000). If exceeded, the cap is shared equally among the winners of that class (rounded down to the cent).
- Prizes can be raised by promotions announced by the Nationale Loterij.

## Computed RTP (our calculation, hypergeometric: 20 of 70 drawn)

| Numbers played | RTP |
|---|---|
| 2 | 51.1% |
| 3 | 50.7% |
| 4 | 53.7% |
| 5 | 52.3% |
| 6 | 52.9% |
| 7 | 54.0% |
| 8 | 52.4% |
| 9 | 53.4% |
| 10 | 52.5% |

## EV research notes

- **Fixed odds:** EV per grid is fixed by the table and independent of other players. Without promotions, no positive EV is possible.
- The only paths to a positive EV are **promotions** (raised prizes) or a **bias in the draw process** (rotating plate or drum).
- Daily draws over many years make Keno a good dataset for the **randomness tests** in Step 9 (20 numbers per draw, large sample).
- The per-class payout cap means heavily played popular grids can be capped on a lucky draw; relevant only for very large syndicates.

## Open questions

- [ ] Check for regulation amendments after December 2019.
- [ ] Which draw method is currently used (drum, rotating plate, electronic)? Relevant for the bias tests.
- [ ] History of Keno promotions.
