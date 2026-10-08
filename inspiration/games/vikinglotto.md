# Vikinglotto (Belgium)

## Source

| Item | Value |
|---|---|
| Regulation | Royal decree 17.11.2020, amended 30.05.2021 and 10.04.2022 (BS 14.04.2022) |
| PDF (NL, May 2022) | <https://www.loterie-nationale.be/content/dam/opp/draw-games/vikinglotto/brand-assets/documents/nl/reglement-vikinglotto-mei2022.pdf> |
| PDF (FR, May 2022) | <https://www.loterie-nationale.be/content/dam/opp/draw-games/vikinglotto/brand-assets/documents/fr/reglement-vikinglotto-mai2022.pdf> |
| Payout tables per draw | <https://www.nationale-loterij.be/uitslagen-trekkingen> (Vikinglotto section) |
| Rules in force since | First draw under the 2022 rules: Wednesday 18 May 2022 |
| Retrieved | 2026-10-08 |

> ⚠️ This is the most recent version linked on the index page, but it is from 2022. Check for newer amendments.

## Game structure

| Item | Value |
|---|---|
| Type | Transnational draw lottery; common tiers 1–2, local (Belgian) tiers 3–12 |
| Matrix | 6 of 48 + 1 Viking number of 5 |
| Price | €2 per combination; in Belgium a participation is always **5 combinations (€10)** with the same 6 numbers and all 5 Viking numbers |
| Favourite Viking | The player picks one of the 5 Viking numbers as "favourite"; if drawn, local prizes (tiers 3–12) are doubled |
| Draws | Wednesday (one per week) |
| Tiers | 12 |

## Prize tiers (Belgium)

| Tier | Match | Odds 1 in (per €10 participation) | Prize |
|---|---|---|---|
| 1 (Jackpot, common) | 6 + Viking | 12,271,512 | Pari-mutuel, common pool |
| 2 (common) | 6 | 12,271,512 | Pari-mutuel, common pool |
| 3 | 5 + favourite Viking | 243,482.38 | Variable ×2 (minimum €6,000) |
| 4 | 5 | 60,870.60 | Variable (minimum €3,000) |
| 5 | 4 + favourite Viking | 4,750.88 | Variable ×2 (minimum €200) |
| 6 | 4 | 1,187.72 | Variable (minimum €100) |
| 7 | 3 + favourite Viking | 267.24 | Fixed €60 |
| 8 | 3 | 66.81 | Fixed €30 |
| 9 | 2 + favourite Viking | 36.55 | Fixed €20 |
| 10 | 2 | 9.14 | Fixed €10 |
| 11 | 1 + favourite Viking | 12.02 | Fixed €4 |
| 12 | 1 | 3.01 | Fixed €2 |

- Overall odds of any prize per participation: 1 in 1.75.
- A Belgian player who matches all 6 numbers wins tier 1 once **and** tier 2 four times (because the participation covers all 5 Viking numbers).

## Money flow

- **Common pool:** €0.185 per combination (9.25% of the Belgian stake): €0.13 to tier 1, €0.013 to tier 2, €0.042 to the common **Booster Fund**.
- **Local prizes:** 48.61% of Belgian stakes go to tiers 3–12. Fixed tiers 7–12 are paid first; the remainder goes to tiers 3–6.
- **Local Reserve Fund:** 1% of Belgian stakes; guarantees the minimum prizes of tiers 3–6. The Nationale Loterij can add or withdraw any amount.
- Split of the remainder over tiers 3–6: 79.60% to tiers 3+4 and 20.40% to tiers 5+6 (if both groups have winners); within a group a tier-3/5 winner gets two shares, a tier-4/6 winner one share. If one group has no winners, the other group gets everything; if none of tiers 3–6 has winners, the remainder goes to the Reserve Fund.

## Jackpot rules (common tiers)

- Tier 1 minimum **€3,000,000** (topped up from the Booster Fund), maximum **€25,000,000**.
- No tier 1 winner: the amount rolls over to tier 1 of the next draw. No tier 2 winner: rolls over to tier 2 of the next draw.
- Amount above the €25M tier 1 cap flows to **tier 2 of the same draw**. Tier 2 is also capped at €25M; the excess goes to tier 1 of the next draw.
- Tier 1 total may not be smaller than tier 2 total, and a tier 1 prize may not be smaller than a tier 2 prize (otherwise pooled).
- Booster Fund is capped at €7,500,000; the excess flows to tier 1 of the next draw.
- **Exceptional Promotional Draw:** the lotteries can jointly add money to the jackpot from the Booster Fund.
- Local promotions can raise tier 3–12 prizes, funded by the Reserve Fund, up to €5M per draw.

## Computed RTP (our calculation)

- Allocated to prizes: 9.25% + 48.61% + 1% = **≈ 58.9%** of Belgian stakes.
- Fixed tiers 7–12 per €10 participation (odds above): ≈ €3.31 → **≈ 33.1%**.

## EV research notes

- **Most relevant rule:** the €25M jackpot cap with overflow to **tier 2 in the same draw**. When the jackpot is capped, a Belgian 6-match pays tier 1 + 4 × tier 2, so the capped regime raises the value of a 6-match beyond the €25M. Worth modelling explicitly.
- With 1-in-12.3M odds per participation and a €25M cap, the jackpot contribution alone is at most ≈ €2.04 per €10 participation (before sharing), so a positive EV requires a large tier 2 overflow and/or promotions. The historical scan will tell.
- Sales are transnational; a sales model needs total sales across all participating lotteries. The current list of participating countries is not in the Belgian regulation and still has to be verified.

## Open questions

- [ ] Check for regulation amendments after May 2022.
- [ ] Which countries participate currently, and are their sales published?
- [ ] History of capped draws and tier 2 overflow amounts.
