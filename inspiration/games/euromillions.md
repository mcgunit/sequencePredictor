# EuroMillions (Belgium)

## Source

| Item | Value |
|---|---|
| Regulation | Royal decree 01.04.2016, last amended by royal decree 14.02.2023 (BS 22.02.2023) |
| PDF (NL) | <https://www.loterie-nationale.be/content/dam/opp/draw-games/euromillions/brand-assets/documents/nl/draw-eum-reglement-februari-2023-nl.pdf> |
| PDF (FR) | <https://www.loterie-nationale.be/content/dam/opp/draw-games/euromillions/brand-assets/documents/fr/draw-eum-reglement-fevrier-2023-fr.pdf> |
| Payout tables per draw | <https://www.nationale-loterij.be/uitslagen-trekkingen> (EuroMillions section) |
| Retrieved | 2026-10-08 |

## Game structure

| Item | Value |
|---|---|
| Type | Transnational draw lottery, common prize pool across participating lotteries |
| Matrix | 5 of 50 numbers + 2 of 12 stars → 139,838,160 combinations |
| Price in Belgium | €2.50 per combination = €2.20 common game + €0.30 mandatory local electronic code draw |
| Draws | Tuesday and Friday evening (organised by SLE – Services aux Loteries en Europe) |
| Draw method | One or two drums (or electronic/physical device), supervised by a bailiff or independent auditor |
| Tiers | 13, all pari-mutuel |

## Money flow (common game)

- Each participating lottery transfers **€1.10 per combination** (50% of €2.20) to the common prize pool.
- That pool is split over the tiers and the common **Reserve Fund**:

| Tier | Match | Odds 1 in | % of common prize pool |
|---|---|---|---|
| 1 (Jackpot) | 5 + 2 | 139,838,160 | 50% (draws 1–5 of a cycle) or 42% (from draw 6) |
| 2 | 5 + 1 | 6,991,908 | 2.61% |
| 3 | 5 + 0 | 3,107,514.67 | 0.61% |
| 4 | 4 + 2 | 621,502.93 | 0.19% |
| 5 | 4 + 1 | 31,075.15 | 0.35% |
| 6 | 3 + 2 | 14,125.07 | 0.37% |
| 7 | 4 + 0 | 13,811.18 | 0.26% |
| 8 | 2 + 2 | 985.47 | 1.30% |
| 9 | 3 + 1 | 706.25 | 1.45% |
| 10 | 3 + 0 | 313.89 | 2.70% |
| 11 | 1 + 2 | 187.71 | 3.27% |
| 12 | 2 + 1 | 49.27 | 10.30% |
| 13 | 2 + 0 | 21.90 | 16.59% |
| Reserve Fund | – | – | 10% (draws 1–5) or 18% (from draw 6) |

- Overall odds of any prize: 1 in 12.97.
- Prizes are not cumulative: a combination only wins in its highest tier.
- Rounding: tier 1 rounded up to the euro; other tiers rounded down to €0.10.
- During Super MJG draws and Super Draws (and the rest of that cycle after a Super MJG draw) the 42% / 18% split applies.

## Rollover, cap and roll-down rules

- **No tier 1 winner:** tier 1 amount rolls over to tier 1 of the next draw.
- **No winner in tiers 2–12:** the amount moves to the next lower tier of the same draw. **No winner in tier 13:** amount goes to tier 1 of the next draw.
- **Jackpot cap:** initial cap €200M, rising by €10M each time the cap is reached, up to a maximum of €250M. The cap stays fixed for the whole cycle; the participating lotteries can change it.
- **Flow down:** while the jackpot is at the cap, the amount *above* the cap flows to the next lower tier **with at least one winner** in that same draw.
- **Roll down:** if **five consecutive draws** at the capped amount produce no tier 1 winner, the capped amount of the fifth draw rolls down to the next lower tier with at least one winner in that draw.
- A cycle ends when the jackpot is won or after a roll-down.

## Special draws

- **Super Draw:** jackpot guaranteed at an agreed minimum (top-up from the Reserve Fund); if not won, it rolls down to the next tier with winners in the same draw.
- **Super MJG draw (Minimum Jackpot Guarantee):** jackpot guaranteed at an agreed minimum (top-up from the Reserve Fund); if not won, it rolls over normally.
- **Event draw:** special rules and tier amounts agreed between the lotteries and announced in advance.
- **Exceptional Promotional Draw:** an extra electronic draw (codes) with guaranteed prizes, cost included in the €2.20.

## Local electronic draw (Belgium only)

- Every combination gets a code (4 letters starting with "B" + 5 digits); participation is automatic and costs €0.30 of the €2.50.
- 40%–60% of the €0.30 (set by the Nationale Loterij) goes to a dedicated fund.
- Prizes are guaranteed to be won: at least one prize, each prize ≥ €1, total ≥ €20,000 per local draw.
- The Nationale Loterij can add extra cash or in-kind prizes for selected draws.

## Computed RTP (our calculation)

- Common game: €1.10 of €2.20 = 50% (incl. Reserve Fund contribution, which is paid out later).
- Local draw: €0.12–€0.18 of €0.30.
- Total per €2.50: ≈ (€1.10 + €0.12…€0.18) / €2.50 ≈ **49%–51%**.

## EV research notes

- **Most relevant rule for the research question:** the roll-down after five capped draws and the Super Draw roll-down. On such draws the jackpot money (up to €250M) is redistributed to lower tiers, comparable in mechanism to the Cash WinFall roll-down. Whether EV per ticket exceeds €2.50 depends on ticket sales (which spike) and the number of winners in the receiving tier. Candidate for the historical EV scan (Step 6).
- **Flow down** during capped draws raises the next tier with winners (usually tier 2) whenever the jackpot exceeds the cap.
- The local code draw has a guaranteed minimum (≥ €20,000); its EV per ticket depends on Belgian sales only.
- Ticket sales are pan-European, so a sales model (Step 4) needs total sales across all participating countries, not only Belgium.

## Open questions

- [ ] Check for a regulation version newer than February 2023.
- [ ] Current cap value and history of cap cycles (when €250M was reached, roll-down dates).
- [ ] Historical list of Super Draws and their guaranteed amounts.
- [ ] Are total (pan-European) sales per draw published?
