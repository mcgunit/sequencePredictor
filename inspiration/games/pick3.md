# Pick3 (Belgium)

## Source

| Item | Value |
|---|---|
| Regulation | Royal decree 09.08.2002, last amended 02.07.2024 (BS 25.07.2024) |
| PDF (NL, July 2024) | <https://www.loterie-nationale.be/content/dam/opp/draw-games/pick3/brand-assets/documents/reglement-pick-3-juli2024.pdf> |
| PDF (FR, July 2024) | <https://www.loterie-nationale.be/content/dam/opp/draw-games/pick3/brand-assets/documents/reglement-pick-3-juillet2024.pdf> |
| Payout tables per draw | <https://www.nationale-loterij.be/uitslagen-trekkingen> (Pick3 section) |
| Retrieved | 2026-10-08 |

## Game structure

| Item | Value |
|---|---|
| Type | National draw lottery, **fixed odds** |
| Draw | A 3-digit number 000–999: one drum with replacement (3 draws) or three drums, or electronic |
| Bet types | "In order" (straight), "Any order" (box), "First two", "Last two" |
| Stake | €1 per bet type per grid per draw; up to 5 grids; max €560 per paper form |
| Draws | Daily (incl. Sundays and holidays since 1 October 2017) |

## Prizes (fixed)

| Bet type | Win condition | Prize | Probability |
|---|---|---|---|
| In order | Exact 3-digit number | €500 | 1 in 1,000 |
| In order (consolation) | Units digit matches (not the exact number) | €1 | 99 in 1,000 |
| Any order | 3 digits in any order, winning number has 3 different digits | €80 | 6 in 1,000 (if player's digits are all different) |
| Any order | 3 digits in any order, winning number has 2 equal digits | €160 | 3 in 1,000 (if player's number has a pair) |
| First two | Hundreds and tens digits match | €50 | 1 in 100 |
| Last two | Tens and units digits match | €50 | 1 in 100 |

- "Any order" cannot be played with three identical digits.

## Reserve fund

- 12% ("Any order"), 10% ("First two") and 10% ("Last two") of stakes go to the "Pick-3 Reserve Fund".
- The fund finances promotions and covers payouts when, for a given draw, the payout exceeds the percentage of stakes set by the Nationale Loterij.

## Computed RTP (our calculation)

| Bet type | RTP |
|---|---|
| In order | 59.9% (€500/1,000 + €1 × 0.099) |
| Any order | 48.0% (both cases) |
| First two / Last two | 50.0% |

## EV research notes

- **Fixed odds, independent of other players.** No positive EV without promotions or a bias in the draw process.
- Pick3 has the simplest outcome space of all games (3 digits, daily since 2002), making it the **best dataset for the randomness tests** in Step 9: digit frequencies per position, serial correlation, and a comparison of draw methods if the method changed over time.
- If a physical drum with replacement is used, per-ball bias would show up as digit-frequency deviations across thousands of draws.

## Open questions

- [ ] Which draw method is currently used, and has it changed over time?
- [ ] History of Pick3 promotions.
- [ ] Are historical results available back to 2002?
