# Game Rules – Belgian National Lottery (Nationale Loterij / Loterie Nationale)

> Step 1 of `RESEARCH_PLAN.md`: structured description of every game in scope.
> Collected: 2026-10-08. Source: official regulations (royal decrees, "officieuze coördinatie") published by the Nationale Loterij.

## Files

| Game | File | Regulation version used | Prize model | Draws |
|---|---|---|---|---|
| EuroMillions | [euromillions.md](euromillions.md) | February 2023 (NL) | Pari-mutuel, 13 tiers, cap + roll-down | Tue, Fri |
| Lotto | [lotto.md](lotto.md) | April 2025 (FR) – first draw under these rules 3 May 2025 | Pari-mutuel tiers 1–6, fixed tiers 7–9 | Wed, Sat |
| EuroDreams | [eurodreams.md](eurodreams.md) | August 2025 (FR) – first draw under these rules 2 Oct 2025 | Fixed annuities (tiers 1–2), pari-mutuel 3–5, fixed 6 | Mon, Thu |
| Vikinglotto | [vikinglotto.md](vikinglotto.md) | May 2022 (NL) | Common pari-mutuel tiers 1–2, local pari-mutuel 3–6, fixed 7–12 | Wed |
| Joker+ | [joker-plus.md](joker-plus.md) | September 2023 (NL) | Fixed prizes + monthly promotional amount | Daily |
| Keno | [keno.md](keno.md) | December 2019 (NL) | Fixed odds, 35 prize classes, per-class cap | Daily |
| Pick3 | [pick3.md](pick3.md) | July 2024 (NL) | Fixed odds | Daily |

## Where to find the sources

- **Index of all regulations (PDF links):**
  - NL: <https://www.nationale-loterij.be/algemene-voorwaarden/gebruiksvoorwaarden>
  - FR: <https://www.loterie-nationale.be/conditions-generales/conditions-d-utilisation>
  - Note: on the index page (checked 2026-10-08) the NL Lotto link still points to the December 2019 version, while the FR link points to the April 2025 version. Always use the most recent version and check the date of the royal decree.
  - Only the royal decree published in the *Belgisch Staatsblad / Moniteur belge* is legally binding; the PDFs are "unofficial coordinations". The Belgian legal database (Justel, <https://www.ejustice.just.fgov.be>) has the official texts and version history.
- **Payout tables per draw (winners and amounts per tier):**
  - NL: <https://www.nationale-loterij.be/uitslagen-trekkingen>
  - FR: <https://www.loterie-nationale.be/resultats-tirages>
  - The results pages are rendered with JavaScript; the static HTML does not contain the data. Scraping will need a headless browser or the underlying JSON API the page calls (to investigate in Step 2).
  - The EuroMillions FAQ confirms that the prize distribution ("winstverdeling") per draw is published on the results page.

## Summary of computed return-to-player (RTP)

These figures are **computed by us from the regulation** (odds × fixed prizes, or the stated pool percentages). They are averages; pari-mutuel tiers vary per draw.

| Game | Stake | Long-run RTP | Can EV exceed the stake? |
|---|---|---|---|
| EuroMillions | €2.50 (€2.20 common game + €0.30 local code draw) | ≈ 49–51% | Only in special situations: cap reached + roll-down, Super Draws, promotions. See file. |
| Lotto | €1.50 | ≈ 50% (29.65% pools + ≈ 20.45% fixed tiers) + code draw | Only with discretionary roll-down or promotions. |
| EuroDreams | €2.50 | 52% (nominal; annuities are paid over 5–30 years) | Very unlikely: top tiers are fixed amounts, no accumulating jackpot. |
| Vikinglotto | €10 per participation (5 × €2) | ≈ 58.9% allocated (9.25% + 48.61% + 1%) | Only at large jackpots / promotions; needs sales and sharing model. |
| Joker+ | €1.50 | ≈ 47.9% base | Monthly promo (max €3M accumulated) is not enough on its own; a "+300% prizes" promotion would be. See file. |
| Keno | €1–€10 per grid | ≈ 50.7–54.0% depending on numbers played | No (fixed odds), except promotions. |
| Pick3 | €1 per game | 48–59.9% depending on bet type | No (fixed odds), except promotions. |

## Common rules (all games)

- Prizes won in Belgium are free of tax on winnings (explicitly stated in the EuroMillions, EuroDreams and Vikinglotto regulations).
- Claim period: 20 weeks from the draw date; unclaimed prizes go to the Nationale Loterij (Vikinglotto: unclaimed tier 1/2 prizes ≥ €3M go to the Booster Fund).
- Prizes above €2,000 require identification.
- Minors are not allowed to play.

## Open points for Step 1

- [ ] Check whether newer regulation versions exist (especially Vikinglotto May 2022, Keno December 2019, EuroMillions February 2023).
- [ ] Download and archive every PDF in the repo (`data/rules/pdf/`) with its retrieval date, since the website only shows the current version.
- [ ] Collect the separate "deelnemingsregels" (participation rules) per game; they contain system/multi-ticket options and maximum stakes.
- [ ] Convert each file into machine-readable `data/rules/<game>.yaml` for the EV calculator (Step 3).
