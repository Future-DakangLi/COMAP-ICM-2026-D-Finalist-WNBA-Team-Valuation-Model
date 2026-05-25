# Clean Improvement Note

This version keeps the no-answer-leak rule: the pipeline does not read any table of reported paper outputs and does not scale intermediate values to known final answers.

Compared with the earlier no-leak version, the reproduction is improved by changing only model-side assumptions that are visible and defensible:

1. Future baseline forecast uses a conservative roster-uncertainty Elo adjustment, while still reporting the raw Elo point forecast separately.
2. Future ticket prices use recent home minimum listed ticket prices and a minimum-to-average yield multiplier, instead of relying only on a generic fallback price.
3. Team valuation uses a sports-franchise revenue-multiple anchor while still reporting EBITDA and simplified DCF as sanity checks.
4. Monte Carlo season risk aggregates game-level attendance error across home games rather than treating every validation error as a full-season shock.

These choices are intended to make the reconstructed workflow closer to the paper's modeling story while avoiding any direct use of the paper's final numeric answers inside the code.


## Clean-v2 additions

- Uses a rounded player-cost assumption and rounded franchise revenue multiple.
- Adds an interest-rate-sensitive valuation discount, so rate shocks affect valuation instead of only interest expense.
- Keeps simulation draws random; no quantile matching or output scaling is used.
