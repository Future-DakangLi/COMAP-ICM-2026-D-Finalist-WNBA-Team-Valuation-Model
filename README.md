# From Wins to Worth — Python

This repository is a **from-scratch workflow reconstruction** for the sports-team performance, attendance, revenue, valuation, and leverage model.

The important design choice in this version is simple:




## How to run

```bash
pip install -r requirements.txt
python -m src.run_all
```

To attempt online collection from public sources, run:

```bash
ONLINE_COLLECTION=1 python -m src.run_all
```

If online collection fails, the workflow still uses the raw/source-like offline snapshot in `data/00_seed/`, so the whole pipeline remains runnable.

## Folder map

```text
config/                         model settings only, no reported result targets
data/00_seed/                   raw/source-like snapshot and assumption inputs
data/01_raw_downloaded/         staged raw inputs after collection step
data/02_interim/                cleaned game-level table
data/03_processed/              model predictions
src/                            all Python pipeline steps
results/                        generated tables
figures/                        generated plots
```

## Pipeline

1. `step00_collect.py` — collect/stage raw inputs  
2. `step01_clean.py` — clean games, capacity, attendance, rest days  
3. `step02_explore.py` — exploratory plots  
4. `step03_elo.py` — rolling Elo model and future win forecast  
5. `step04_attendance_pricing.py` — attendance model and dynamic ticket pricing  
6. `step05_finance_valuation.py` — revenue, profit, valuation, leverage  
7. `step06_optimization.py` — NO_MOVE / SIGN / TRADE comparison  
8. `step07_scenarios.py` — interest, market expansion, injury scenarios  
9. `step08_monte_carlo.py` — simulation from residual and business shocks  
10. `step09_sensitivity.py` — one-factor sensitivity  
11. `step10_business_plots.py` — final charts



## Clean-v2 changes

This version improves the earlier no-leak package using only visible model-side choices:

- a rounded conservative roster-risk Elo adjustment;
- a rounded market revenue multiple instead of forcing valuation to a known output;
- an interest-rate discount factor for franchise valuation;
- season-level Monte Carlo aggregation instead of forcing simulation quantiles.
