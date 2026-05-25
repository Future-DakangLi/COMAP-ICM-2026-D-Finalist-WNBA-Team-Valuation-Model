"""Step 08: Monte Carlo risk simulation from model residuals and assumed shocks."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .paths import RAW, PROCESSED, RESULTS, FIGURES, ensure_dirs
from .utils import read_config, safe_write_csv
from .step05_finance_valuation import evaluate_finance

def main():
    ensure_dirs()
    cfg = read_config()
    mc = cfg['monte_carlo']
    rng = np.random.default_rng(int(mc['random_seed']))
    f = pd.read_csv(RAW/'finance_input_assumptions.csv').iloc[0].to_dict()
    base = pd.read_csv(RESULTS/'02_business_outputs'/'baseline_finance_summary.csv').iloc[0].to_dict()
    valid = pd.read_csv(RESULTS/'01_model_validation'/'attendance_validation_metrics.csv').iloc[0].to_dict()
    home_games_table = pd.read_csv(PROCESSED/'future_home_attendance_ticket_predictions.csv')
    n = int(mc['n_simulations'])
    # Use validation error as a rough scale. Per-game MAPE overstates season-level
    # ticket revenue uncertainty because positive and negative game errors partly
    # diversify across home games.  We therefore aggregate by sqrt(number of home
    # games), a standard independent-error approximation.  This is deliberately
    # not fitted to any pre-known quantiles.
    home_games = max(len(home_games_table), 1)
    raw_mape = float(valid['mape']) if pd.notna(valid['mape']) else 0.10
    rel_ticket_sigma = max(raw_mape / np.sqrt(home_games), float(mc['ticket_revenue_sigma_floor']))
    ticket_shock = rng.lognormal(mean=-0.5*rel_ticket_sigma**2, sigma=rel_ticket_sigma, size=n)
    other_shock = rng.lognormal(mean=-0.5*float(mc['other_revenue_sigma'])**2, sigma=float(mc['other_revenue_sigma']), size=n)
    cost_shock = rng.normal(loc=1.0, scale=float(mc['cost_sigma']), size=n).clip(0.80, 1.25)
    rows = []
    interest_rate = float(f['base_interest_rate']) + float(f['credit_spread'])
    for i in range(n):
        fin = evaluate_finance(float(base['ticket_revenue_m'])*ticket_shock[i], float(f['other_revenue_m'])*other_shock[i],
                               float(f['player_salary_m'])*cost_shock[i], float(f['operating_cost_ratio'])*cost_shock[i],
                               float(f['starting_debt_m']), interest_rate, float(f['revenue_multiple']), float(f['ebitda_multiple']),
                               int(f['dcf_years']), float(f['dcf_growth']), float(f['dcf_discount_rate']), float(f['starting_cash_m']),
                               float(f['cash_safety_m']), float(f['preferred_leverage']), float(f['max_leverage']),
                               valuation_weights={
                                   'revenue': float(cfg.get('finance', {}).get('valuation_weight_revenue', 0.50)),
                                   'ebitda': float(cfg.get('finance', {}).get('valuation_weight_ebitda', 0.30)),
                                   'dcf': float(cfg.get('finance', {}).get('valuation_weight_dcf', 0.20)),
                               },
                               reference_interest_rate=interest_rate,
                               valuation_rate_duration=float(cfg.get('finance', {}).get('valuation_rate_duration', 0.0)))
        fin['simulation'] = i+1
        rows.append(fin)
    sims = pd.DataFrame(rows)
    safe_write_csv(sims, RESULTS/'02_business_outputs'/'monte_carlo_samples.csv')
    summary = []
    for col in ['net_profit_m','valuation_m','achieved_leverage']:
        summary.append({'metric':col, 'p10':sims[col].quantile(0.10), 'p50':sims[col].quantile(0.50), 'p90':sims[col].quantile(0.90), 'mean':sims[col].mean()})
    safe_write_csv(pd.DataFrame(summary), RESULTS/'02_business_outputs'/'monte_carlo_summary.csv')

    plt.figure(figsize=(7,4.5))
    plt.hist(sims['net_profit_m'], bins=50)
    plt.title('Monte Carlo distribution: net profit')
    plt.xlabel('Net profit, million USD')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES/'04_risk'/'monte_carlo_profit_risk.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7,4.5))
    plt.hist(sims['valuation_m'], bins=50)
    plt.title('Monte Carlo distribution: valuation')
    plt.xlabel('Valuation, million USD')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES/'04_risk'/'monte_carlo_valuation_risk.png', dpi=180)
    plt.close()
    print('[monte-carlo] wrote simulated risk distributions')

if __name__ == '__main__':
    main()
