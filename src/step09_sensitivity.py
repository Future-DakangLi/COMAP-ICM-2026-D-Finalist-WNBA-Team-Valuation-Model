"""Step 09: one-factor sensitivity analysis."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from .paths import RAW, RESULTS, FIGURES, ensure_dirs
from .step05_finance_valuation import evaluate_finance
from .utils import safe_write_csv, read_config

def main():
    ensure_dirs()
    cfg = read_config()
    f = pd.read_csv(RAW/'finance_input_assumptions.csv').iloc[0].to_dict()
    base = pd.read_csv(RESULTS/'02_business_outputs'/'baseline_finance_summary.csv').iloc[0].to_dict()
    base_reference_rate = float(f['base_interest_rate']) + float(f['credit_spread'])
    tests = []
    factors = [
        ('ticket_revenue_-20pct', 0.80, 1.00, 1.00, 0.0),
        ('ticket_revenue_+20pct', 1.20, 1.00, 1.00, 0.0),
        ('other_revenue_-15pct', 1.00, 0.85, 1.00, 0.0),
        ('other_revenue_+15pct', 1.00, 1.15, 1.00, 0.0),
        ('salary_cost_+15pct', 1.00, 1.00, 1.15, 0.0),
        ('salary_cost_-10pct', 1.00, 1.00, 0.90, 0.0),
        ('interest_rate_+1pct', 1.00, 1.00, 1.00, 0.01),
        ('interest_rate_-1pct', 1.00, 1.00, 1.00, -0.01),
    ]
    for name, t_mult, o_mult, s_mult, r_shift in factors:
        fin = evaluate_finance(float(base['ticket_revenue_m'])*t_mult, float(f['other_revenue_m'])*o_mult, float(f['player_salary_m'])*s_mult,
                               float(f['operating_cost_ratio']), float(f['starting_debt_m']), float(f['base_interest_rate'])+float(f['credit_spread'])+r_shift,
                               float(f['revenue_multiple']), float(f['ebitda_multiple']), int(f['dcf_years']), float(f['dcf_growth']), float(f['dcf_discount_rate'])+max(r_shift,0)*0.6,
                               float(f['starting_cash_m']), float(f['cash_safety_m']), float(f['preferred_leverage']), float(f['max_leverage']),
                               valuation_weights={
                                   'revenue': float(cfg.get('finance', {}).get('valuation_weight_revenue', 0.50)),
                                   'ebitda': float(cfg.get('finance', {}).get('valuation_weight_ebitda', 0.30)),
                                   'dcf': float(cfg.get('finance', {}).get('valuation_weight_dcf', 0.20)),
                               },
                               reference_interest_rate=base_reference_rate,
                               valuation_rate_duration=float(cfg.get('finance', {}).get('valuation_rate_duration', 0.0)))
        tests.append({'factor':name, 'net_profit_m':fin['net_profit_m'], 'valuation_m':fin['valuation_m'],
                      'profit_delta_m':fin['net_profit_m']-float(base['net_profit_m']), 'valuation_delta_m':fin['valuation_m']-float(base['valuation_m'])})
    out = pd.DataFrame(tests)
    safe_write_csv(out, RESULTS/'02_business_outputs'/'sensitivity_analysis.csv')

    plt.figure(figsize=(8,4.8))
    plt.barh(out['factor'], out['profit_delta_m'])
    plt.title('One-factor sensitivity: profit')
    plt.xlabel('Change in net profit, million USD')
    plt.tight_layout()
    plt.savefig(FIGURES/'04_risk'/'one_factor_sensitivity_profit.png', dpi=180)
    plt.close()

    plt.figure(figsize=(8,4.8))
    plt.barh(out['factor'], out['valuation_delta_m'])
    plt.title('One-factor sensitivity: valuation')
    plt.xlabel('Change in valuation, million USD')
    plt.tight_layout()
    plt.savefig(FIGURES/'04_risk'/'one_factor_sensitivity_valuation.png', dpi=180)
    plt.close()
    print('[sensitivity] wrote sensitivity analysis')

if __name__ == '__main__':
    main()
