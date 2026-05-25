"""Step 06: action optimization using player-acquisition buttons."""
from __future__ import annotations
import pandas as pd
from .paths import RAW, PROCESSED, RESULTS, ensure_dirs
from .utils import read_config, safe_write_csv
from .step05_finance_valuation import evaluate_finance

def main():
    ensure_dirs()
    cfg = read_config()
    f = pd.read_csv(RAW/'finance_input_assumptions.csv').iloc[0].to_dict()
    actions = pd.read_csv(RAW/'player_action_assumptions.csv')
    home = pd.read_csv(PROCESSED/'future_home_attendance_ticket_predictions.csv')
    wins = pd.read_csv(RESULTS/'01_model_validation'/'elo_expected_wins_forecast.csv')
    base_wins = float(wins.loc[wins['scenario'].eq('risk_adjusted_baseline'), 'expected_wins'].iloc[0])
    rows = []
    interest_rate = float(f['base_interest_rate']) + float(f['credit_spread'])
    base_ticket = float(home['ticket_revenue_m'].sum())
    base_att = float(home['pred_attendance'].sum())
    base_capacity = float(home['capacity'].sum())
    for _, a in actions.iterrows():
        # Translate action into model-side increments. This is an assumption table, not a result table.
        ticket_mult = 1.0 + 0.35*float(a['popularity_boost']) + 0.0025*float(a['elo_boost'])
        expected_wins = base_wins + float(a['elo_boost'])/18.0
        ticket_rev = base_ticket * ticket_mult
        other_rev = float(f['other_revenue_m']) + float(a['extra_other_revenue_m'])
        salary = float(f['player_salary_m']) + float(a['extra_salary_m'])
        fin = evaluate_finance(ticket_rev, other_rev, salary, float(f['operating_cost_ratio']), float(f['starting_debt_m']), interest_rate,
                               float(f['revenue_multiple']), float(f['ebitda_multiple']), int(f['dcf_years']), float(f['dcf_growth']),
                               float(f['dcf_discount_rate']), float(f['starting_cash_m']), float(f['cash_safety_m']), float(f['preferred_leverage']), float(f['max_leverage']),
                               valuation_weights={
                                   'revenue': float(cfg.get('finance', {}).get('valuation_weight_revenue', 0.50)),
                                   'ebitda': float(cfg.get('finance', {}).get('valuation_weight_ebitda', 0.30)),
                                   'dcf': float(cfg.get('finance', {}).get('valuation_weight_dcf', 0.20)),
                               },
                               reference_interest_rate=interest_rate,
                               valuation_rate_duration=float(cfg.get('finance', {}).get('valuation_rate_duration', 0.0)))
        avg_att_rate = min(1.0, (base_att/base_capacity) * ticket_mult)
        fin.update({'action':a['action'], 'expected_wins':expected_wins, 'avg_attendance_rate':avg_att_rate,
                    'total_home_attendance':base_att*ticket_mult, 'extra_salary_m':float(a['extra_salary_m'])})
        # Conservative score balances profit, valuation, and debt pressure.
        fin['decision_score'] = fin['net_profit_m'] + 0.015*fin['valuation_m'] - 8.0*fin['achieved_leverage']
        rows.append(fin)
    out = pd.DataFrame(rows).sort_values('decision_score', ascending=False)
    safe_write_csv(out, RESULTS/'02_business_outputs'/'decision_comparison_table.csv')
    rec = out.head(1)[['action','decision_score','net_profit_m','valuation_m','achieved_leverage']]
    safe_write_csv(rec, RESULTS/'02_business_outputs'/'recommended_strategy.csv')

    triggers = pd.DataFrame([
        {'trigger':'cash warning','condition':'ending cash <= 1.2 * safety cash','response':'lower preferred leverage and pause new debt'},
        {'trigger':'demand warning','condition':'predicted attendance rate falls for two review windows','response':'favor marketing or popularity-improving action'},
        {'trigger':'competition warning','condition':'win forecast drops materially after injury/performance news','response':'consider strength-improving action but tighten leverage'},
    ])
    safe_write_csv(triggers, RESULTS/'02_business_outputs'/'trigger_rules.csv')
    print('[optimization] wrote action comparison and recommendation')

if __name__ == '__main__':
    main()
