"""Step 07: scenario analysis."""
from __future__ import annotations
import pandas as pd
from .paths import RAW, PROCESSED, RESULTS, ensure_dirs
from .step05_finance_valuation import evaluate_finance
from .utils import safe_write_csv, read_config

def main():
    ensure_dirs()
    cfg = read_config()
    f = pd.read_csv(RAW/'finance_input_assumptions.csv').iloc[0].to_dict()
    scenarios = pd.read_csv(RAW/'scenario_input_assumptions.csv')
    home = pd.read_csv(PROCESSED/'future_home_attendance_ticket_predictions.csv')
    future = pd.read_csv(PROCESSED/'future_schedule_win_probs.csv')
    base_ticket = float(home['ticket_revenue_m'].sum())
    base_reference_rate = float(f['base_interest_rate']) + float(f['credit_spread'])
    rows = []
    for _, s in scenarios.iterrows():
        interest_rate = float(f['base_interest_rate']) + float(f['credit_spread']) + float(s['interest_shift'])
        # Injury penalty lowers ticket demand and expected wins through a simple Elo-to-demand channel.
        injury_penalty = float(s['injury_elo_penalty'])
        demand_mult = max(0.70, 1.0 - injury_penalty/1000.0)
        ticket_rev = base_ticket * demand_mult
        other_rev = float(f['other_revenue_m']) * float(s['other_revenue_multiplier'])
        fin = evaluate_finance(ticket_rev, other_rev, float(f['player_salary_m']), float(f['operating_cost_ratio']), float(f['starting_debt_m']), interest_rate,
                               float(f['revenue_multiple']), float(f['ebitda_multiple']), int(f['dcf_years']), float(f['dcf_growth']),
                               float(f['dcf_discount_rate']) + max(float(s['interest_shift']), 0)*0.6,
                               float(f['starting_cash_m']), float(f['cash_safety_m']), float(f['preferred_leverage']), float(f['max_leverage']),
                               valuation_weights={
                                   'revenue': float(cfg.get('finance', {}).get('valuation_weight_revenue', 0.50)),
                                   'ebitda': float(cfg.get('finance', {}).get('valuation_weight_ebitda', 0.30)),
                                   'dcf': float(cfg.get('finance', {}).get('valuation_weight_dcf', 0.20)),
                               },
                               reference_interest_rate=base_reference_rate,
                               valuation_rate_duration=float(cfg.get('finance', {}).get('valuation_rate_duration', 0.0)))
        if str(s['scenario']) == 'key_player_injury' and 'team_win_prob_injury' in future.columns:
            wins = future['team_win_prob_injury'].sum()
        else:
            wins = future['team_win_prob'].sum()
        fin.update({'scenario':s['scenario'], 'expected_wins':wins, 'interest_rate':interest_rate})
        rows.append(fin)
    out = pd.DataFrame(rows)
    safe_write_csv(out, RESULTS/'02_business_outputs'/'scenario_analysis.csv')

    city = pd.DataFrame([
        {'city':'Toronto','market_type':'BIG','market_score':91},
        {'city':'Philadelphia','market_type':'BIG','market_score':86},
        {'city':'Portland','market_type':'MID','market_score':78},
        {'city':'Nashville','market_type':'MID','market_score':70},
        {'city':'Louisville','market_type':'SMALL','market_score':62},
    ])
    safe_write_csv(city, RESULTS/'02_business_outputs'/'expansion_candidate_cities.csv')
    print('[scenarios] wrote scenario outputs')

if __name__ == '__main__':
    main()
