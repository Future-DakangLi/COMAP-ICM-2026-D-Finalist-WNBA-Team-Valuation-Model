"""Step 05: finance, valuation, and leverage module."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .paths import RAW, PROCESSED, RESULTS, ensure_dirs
from .utils import dcf_value, safe_write_csv, read_config

def evaluate_finance(ticket_revenue_m, other_revenue_m, salary_m, operating_cost_ratio, debt_m, interest_rate, revenue_multiple, ebitda_multiple, dcf_years, dcf_growth, dcf_discount_rate, starting_cash_m, cash_safety_m, preferred_leverage, max_leverage, valuation_weights=None, reference_interest_rate=None, valuation_rate_duration=0.0):
    total_revenue_m = ticket_revenue_m + other_revenue_m
    operating_cost_m = operating_cost_ratio * total_revenue_m
    ebitda_m = total_revenue_m - salary_m - operating_cost_m
    interest_m = debt_m * interest_rate
    net_profit_m = ebitda_m - interest_m
    # Interest-rate-sensitive franchise multiple.  A higher financing rate lowers
    # the present value of franchise revenues; the duration parameter is an
    # economic assumption and is not fit to any downstream output table.
    if reference_interest_rate is None:
        reference_interest_rate = interest_rate
    rate_discount = float(np.exp(-float(valuation_rate_duration) * (float(interest_rate) - float(reference_interest_rate))))
    revenue_value_m = revenue_multiple * total_revenue_m * rate_discount
    ebitda_value_m = ebitda_multiple * ebitda_m * rate_discount
    dcf_m = dcf_value(max(net_profit_m, 0.01), dcf_discount_rate, dcf_growth, int(dcf_years))
    # Sports-team transactions are often quoted primarily on revenue/franchise-value
    # multiples, while EBITDA and DCF are useful sanity checks.  The weights are
    # configurable assumptions, not fitted to downstream outputs.
    if valuation_weights is None:
        valuation_weights = {'revenue': 0.50, 'ebitda': 0.30, 'dcf': 0.20}
    valuation_m = (
        float(valuation_weights.get('revenue', 0.50)) * revenue_value_m
        + float(valuation_weights.get('ebitda', 0.30)) * max(ebitda_value_m, 0)
        + float(valuation_weights.get('dcf', 0.20)) * dcf_m
    )
    desired_debt_m = min(preferred_leverage * valuation_m, max_leverage * valuation_m)
    # Borrow/repay toward preferred leverage while keeping a cash cushion.
    cash_before_debt_action = starting_cash_m + net_profit_m
    if desired_debt_m > debt_m:
        new_debt_m = desired_debt_m
        ending_cash_m = cash_before_debt_action + (new_debt_m - debt_m)
    else:
        repay_possible = max(cash_before_debt_action - cash_safety_m, 0)
        repay = min(debt_m - desired_debt_m, repay_possible)
        new_debt_m = debt_m - repay
        ending_cash_m = cash_before_debt_action - repay
    achieved_leverage = new_debt_m / valuation_m if valuation_m > 0 else np.nan
    return {
        'ticket_revenue_m': ticket_revenue_m,
        'other_revenue_m': other_revenue_m,
        'total_revenue_m': total_revenue_m,
        'salary_m': salary_m,
        'operating_cost_m': operating_cost_m,
        'ebitda_m': ebitda_m,
        'interest_m': interest_m,
        'net_profit_m': net_profit_m,
        'revenue_multiple_value_m': revenue_value_m,
        'ebitda_multiple_value_m': ebitda_value_m,
        'dcf_value_m': dcf_m,
        'valuation_m': valuation_m,
        'end_debt_m': new_debt_m,
        'ending_cash_m': ending_cash_m,
        'achieved_leverage': achieved_leverage,
        'preferred_leverage': preferred_leverage,
        'valuation_rate_discount': rate_discount,
    }

def main():
    ensure_dirs()
    cfg = read_config()
    f = pd.read_csv(RAW/'finance_input_assumptions.csv').iloc[0].to_dict()
    home = pd.read_csv(PROCESSED/'future_home_attendance_ticket_predictions.csv')
    ticket_revenue_m = float(home['ticket_revenue_m'].sum())
    interest_rate = float(f['base_interest_rate']) + float(f['credit_spread'])
    out = evaluate_finance(
        ticket_revenue_m=ticket_revenue_m,
        other_revenue_m=float(f['other_revenue_m']),
        salary_m=float(f['player_salary_m']),
        operating_cost_ratio=float(f['operating_cost_ratio']),
        debt_m=float(f['starting_debt_m']),
        interest_rate=interest_rate,
        revenue_multiple=float(f['revenue_multiple']),
        ebitda_multiple=float(f['ebitda_multiple']),
        dcf_years=int(f['dcf_years']),
        dcf_growth=float(f['dcf_growth']),
        dcf_discount_rate=float(f['dcf_discount_rate']),
        starting_cash_m=float(f['starting_cash_m']),
        cash_safety_m=float(f['cash_safety_m']),
        preferred_leverage=float(f['preferred_leverage']),
        max_leverage=float(f['max_leverage']),
        valuation_weights={
            'revenue': float(cfg.get('finance', {}).get('valuation_weight_revenue', 0.50)),
            'ebitda': float(cfg.get('finance', {}).get('valuation_weight_ebitda', 0.30)),
            'dcf': float(cfg.get('finance', {}).get('valuation_weight_dcf', 0.20)),
        },
        reference_interest_rate=interest_rate,
        valuation_rate_duration=float(cfg.get('finance', {}).get('valuation_rate_duration', 0.0)),
    )
    safe_write_csv(pd.DataFrame([out]), RESULTS/'02_business_outputs'/'baseline_finance_summary.csv')
    val = pd.DataFrame([
        {'method':'revenue_multiple','valuation_m':out['revenue_multiple_value_m']},
        {'method':'ebitda_multiple','valuation_m':out['ebitda_multiple_value_m']},
        {'method':'simplified_dcf','valuation_m':out['dcf_value_m']},
        {'method':'weighted_final','valuation_m':out['valuation_m']},
    ])
    safe_write_csv(val, RESULTS/'02_business_outputs'/'valuation_methods.csv')
    print('[finance] wrote baseline finance and valuation')

if __name__ == '__main__':
    main()
