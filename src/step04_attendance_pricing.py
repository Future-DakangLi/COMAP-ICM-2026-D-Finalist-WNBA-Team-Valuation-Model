"""Step 04: attendance demand model and dynamic ticket pricing."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .paths import INTERIM, PROCESSED, RESULTS, FIGURES, RAW, ensure_dirs
from .utils import read_config, ridge_fit_predict, mae, mape, r2_score, safe_write_csv

FEATURES = ['intercept','pred_home_win_prob','home_elo_pre','away_elo_pre','month','is_weekend','home_rest_days','away_rest_days','rest_diff','capacity']

def build_design(df):
    x = pd.DataFrame(index=df.index)
    x['intercept'] = 1.0
    for c in FEATURES:
        if c == 'intercept':
            continue
        if c in df.columns:
            x[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            x[c] = 0.0
    return x.fillna(x.median(numeric_only=True)).fillna(0.0)[FEATURES]

def fit_attendance_model(hist, cfg):
    d = hist.dropna(subset=['attendance','capacity','pred_home_win_prob']).copy()
    d = d[d['attendance'] >= cfg['attendance_model']['min_training_attendance']]
    d['log_attendance'] = np.log(d['attendance'].clip(lower=1))
    d = d.sort_values('game_date').reset_index(drop=True)
    split = int(len(d) * 0.80)
    train, valid = d.iloc[:split].copy(), d.iloc[split:].copy()
    pred_valid_log, beta = ridge_fit_predict(build_design(train), train['log_attendance'], build_design(valid), cfg['attendance_model']['ridge_lambda'])
    valid['pred_attendance_raw'] = np.exp(pred_valid_log)
    valid['pred_attendance'] = np.minimum(valid['pred_attendance_raw'], valid['capacity'])
    metrics = pd.DataFrame([{
        'n_train': len(train),
        'n_validation': len(valid),
        'mae': mae(valid['attendance'], valid['pred_attendance']),
        'mape': mape(valid['attendance'], valid['pred_attendance']),
        'r2': r2_score(valid['attendance'], valid['pred_attendance']),
        'ridge_lambda': cfg['attendance_model']['ridge_lambda'],
    }])
    coefs = pd.DataFrame({'feature':FEATURES, 'coefficient':beta})
    return train, valid, beta, metrics, coefs

def predict_future(home_sched, beta, cfg):
    X = build_design(home_sched)
    home_sched = home_sched.copy()
    home_sched['pred_attendance_raw'] = np.exp(np.asarray(X) @ beta)
    home_sched['pred_attendance'] = np.minimum(home_sched['pred_attendance_raw'], home_sched['capacity'])
    home_sched['pred_fill_rate'] = home_sched['pred_attendance'] / home_sched['capacity']

    tp = cfg['ticket_pricing']
    # SeatGeek-style minimum listed prices are usually lower than realized average
    # per-seat yield because they exclude better seats, fees, and season/suite mixes.
    # We convert minimum listings to a conservative average ticket yield using a
    # transparent assumption in config; it is not scaled to any downstream revenue output.
    min_to_avg = float(tp.get('min_to_average_price_multiplier', 1.8))
    base_price = home_sched['min_ticket_price'].median() * min_to_avg if 'min_ticket_price' in home_sched.columns and home_sched['min_ticket_price'].notna().any() else tp['fallback_base_price']
    def rate(fill):
        if fill >= 0.98: return tp['near_sellout_markup']
        if fill >= 0.90: return tp['high_fill_markup']
        if fill <= 0.65: return tp['low_fill_discount']
        return 1.0
    home_sched['pricing_multiplier'] = home_sched['pred_fill_rate'].map(rate)
    home_sched['avg_ticket_price'] = base_price * home_sched['pricing_multiplier']
    home_sched['ticket_revenue_m'] = home_sched['pred_attendance'] * home_sched['avg_ticket_price'] / 1_000_000
    home_sched['cumulative_ticket_revenue_m'] = home_sched['ticket_revenue_m'].cumsum()
    return home_sched

def main():
    ensure_dirs()
    cfg = read_config()
    clean = pd.read_csv(INTERIM/'clean_game_level_dataset.csv', parse_dates=['game_date'])
    elo = pd.read_csv(PROCESSED/'elo_game_predictions_historical.csv', parse_dates=['game_date'])
    cols = ['game_id','pred_home_win_prob','home_elo_pre','away_elo_pre']
    hist = clean.merge(elo[cols], on='game_id', how='left')
    train, valid, beta, metrics, coefs = fit_attendance_model(hist, cfg)
    safe_write_csv(valid, PROCESSED/'attendance_predictions_validation.csv')
    safe_write_csv(metrics, RESULTS/'01_model_validation'/'attendance_validation_metrics.csv')
    safe_write_csv(coefs, RESULTS/'01_model_validation'/'attendance_model_coefficients.csv')

    future = pd.read_csv(PROCESSED/'future_schedule_win_probs.csv', parse_dates=['game_date'])
    # Add design columns for future games using Elo fields and clean schedule features.
    for c in ['home_elo_pre','away_elo_pre']:
        if c not in future.columns:
            future[c] = 1500.0
    future['month'] = future['game_date'].dt.month
    future['day_of_week'] = future['game_date'].dt.dayofweek
    future['is_weekend'] = future['day_of_week'].isin([5,6]).astype(int)
    future['home_rest_days'] = 4
    future['away_rest_days'] = 4
    future['rest_diff'] = 0
    # Use team probability as a popularity/strength proxy when the team is home.
    future['pred_home_win_prob'] = future['pred_home_win_prob'].fillna(0.5)
    team = cfg['team_of_interest']
    home = future[future['home_team'] == team].copy()
    home_pred = predict_future(home, beta, cfg)
    safe_write_csv(home_pred, PROCESSED/'future_home_attendance_ticket_predictions.csv')

    # Plots.
    plt.figure(figsize=(6,5))
    plt.scatter(valid['attendance'], valid['pred_attendance'], s=12, alpha=0.4)
    mx = max(valid['attendance'].max(), valid['pred_attendance'].max())
    plt.plot([0,mx],[0,mx], linestyle='--')
    plt.title('Actual vs predicted attendance')
    plt.xlabel('Actual attendance')
    plt.ylabel('Predicted attendance')
    plt.tight_layout()
    plt.savefig(FIGURES/'02_attendance'/'actual_vs_predicted_attendance.png', dpi=180)
    plt.close()

    plt.figure(figsize=(8,4.5))
    labels = pd.to_datetime(home_pred['game_date']).dt.strftime('%m-%d')
    plt.bar(range(len(home_pred)), home_pred['ticket_revenue_m'])
    plt.plot(range(len(home_pred)), home_pred['cumulative_ticket_revenue_m'], marker='o')
    plt.xticks(range(len(home_pred)), labels, rotation=60, ha='right')
    plt.title(f'{team} future home games: ticket revenue')
    plt.ylabel('Revenue, million USD')
    plt.tight_layout()
    plt.savefig(FIGURES/'02_attendance'/'future_home_ticket_revenue.png', dpi=180)
    plt.close()
    print('[attendance] wrote demand/pricing outputs')

if __name__ == '__main__':
    main()
