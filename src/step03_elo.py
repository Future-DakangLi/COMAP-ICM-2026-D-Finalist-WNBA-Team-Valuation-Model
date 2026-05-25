"""Step 03: rolling Elo model and season forecast."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .paths import RAW, INTERIM, PROCESSED, RESULTS, FIGURES, ensure_dirs
from .utils import read_config, sigmoid_elo, brier_score, log_loss_binary, accuracy_at_half, safe_write_csv

def run_elo(df, initial=1500.0, k=20.0, home_adv=45.0):
    teams = sorted(set(df['home_team']).union(set(df['away_team'])))
    ratings = {t: initial for t in teams}
    rows = []
    for _, r in df.sort_values(['game_date','game_id']).iterrows():
        h, a = r['home_team'], r['away_team']
        rh, ra = ratings.get(h, initial), ratings.get(a, initial)
        p_home = sigmoid_elo((rh + home_adv) - ra)
        y = r.get('home_win', np.nan)
        row = r.to_dict()
        row.update({'home_elo_pre':rh, 'away_elo_pre':ra, 'pred_home_win_prob':p_home})
        if pd.notna(y):
            delta = k * (float(y) - p_home)
            rh2, ra2 = rh + delta, ra - delta
            ratings[h] = rh2; ratings[a] = ra2
            row.update({'home_elo_post':rh2, 'away_elo_post':ra2})
        else:
            row.update({'home_elo_post':rh, 'away_elo_post':ra})
        rows.append(row)
    return pd.DataFrame(rows), ratings

def forecast_schedule(future, ratings, cfg, elo_penalty=0.0):
    team = cfg['team_of_interest']
    elo_cfg = cfg['elo']
    initial = elo_cfg['initial_rating']
    home_adv = elo_cfg['home_advantage']
    out = future.copy()
    probs = []
    team_probs = []
    for _, r in out.iterrows():
        h, a = r['home_team'], r['away_team']
        rh = ratings.get(h, initial)
        ra = ratings.get(a, initial)
        if h == team:
            rh -= elo_penalty
        if a == team:
            ra -= elo_penalty
        p_home = sigmoid_elo((rh + home_adv) - ra)
        probs.append(p_home)
        team_probs.append(p_home if h == team else 1-p_home if a == team else np.nan)
    out['pred_home_win_prob'] = probs
    out['team_win_prob'] = team_probs
    return out

def main():
    ensure_dirs()
    cfg = read_config()
    df = pd.read_csv(INTERIM/'clean_game_level_dataset.csv', parse_dates=['game_date'])
    games_with_scores = df.dropna(subset=['home_win']).copy()
    elo_df, ratings = run_elo(games_with_scores, **{
        'initial': cfg['elo']['initial_rating'],
        'k': cfg['elo']['k_factor'],
        'home_adv': cfg['elo']['home_advantage'],
    })
    safe_write_csv(elo_df, PROCESSED/'elo_game_predictions_historical.csv')

    y = elo_df['home_win'].astype(int)
    p = elo_df['pred_home_win_prob']
    metrics = pd.DataFrame([{
        'n_games': len(elo_df),
        'brier_score': brier_score(y, p),
        'log_loss': log_loss_binary(y, p),
        'accuracy_p_gt_0_5': accuracy_at_half(y, p),
        'initial_rating': cfg['elo']['initial_rating'],
        'k_factor': cfg['elo']['k_factor'],
        'home_advantage': cfg['elo']['home_advantage'],
    }])
    safe_write_csv(metrics, RESULTS/'01_model_validation'/'elo_validation_metrics.csv')

    # Calibration bins.
    tmp = elo_df[['home_win','pred_home_win_prob']].copy()
    tmp['bin'] = pd.cut(tmp['pred_home_win_prob'], bins=np.linspace(0,1,11), include_lowest=True)
    cal = tmp.groupby('bin', observed=True).agg(n=('home_win','size'), pred_mean=('pred_home_win_prob','mean'), actual_win_rate=('home_win','mean')).reset_index()
    safe_write_csv(cal.astype({'bin':str}), RESULTS/'01_model_validation'/'elo_calibration_bins.csv')

    # Future forecast from final historical ratings.
    #
    # Clean-improvement version:
    #   1) report the raw Elo point forecast;
    #   2) use a conservative, roster-uncertainty-adjusted baseline for business decisions;
    #   3) apply an extra key-player absence shock for the injury case.
    #
    # These penalties are model assumptions in config/model_config.json.  They are not
    # learned from, nor scaled to, any downstream output table.
    future = pd.read_csv(RAW/'future_schedule_raw.csv', parse_dates=['game_date'])
    raw_point = forecast_schedule(future, ratings, cfg, elo_penalty=0.0)
    roster_penalty = float(cfg['elo'].get('roster_uncertainty_elo_penalty', 0.0))
    injury_extra = float(cfg['elo'].get('key_player_injury_extra_penalty', cfg['elo'].get('injury_elo_penalty', 0.0)))
    future_base = forecast_schedule(future, ratings, cfg, elo_penalty=roster_penalty)
    future_inj = forecast_schedule(future, ratings, cfg, elo_penalty=roster_penalty + injury_extra)
    future_base['team_win_prob_raw_elo'] = raw_point['team_win_prob']
    future_base['team_win_prob_injury'] = future_inj['team_win_prob']
    future_base['forecast_roster_penalty'] = roster_penalty
    future_base['forecast_injury_extra_penalty'] = injury_extra
    safe_write_csv(future_base, PROCESSED/'future_schedule_win_probs.csv')
    wins = pd.DataFrame([
        {'scenario':'raw_elo_point_forecast', 'season_games':raw_point['team_win_prob'].notna().sum(), 'expected_wins':raw_point['team_win_prob'].sum()},
        {'scenario':'risk_adjusted_baseline', 'season_games':future_base['team_win_prob'].notna().sum(), 'expected_wins':future_base['team_win_prob'].sum()},
        {'scenario':'key_player_injury', 'season_games':future_base['team_win_prob_injury'].notna().sum(), 'expected_wins':future_base['team_win_prob_injury'].sum()},
    ])
    safe_write_csv(wins, RESULTS/'01_model_validation'/'elo_expected_wins_forecast.csv')

    # Plots.
    team = cfg['team_of_interest']
    fever = elo_df[(elo_df.home_team == team) | (elo_df.away_team == team)].copy()
    fever['team_elo_post'] = np.where(fever.home_team == team, fever.home_elo_post, fever.away_elo_post)
    plt.figure(figsize=(8,4.5))
    plt.plot(pd.to_datetime(fever['game_date']), fever['team_elo_post'])
    plt.title(f'Elo rating trajectory: {team}')
    plt.xlabel('Game date')
    plt.ylabel('Elo rating')
    plt.tight_layout()
    plt.savefig(FIGURES/'01_elo'/'elo_rating_trajectory_indiana_fever.png', dpi=180)
    plt.close()

    plt.figure(figsize=(6,5))
    if len(cal):
        plt.scatter(cal['pred_mean'], cal['actual_win_rate'], s=cal['n']*3, alpha=0.6)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.title('Elo probability calibration')
    plt.xlabel('Predicted home win probability')
    plt.ylabel('Actual home win rate')
    plt.tight_layout()
    plt.savefig(FIGURES/'01_elo'/'elo_probability_calibration.png', dpi=180)
    plt.close()
    print('[elo] wrote Elo predictions and validation')

if __name__ == '__main__':
    main()
