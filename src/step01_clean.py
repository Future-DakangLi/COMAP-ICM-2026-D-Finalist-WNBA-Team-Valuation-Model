"""Step 01: clean game-level data and construct model features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .paths import RAW, INTERIM, RESULTS, ensure_dirs
from .utils import safe_write_csv

TEAM_FIX = {
    'Las Vegas Aces': 'Las Vegas Aces',
    'LA Sparks': 'Los Angeles Sparks',
    'Los Angeles Sparks': 'Los Angeles Sparks',
    'NY Liberty': 'New York Liberty',
    'New York Liberty': 'New York Liberty',
}

def clean_team(x):
    if pd.isna(x): return x
    s = str(x).strip()
    return TEAM_FIX.get(s, s)

def main():
    ensure_dirs()
    raw = pd.read_csv(RAW/'games_raw.csv')
    venues = pd.read_csv(RAW/'venues_raw.csv')

    df = raw.copy()
    df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
    df = df.dropna(subset=['game_date','home_team','away_team'])
    df['home_team'] = df['home_team'].map(clean_team)
    df['away_team'] = df['away_team'].map(clean_team)
    df = df[df['home_team'].ne(df['away_team'])]
    df['season'] = df['season'].fillna(df['game_date'].dt.year).astype(int)

    # Coerce numeric fields. Bad zeros and negative values become missing.
    for col in ['capacity','attendance','min_ticket_price','home_score','away_score']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] <= 0, col] = np.nan

    # Capacity repair: venue first, then team median, then league median.
    if 'capacity' not in df.columns:
        df['capacity'] = np.nan
    venue_cap = venues.groupby('venue', dropna=True)['capacity'].median().to_dict() if 'venue' in venues.columns else {}
    df['capacity_repaired'] = df['capacity']
    miss = df['capacity_repaired'].isna() & df.get('venue', pd.Series(index=df.index)).notna()
    df.loc[miss, 'capacity_repaired'] = df.loc[miss, 'venue'].map(venue_cap)
    team_med = df.groupby('home_team')['capacity_repaired'].transform('median')
    df['capacity_repaired'] = df['capacity_repaired'].fillna(team_med).fillna(df['capacity_repaired'].median())

    # Attendance fill rate and right-censor marker.
    df['attendance_clean'] = df['attendance']
    df['fill_rate'] = df['attendance_clean'] / df['capacity_repaired']
    df['fill_rate'] = df['fill_rate'].clip(lower=0, upper=1.60)
    df['censored_by_capacity'] = (df['fill_rate'] >= 0.97).astype(int)

    # Calendar features.
    df = df.sort_values(['game_date','game_id']).reset_index(drop=True)
    df['month'] = df['game_date'].dt.month
    df['day_of_week'] = df['game_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

    # Rest-day features for both teams.
    last_game = {}
    home_rest = []
    away_rest = []
    for _, row in df.iterrows():
        d = row['game_date']
        h = row['home_team']; a = row['away_team']
        hr = 10 if h not in last_game else (d - last_game[h]).days
        ar = 10 if a not in last_game else (d - last_game[a]).days
        home_rest.append(max(0, min(10, hr)))
        away_rest.append(max(0, min(10, ar)))
        last_game[h] = d; last_game[a] = d
    df['home_rest_days'] = home_rest
    df['away_rest_days'] = away_rest
    df['rest_diff'] = df['home_rest_days'] - df['away_rest_days']

    # Game outcome.
    df['home_win'] = np.where(df['home_score'].notna() & df['away_score'].notna(), (df['home_score'] > df['away_score']).astype(int), np.nan)

    keep = ['game_id','season','game_date','home_team','away_team','venue','capacity_repaired','attendance_clean','fill_rate',
            'censored_by_capacity','min_ticket_price','home_score','away_score','home_win','month','day_of_week','is_weekend',
            'home_rest_days','away_rest_days','rest_diff']
    keep = [c for c in keep if c in df.columns]
    out = df[keep].rename(columns={'capacity_repaired':'capacity','attendance_clean':'attendance'})
    safe_write_csv(out, INTERIM/'clean_game_level_dataset.csv')

    profile = pd.DataFrame([
        {'metric':'raw_rows','value':len(raw)},
        {'metric':'clean_rows','value':len(out)},
        {'metric':'rows_with_scores','value':int(out['home_win'].notna().sum())},
        {'metric':'rows_with_attendance','value':int(out['attendance'].notna().sum())},
        {'metric':'rows_with_capacity','value':int(out['capacity'].notna().sum())},
        {'metric':'seasons','value':out['season'].nunique()},
        {'metric':'teams','value':len(set(out.home_team).union(set(out.away_team)))},
    ])
    safe_write_csv(profile, RESULTS/'00_data_audit'/'cleaning_summary.csv')

    by_team = out.groupby('home_team').agg(home_games=('game_id','count'), mean_attendance=('attendance','mean'), mean_fill_rate=('fill_rate','mean')).reset_index()
    safe_write_csv(by_team, RESULTS/'00_data_audit'/'home_team_attendance_profile.csv')
    print('[clean] wrote clean game-level dataset')

if __name__ == '__main__':
    main()
