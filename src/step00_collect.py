"""Step 00: collect or stage data.

This script is designed to start from public source URLs when the user has internet.
For offline reproducibility, the repository also carries a raw snapshot in data/00_seed.
No known final-output table is read here; this step only stages raw/source-like inputs.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path
import pandas as pd

from .paths import RAW, SEED, RESULTS, ensure_dirs
from .utils import safe_write_csv

PUBLIC_URLS = {
    'wehoop_schedule_master': 'https://raw.githubusercontent.com/sportsdataverse/wehoop-data/main/wnba_schedule_master.csv',
    'wehoop_stats_schedule': 'https://raw.githubusercontent.com/sportsdataverse/wehoop-data/main/wnba_stats_schedule_master.csv',
    'wehoop_games_list': 'https://raw.githubusercontent.com/sportsdataverse/wehoop-data/main/wnba/wnba_games_in_data_repo.csv',
    'fred_fed_funds': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF',
    'fred_high_yield_spread': 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2',
}

def try_download_csv(url: str, out: Path) -> bool:
    try:
        df = pd.read_csv(url)
        if len(df) == 0:
            return False
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        return True
    except Exception as exc:
        print(f'[collect] Could not download {url}: {exc}')
        return False

def copy_seed(name: str, dst_name: str):
    src = SEED / name
    dst = RAW / dst_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    return dst

def main():
    ensure_dirs()
    online = os.getenv('ONLINE_COLLECTION', '0').strip() == '1'
    audit_rows = []

    if online:
        for name, url in PUBLIC_URLS.items():
            ok = try_download_csv(url, RAW / f'{name}.csv')
            audit_rows.append({'dataset': name, 'method': 'download', 'success': ok, 'path': str(RAW / f'{name}.csv')})

    # Raw-like snapshots make the repository runnable in places where web access is blocked.
    # They are NOT final-output tables; downstream numbers are recomputed from them.
    staged = [
        ('offline_snapshot_games.csv', 'games_raw.csv'),
        ('offline_snapshot_venues.csv', 'venues_raw.csv'),
        ('offline_future_schedule.csv', 'future_schedule_raw.csv'),
        ('finance_input_assumptions.csv', 'finance_input_assumptions.csv'),
        ('player_action_assumptions.csv', 'player_action_assumptions.csv'),
        ('scenario_input_assumptions.csv', 'scenario_input_assumptions.csv'),
        ('source_manifest.csv', 'source_manifest.csv'),
    ]
    for src, dst in staged:
        out = copy_seed(src, dst)
        audit_rows.append({'dataset': dst, 'method': 'offline_source_snapshot', 'success': True, 'path': str(out)})

    safe_write_csv(pd.DataFrame(audit_rows), RESULTS/'00_data_audit'/'collection_audit.csv')
    print('[collect] staged raw inputs in data/01_raw_downloaded')

if __name__ == '__main__':
    main()
