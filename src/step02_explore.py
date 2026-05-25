"""Step 02: exploratory plots."""
from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
from .paths import INTERIM, FIGURES, ensure_dirs


def main():
    ensure_dirs()
    df = pd.read_csv(INTERIM/'clean_game_level_dataset.csv', parse_dates=['game_date'])

    # Attendance by season.
    by = df.dropna(subset=['attendance']).groupby('season')['attendance'].mean()
    plt.figure(figsize=(8,4.5))
    plt.plot(by.index, by.values, marker='o')
    plt.title('Average home attendance by season')
    plt.xlabel('Season')
    plt.ylabel('Average attendance')
    plt.tight_layout()
    plt.savefig(FIGURES/'00_eda'/'attendance_by_season.png', dpi=180)
    plt.close()

    # Fill rate distribution.
    plt.figure(figsize=(7,4.5))
    plt.hist(df['fill_rate'].dropna().clip(0,1.5), bins=40)
    plt.axvline(0.97, linestyle='--')
    plt.title('Fill-rate distribution')
    plt.xlabel('Attendance / capacity')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES/'00_eda'/'fill_rate_distribution.png', dpi=180)
    plt.close()

    # Capacity vs fill-rate scatter.
    d = df.dropna(subset=['capacity','fill_rate'])
    plt.figure(figsize=(7,4.8))
    plt.scatter(d['capacity'], d['fill_rate'], s=10, alpha=0.35)
    plt.axhline(0.97, linestyle='--')
    plt.title('Capacity vs fill-rate')
    plt.xlabel('Venue capacity')
    plt.ylabel('Fill rate')
    plt.tight_layout()
    plt.savefig(FIGURES/'00_eda'/'capacity_fill_rate_scatter.png', dpi=180)
    plt.close()
    print('[explore] wrote EDA figures')

if __name__ == '__main__':
    main()
