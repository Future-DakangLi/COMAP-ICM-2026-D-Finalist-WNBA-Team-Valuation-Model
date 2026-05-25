from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config'
DATA = ROOT / 'data'
SEED = DATA / '00_seed'
RAW = DATA / '01_raw_downloaded'
INTERIM = DATA / '02_interim'
PROCESSED = DATA / '03_processed'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'figures'

def ensure_dirs():
    for p in [RAW, INTERIM, PROCESSED, RESULTS/'00_data_audit', RESULTS/'01_model_validation', RESULTS/'02_business_outputs',
              FIGURES/'00_eda', FIGURES/'01_elo', FIGURES/'02_attendance', FIGURES/'03_business', FIGURES/'04_risk']:
        p.mkdir(parents=True, exist_ok=True)
