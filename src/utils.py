import json
from pathlib import Path
import numpy as np
import pandas as pd

from .paths import CONFIG

def read_config():
    return json.loads((CONFIG / 'model_config.json').read_text(encoding='utf-8'))

def safe_write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def sigmoid_elo(delta):
    return 1.0 / (1.0 + 10.0 ** (-delta / 400.0))

def brier_score(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))

def log_loss_binary(y, p, eps=1e-12):
    y = np.asarray(y, dtype=float); p = np.clip(np.asarray(p, dtype=float), eps, 1-eps)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

def accuracy_at_half(y, p):
    y = np.asarray(y, dtype=int); p = np.asarray(p, dtype=float)
    return float(np.mean((p >= 0.5).astype(int) == y))

def ridge_fit_predict(train_x, train_y, pred_x, lam=1.0):
    X = np.asarray(train_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    Z = np.asarray(pred_x, dtype=float)
    if X.ndim != 2:
        raise ValueError('train_x must be 2-D')
    penalty = np.eye(X.shape[1]) * lam
    penalty[0, 0] = 0.0  # do not penalize intercept
    beta = np.linalg.pinv(X.T @ X + penalty) @ X.T @ y
    return Z @ beta, beta

def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y)-np.asarray(pred))))

def mape(y, pred):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    mask = y != 0
    return float(np.mean(np.abs((y[mask]-pred[mask])/y[mask])))

def r2_score(y, pred):
    y = np.asarray(y, dtype=float); pred = np.asarray(pred, dtype=float)
    ss_res = np.sum((y-pred)**2)
    ss_tot = np.sum((y-y.mean())**2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

def dcf_value(fcf_m, discount_rate, growth, years):
    fcf_m = float(fcf_m)
    vals = []
    for t in range(1, int(years)+1):
        vals.append(fcf_m * ((1+growth)**t) / ((1+discount_rate)**t))
    terminal = (fcf_m * ((1+growth)**(years+1))) / max(discount_rate-growth, 1e-6)
    terminal_pv = terminal / ((1+discount_rate)**years)
    return float(sum(vals) + terminal_pv)
