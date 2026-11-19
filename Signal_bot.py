
"""
Generador de señales con ML para opciones binarias (solo SEÑALES, no opera).
Pensado para ejecutarse en la nube (GitHub Actions) y avisar al iPhone por Telegram.

Filosofía: calidad > cantidad. El bot solo emite una señal si, en validación
walk-forward (datos que el modelo nunca vio), la precisión con ese umbral de
confianza supera de forma estadísticamente fiable el punto de equilibrio del
payout. Si no hay ventaja demostrable, NO emite señales.
"""
import argparse
import math
import os
import time

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

# ---------------- Configuración ----------------
SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]  # mercado real, no OTC
INTERVAL = "5m"
PERIOD = "60d"
HORIZON = 1              # expiración = 1 vela (5 min)
PAYOUT = 0.85            # beneficio si aciertas (mira el de tu activo en la app)
MIN_TRADES = 150
THRESHOLDS = [0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.70]
RETRAIN_EVERY_H = 6
MAX_AGE_S = 240          # no avisar si la vela cerró hace más de 4 min (señal vieja)
CACHE_FILE = "cache/bundles.joblib"

BREAKEVEN = 1.0 / (1.0 + PAYOUT)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
STEP = pd.Timedelta(minutes=5)


# ---------------- Datos e indicadores ----------------
def download(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, interval=INTERVAL, period=PERIOD,
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, l = df["Close"], df["High"], df["Low"]
    f = pd.DataFrame(index=df.index)
    for k in (1, 3, 6, 12):
        f[f"ret{k}"] = c.pct_change(k)
    f["rsi14"] = rsi(c, 14)
    f["rsi7"] = rsi(c, 7)
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / c
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - ma20) / (2 * sd20.replace(0, np.nan))
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    f["atr_pct"] = tr.rolling(14).mean() / c
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    f["stoch"] = (c - ll) / (hh - ll).replace(0, np.nan)
    f["ema9_21"] = (c.ewm(span=9).mean() - c.ewm(span=21).mean()) / c
    f["body"] = (c - df["Open"]) / c
    hour = df.index.hour + df.index.minute / 60
    f["hsin"] = np.sin(2 * np.pi * hour / 24)
    f["hcos"] = np.cos(2 * np.pi * hour / 24)
    return f.replace([np.inf, -np.inf], np.nan)


def make_dataset(df: pd.DataFrame):
    X = add_features(df)
    fut = df["Close"].shift(-HORIZON) - df["Close"]
    y = (fut > 0).astype(int)
    data = X.join(y.rename("y")).dropna()
    data = data[fut.reindex(data.index) != 0]
    return data.drop(columns="y"), data["y"]


# ---------------- Modelo y validación ----------------
def new_model():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200,
        l2_regularization=1.0, min_samples_leaf=60, random_state=42)


def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / den


def walk_forward(X: pd.DataFrame, y: pd.Series):
    oof = pd.Series(np.nan, index=X.index)
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        m = new_model().fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof.dropna()


def pick_threshold(proba: pd.Series, y: pd.Series):
    best = None
    yy = y.reindex(proba.index)
    for thr in THRESHOLDS:
        up, dn = proba >= thr, proba <= 1 - thr
        n = int(up.sum() + dn.sum())
        wins = int(((yy == 1) & up).sum() + ((yy == 0) & dn).sum())
        if n < MIN_TRADES:
            continue
        lb = wilson_lower(wins, n)
        stats = dict(thr=thr, n=n, winrate=wins / n, lb=lb)
        if lb > BREAKEVEN and (best is None or lb > best["lb"]):
            best = stats
    return best


def train_symbol(symbol: str):
    df = download(symbol)
    if len(df) < 2000:
        return None
    X, y = make_dataset(df)
    proba = walk_forward(X, y)
    stats = pick_threshold(proba, y)
    model = new_model().fit(X, y)
    return dict(model=model, stats=stats)


def load_or_train():
    """Reutiliza modelos guardados (menos de RETRAIN_EVERY_H horas) o reentrena."""
    if os.path.exists(CACHE_FILE):
        try:
            data = joblib.load(CACHE_FILE)
            if time.time() - data["trained_at"] < RETRAIN_EVERY_H * 3600:
                return data["bundles"]
        except Exception as e:  # caché corrupta -> reentrenar
            print("Caché no válida:", e)

    print("Entrenando y validando modelos (walk-forward)...", flush=True)
    bundles = {}
    for s in SYMBOLS:
        b = train_symbol(s)
        if b is None:
            print(f"{s}: datos insuficientes (¿mercado cerrado?)")
            continue
        bundles[s] = b
        st = b["stats"]
        if st:
            print(f"{s}: VENTAJA validada thr={st['thr']} winrate={st['winrate']:.1%} "
                  f"(cota inf. {st['lb']:.1%}, n={st['n']}, equilibrio {BREAKEVEN:.1%})")
        else:
            print(f"{s}: sin ventaja demostrable -> no se emitirán señales")
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    joblib.dump(dict(trained_at=time.time(), bundles=bundles), CACHE_FILE)
    return bundles


# ---------------- Señales y notificación ----------------
def notify(text: str):
    print(text, flush=True)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                data={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        except requests.RequestException as e:
            print("Error Telegram:", e)
    else:
        print("(Telegram no configurado: faltan TELEGRAM_TOKEN / TELEGRAM_CHAT_ID)")


def current_signal(symbol: str, bundle: dict):
    stats = bundle["stats"]
    if stats is None:
        return None
    df = download(symbol)
    if df.empty:
        return None
    now = pd.Timestamp.now(tz="UTC")
    if df.index[-1] + STEP > now:      # la última vela aún se está formando
        df = df.iloc[:-1]
    if df.empty:
        return None
    age = now - (df.index[-1] + STEP)
    if age > pd.Timedelta(seconds=MAX_AGE_S):
        return None                    # mercado cerrado o señal ya vieja
    feats = add_features(df).iloc[[-1]]
    if feats.isna().any(axis=1).iloc[0]:
        return None
    p_up = float(bundle["model"].predict_proba(feats)[0, 1])
    thr = stats["thr"]
    if p_up >= thr:
        side = "CALL (SUBE)"
    elif p_up <= 1 - thr:
        side = "PUT (BAJA)"
    else:
        return None
    return dict(side=side, conf=max(p_up, 1 - p_up), last=df.index[-1], stats=stats)


def run_once():
    bundles = load_or_train()
    for s, b in bundles.items():
        sig = current_signal(s, b)
        if sig:
            notify(
                f"SEÑAL {s.replace('=X', '')}\n{sig['side']}\n"
                f"Expiración: {HORIZON * 5} min\n"
                f"Confianza modelo: {sig['conf']:.0%}\n"
                f"Winrate validado: {sig['stats']['winrate']:.1%} (n={sig['stats']['n']})\n"
                f"Vela cerrada: {sig['last'].strftime('%H:%M')} UTC\n"
                f"Entra ya o descarta: la señal caduca en pocos minutos.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="una pasada y salir (nube)")
    ap.add_argument("--test", action="store_true", help="enviar mensaje de prueba")
    args = ap.parse_args()

    if args.test:
        notify("Prueba OK: tu bot de señales está conectado a Telegram.")
        return
    if args.once:
        run_once()
        return
    while True:                         # modo local continuo
        run_once()
        time.sleep(300 - (time.time() % 300) + 20)


if __name__ == "__main__":
    main()
