"""
Technical Indicators — Pure pandas/numpy implementation
ไม่ต้องพึ่ง pandas_ta หรือ ta-lib — รองรับ Python 3.9+
"""
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


# ─── Core Calculation Helpers ─────────────────────────────────────────

def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(window=length).mean()

def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast    = _ema(series, fast)
    ema_slow    = _ema(series, slow)
    macd_line   = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def _bbands(series: pd.Series, length=20, std=2):
    mid   = _sma(series, length)
    sigma = series.rolling(length).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length=14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=length - 1, min_periods=length, adjust=False).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length=14):
    prev_high = high.shift(1)
    prev_low  = low.shift(1)
    dm_plus   = (high - prev_high).clip(lower=0)
    dm_minus  = (prev_low - low).clip(lower=0)
    both_pos  = (dm_plus > 0) & (dm_minus > 0)
    larger    = dm_plus > dm_minus
    dm_plus   = dm_plus.where(~both_pos | larger, 0)
    dm_minus  = dm_minus.where(~both_pos | ~larger, 0)
    atr      = _atr(high, low, close, length)
    di_plus  = 100 * _ema(dm_plus,  length) / atr.replace(0, np.nan)
    di_minus = 100 * _ema(dm_minus, length) / atr.replace(0, np.nan)
    dx  = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = _ema(dx, length)
    return adx, di_plus, di_minus

def _stochrsi(series: pd.Series, rsi_length=14, k=3, d=3):
    rsi      = _rsi(series, rsi_length)
    rsi_low  = rsi.rolling(rsi_length).min()
    rsi_high = rsi.rolling(rsi_length).max()
    stoch    = 100 * (rsi - rsi_low) / (rsi_high - rsi_low).replace(0, np.nan)
    k_line   = stoch.rolling(k).mean()
    d_line   = k_line.rolling(d).mean()
    return k_line, d_line

def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()

def _ichimoku(high: pd.Series, low: pd.Series, close: pd.Series,
              tenkan=9, kijun=26, senkou=52):
    def midpoint(h, l, n):
        return (h.rolling(n).max() + l.rolling(n).min()) / 2
    t = midpoint(high, low, tenkan)
    k = midpoint(high, low, kijun)
    sa = ((t + k) / 2).shift(kijun)
    sb = midpoint(high, low, senkou).shift(kijun)
    ch = close.shift(-kijun)
    return t, k, sa, sb, ch


# ─── Main Indicator Builder ───────────────────────────────────────────

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA9']   = _ema(df['Close'], 9)
    df['EMA21']  = _ema(df['Close'], 21)
    df['EMA50']  = _ema(df['Close'], 50)
    df['EMA200'] = _ema(df['Close'], 200)
    df['SMA20']  = _sma(df['Close'], 20)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = _macd(df['Close'])
    df['RSI'] = _rsi(df['Close'], 14)
    df['BB_upper'], df['BB_middle'], df['BB_lower'], df['BB_width'] = _bbands(df['Close'], 20, 2)
    df['ATR'] = _atr(df['High'], df['Low'], df['Close'], 14)
    df['ADX'], df['DI_plus'], df['DI_minus'] = _adx(df['High'], df['Low'], df['Close'], 14)
    df['StochRSI_k'], df['StochRSI_d'] = _stochrsi(df['Close'])
    df['OBV'] = _obv(df['Close'], df['Volume'])
    df['Tenkan'], df['Kijun'], df['Senkou_A'], df['Senkou_B'], df['Chikou'] = \
        _ichimoku(df['High'], df['Low'], df['Close'])
    df['Vol_SMA20'] = df['Volume'].rolling(20).mean()
    df['Vol_ratio'] = df['Volume'] / df['Vol_SMA20'].replace(0, np.nan)
    df = df.dropna(subset=['EMA21', 'RSI', 'MACD'])
    return df


# ─── Support / Resistance ─────────────────────────────────────────────

def find_support_resistance(df: pd.DataFrame, window: int = 10) -> tuple:
    closes  = df['Close'].values
    current = closes[-1]
    try:
        local_min_idx = argrelextrema(closes, np.less_equal,    order=window)[0]
        local_max_idx = argrelextrema(closes, np.greater_equal, order=window)[0]
    except Exception:
        return [], []

    def cluster_levels(levels, threshold=0.01):
        if not levels:
            return []
        clustered, group = [], [levels[0]]
        for lv in levels[1:]:
            if abs(lv - group[-1]) / max(group[-1], 1e-9) <= threshold:
                group.append(lv)
            else:
                clustered.append(float(np.mean(group)))
                group = [lv]
        clustered.append(float(np.mean(group)))
        return clustered

    supports    = cluster_levels(sorted(set(closes[local_min_idx])))
    resistances = cluster_levels(sorted(set(closes[local_max_idx])))
    supports    = sorted([s for s in supports    if s < current * 1.02], key=lambda x: abs(x - current))
    resistances = sorted([r for r in resistances if r > current * 0.98], key=lambda x: abs(x - current))
    return supports[:7], resistances[:7]


# ─── Candlestick Pattern Detection ───────────────────────────────────

def detect_candlestick_patterns(df: pd.DataFrame) -> list:
    patterns = []
    if len(df) < 3:
        return patterns
    c1, c2, c3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]

    def body(c):       return abs(c['Close'] - c['Open'])
    def upper_wick(c): return c['High'] - max(c['Close'], c['Open'])
    def lower_wick(c): return min(c['Close'], c['Open']) - c['Low']
    def full_range(c): return max(c['High'] - c['Low'], 1e-9)
    def is_bull(c):    return c['Close'] > c['Open']
    def is_bear(c):    return c['Close'] < c['Open']

    if lower_wick(c3) > body(c3) * 2 and upper_wick(c3) < body(c3) * 0.5 and is_bear(c2):
        patterns.append({"pattern": "Hammer", "type": "BUY",
                          "description_th": "รูปแบบค้อน — สัญญาณกลับตัวขาขึ้น"})

    if upper_wick(c3) > body(c3) * 2 and lower_wick(c3) < body(c3) * 0.5 and is_bull(c2):
        patterns.append({"pattern": "Shooting Star", "type": "SELL",
                          "description_th": "ดาวตก — สัญญาณกลับตัวขาลง"})

    if is_bear(c2) and is_bull(c3) and c3['Open'] < c2['Close'] and c3['Close'] > c2['Open']:
        patterns.append({"pattern": "Bullish Engulfing", "type": "BUY",
                          "description_th": "แท่งกลืนกินขาขึ้น — สัญญาณซื้อแรง"})

    if is_bull(c2) and is_bear(c3) and c3['Open'] > c2['Close'] and c3['Close'] < c2['Open']:
        patterns.append({"pattern": "Bearish Engulfing", "type": "SELL",
                          "description_th": "แท่งกลืนกินขาลง — สัญญาณขายแรง"})

    if (is_bear(c1) and body(c2) < body(c1) * 0.3 and is_bull(c3) and
            c3['Close'] > (c1['Open'] + c1['Close']) / 2):
        patterns.append({"pattern": "Morning Star", "type": "BUY",
                          "description_th": "ดาวรุ่ง — สัญญาณกลับตัวขาขึ้นแรง"})

    if (is_bull(c1) and body(c2) < body(c1) * 0.3 and is_bear(c3) and
            c3['Close'] < (c1['Open'] + c1['Close']) / 2):
        patterns.append({"pattern": "Evening Star", "type": "SELL",
                          "description_th": "ดาวตอนเย็น — สัญญาณกลับตัวขาลงแรง"})

    if body(c3) < full_range(c3) * 0.1:
        patterns.append({"pattern": "Doji", "type": "NEUTRAL",
                          "description_th": "โดจิ — ความลังเลของตลาด รอยืนยัน"})

    return patterns
