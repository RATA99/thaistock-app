"""
Fibonacci Scanner — สแกนหุ้น SET ที่น่าลงทุนตามหลัก Fibonacci
"""
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List
from modules.indicators import add_all_indicators
from modules.signals import calculate_signal_score

SET_UNIVERSE = [
    "PTT","ADVANC","SCB","KBANK","KTB","BBL","BAY","AOT","CPALL","SCC",
    "GULF","GPSC","PTTEP","PTTGC","RATCH","BGRIM","EGCO","BANPU","EA",
    "TOP","IRPC","BCP","SPRC",
    "MTC","SAWAD","TIDLOR","TTB","TISCO","KKP","TCAP","AEONTS","JMT","JMART",
    "TRUE","DELTA","HANA","KCE","INTUCH","INSET","THCOM",
    "BH","BDMS","BCH","CHG","PRINC",
    "HMPRO","COM7","BJC","GLOBAL","MAKRO","CRC","CPAXTRA","DOHOME",
    "OSP","TU","CPF","GFPT","OISHI","ICHI",
    "CPN","LH","SPALI","QH","AP","SC","SIRI","ORI","WHA","AMATA",
    "MINT","CENTEL","ERW","MAJOR","VGI","BEC","RS",
    "IVL","STA","STGT","TTA","PSL","THAI","AAV","BA",
]


def _fib_zone_label(zone_mid: float) -> str:
    zones = {
        0.118: "0.0–23.6%",
        0.309: "23.6–38.2%",
        0.441: "38.2–50.0% ✨",
        0.559: "50.0–61.8% 🌟",
        0.702: "61.8–78.6%",
        0.893: "78.6–100%",
    }
    for mid, label in zones.items():
        if zone_mid <= mid + 0.05:
            return label
    return "นอกช่วง"


def _scan_one(symbol: str, period: str = "1y") -> Optional[dict]:
    """สแกนหุ้น 1 ตัว — คืนผลเสมอ (ไม่ filter ที่นี่)"""
    try:
        ticker = yf.Ticker(f"{symbol}.BK")
        df = ticker.history(period=period, auto_adjust=True)
        if df is None or df.empty or len(df) < 20:
            return None

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[['Open','High','Low','Close','Volume']].copy()

        df = add_all_indicators(df)
        if df.empty or len(df) < 15:
            return None

        current = float(df['Close'].iloc[-1])
        if current <= 0:
            return None

        # ── Fibonacci ────────────────────────────────────────────────
        swing_high = float(df['High'].max())
        swing_low  = float(df['Low'].min())
        fib_range  = swing_high - swing_low
        if fib_range <= 0:
            return None

        # Determine trend by comparing EMA slopes
        if 'EMA50' in df.columns and len(df) >= 10:
            ema_now  = float(df['EMA50'].iloc[-1])
            ema_prev = float(df['EMA50'].iloc[-10])
            is_uptrend = ema_now >= ema_prev
        else:
            mid = len(df) // 2
            is_uptrend = df['Close'].iloc[mid:].mean() >= df['Close'].iloc[:mid].mean()

        # In uptrend: fib measures pullback from high (base=low, direction=up)
        # In downtrend: fib measures bounce from low (base=high, direction=down)
        base      = swing_low  if is_uptrend else swing_high
        direction = 1          if is_uptrend else -1

        def fp(r): return base + direction * fib_range * r

        fib_prices = {r: fp(r) for r in [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]}

        # ── Which zone is current price in? ──────────────────────────
        zone_ratio = None
        zone_name  = "นอกช่วง"
        for r0, r1 in [(0.0,0.236),(0.236,0.382),(0.382,0.5),(0.5,0.618),(0.618,0.786),(0.786,1.0)]:
            p0, p1 = fp(r0), fp(r1)
            lo, hi = min(p0, p1), max(p0, p1)
            if lo <= current <= hi:
                zone_ratio = (r0 + r1) / 2
                if 0.382 <= zone_ratio <= 0.618:
                    zone_name = f"{int(r0*100)}.{int((r0*1000)%10)}–{int(r1*100)}.{int((r1*1000)%10)}% 🌟"
                else:
                    zone_name = f"{r0*100:.1f}–{r1*100:.1f}%"
                break

        # ── Distance to nearest key Fib level ────────────────────────
        key_levels = {
            "38.2%": fp(0.382),
            "50.0%": fp(0.500),
            "61.8%": fp(0.618),
        }
        nearest_level = min(key_levels, key=lambda k: abs(key_levels[k] - current))
        nearest_price = key_levels[nearest_level]
        dist_nearest  = abs((current - nearest_price) / nearest_price * 100)

        dist_to_golden = abs((current - fp(0.618)) / fp(0.618) * 100)

        # ── Technical indicators ──────────────────────────────────────
        score, signals, regime = calculate_signal_score(df)

        rsi = float(df['RSI'].iloc[-1])        if 'RSI' in df.columns else 50.0
        atr = float(df['ATR'].iloc[-1])        if 'ATR' in df.columns else current * 0.02
        vol_ratio = float(df['Vol_ratio'].iloc[-1]) if 'Vol_ratio' in df.columns else 1.0

        for v in [rsi, atr, vol_ratio]:
            if pd.isna(v): v = 50.0 if v == rsi else current*0.02 if v == atr else 1.0

        # ── Stop Loss — always on the correct side ────────────────────
        if is_uptrend:
            # SL below current: the lower of (fib 78.6% or current - 1.5*ATR)
            sl_fib = fp(0.786)
            sl_atr = current - atr * 1.5
            stop_loss = round(max(min(sl_fib, sl_atr), current * 0.85), 2)  # floor at -15%
            stop_loss = min(stop_loss, current * 0.98)  # must be below current
        else:
            # SL above current
            sl_fib = fp(0.786)
            sl_atr = current + atr * 1.5
            stop_loss = round(min(max(sl_fib, sl_atr), current * 1.15), 2)
            stop_loss = max(stop_loss, current * 1.02)

        risk_pct = abs((current - stop_loss) / current * 100)
        risk_pct = max(risk_pct, 0.1)  # avoid div/0

        # ── Take Profit levels ────────────────────────────────────────
        tp_candidates = []
        for r in [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]:
            tp = fp(r)
            if is_uptrend and tp > current * 1.005:
                tp_candidates.append(round(tp, 2))
            elif not is_uptrend and tp < current * 0.995:
                tp_candidates.append(round(tp, 2))

        tp1 = tp_candidates[0] if tp_candidates          else round(current * (1.05 if is_uptrend else 0.95), 2)
        tp2 = tp_candidates[1] if len(tp_candidates) > 1 else round(current * (1.10 if is_uptrend else 0.90), 2)

        profit_pct = abs(tp1 - current) / current * 100
        rr = profit_pct / risk_pct

        # ── Fib Score (0–100) ─────────────────────────────────────────
        fib_score = 0

        # A) Zone quality (max 35 pts)
        if zone_ratio is not None:
            if 0.382 <= zone_ratio <= 0.618:
                fib_score += 35   # Golden zone
            elif 0.236 <= zone_ratio <= 0.786:
                fib_score += 18   # Acceptable zone

        # B) Proximity to nearest key level (max 30 pts)
        # 0% away = 30 pts, 10% away = 0 pts
        prox = max(0.0, 30.0 - dist_nearest * 3.0)
        fib_score += prox

        # C) Technical signal score (max 20 pts)
        fib_score += (score / 100.0) * 20.0

        # D) Volume confirmation (max 10 pts)
        if vol_ratio >= 2.0:   fib_score += 10
        elif vol_ratio >= 1.5: fib_score += 7
        elif vol_ratio >= 1.0: fib_score += 4

        # E) RSI health bonus (max 5 pts)
        if 30 <= rsi <= 60:   fib_score += 5   # Healthy / room to run
        elif rsi < 30:        fib_score += 4   # Oversold — potential bounce
        elif rsi <= 70:       fib_score += 2

        fib_score = min(100.0, round(fib_score, 1))

        # ── Grade ─────────────────────────────────────────────────────
        if   fib_score >= 75: grade = "A+"
        elif fib_score >= 65: grade = "A"
        elif fib_score >= 55: grade = "B+"
        elif fib_score >= 45: grade = "B"
        else:                 grade = "C"

        buy_count  = sum(1 for s in signals if s['type'] == 'BUY')
        sell_count = sum(1 for s in signals if s['type'] == 'SELL')

        price_5d  = float(df['Close'].iloc[-6]) if len(df) >= 6 else current
        change_5d = (current - price_5d) / price_5d * 100 if price_5d > 0 else 0.0

        return {
            "symbol":        symbol,
            "price":         round(current, 2),
            "fib_score":     fib_score,
            "grade":         grade,
            "zone":          zone_name,
            "dist_nearest":  round(dist_nearest, 1),
            "nearest_level": nearest_level,
            "dist_golden":   round(dist_to_golden, 1),
            "signal_score":  score,
            "regime":        regime,
            "rsi":           round(rsi, 1),
            "vol_ratio":     round(vol_ratio, 2),
            "buy_signals":   buy_count,
            "sell_signals":  sell_count,
            "stop_loss":     stop_loss,
            "tp1":           tp1,
            "tp2":           tp2,
            "risk_pct":      round(risk_pct, 1),
            "risk_reward":   round(rr, 2),
            "is_uptrend":    is_uptrend,
            "change_5d":     round(change_5d, 2),
            "swing_high":    round(swing_high, 2),
            "swing_low":     round(swing_low, 2),
            "golden_price":  round(fp(0.618), 2),
        }

    except Exception:
        return None


def run_fibonacci_scan(
    symbols: Optional[List[str]] = None,
    period: str = "1y",
    min_fib_score: int = 40,
    min_rr: float = 1.0,
    max_workers: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """
    สแกนหุ้นหลายตัวพร้อมกัน คืน DataFrame เรียงตาม fib_score
    """
    if symbols is None:
        symbols = SET_UNIVERSE

    results = []
    total = len(symbols)
    done  = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_scan_one, sym, period): sym for sym in symbols}
        for future in as_completed(future_map):
            done += 1
            sym = future_map[future]
            if progress_callback:
                progress_callback(done, total, sym)
            try:
                result = future.result(timeout=30)
                if result is not None:
                    results.append(result)
            except Exception:
                pass

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Apply user filters AFTER collecting all results
    df = df[df['fib_score']   >= min_fib_score]
    df = df[df['risk_reward'] >= min_rr]
    df = df.sort_values('fib_score', ascending=False).reset_index(drop=True)
    return df


def run_multi_timeframe_scan(
    symbols: Optional[List[str]] = None,
    periods: Optional[List[str]] = None,
    min_fib_score: int = 40,
    min_rr: float = 1.0,
    max_workers: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Multi-timeframe Fibonacci scan
    สแกนแต่ละหุ้นใน 3 timeframe พร้อมกัน
    หุ้นที่ผ่านหลาย timeframe = confluence = signal แข็งแกร่งมาก
    """
    if symbols is None:
        symbols = SET_UNIVERSE
    if periods is None:
        periods = ["3mo", "6mo", "1y"]

    # Collect results per (symbol, period)
    tasks = [(sym, p) for sym in symbols for p in periods]
    total = len(tasks)
    done  = 0
    raw   = {}  # symbol -> {period -> result}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_scan_one, sym, p): (sym, p) for sym, p in tasks}
        for future in as_completed(future_map):
            done += 1
            sym, p = future_map[future]
            if progress_callback:
                progress_callback(done, total, f"{sym} ({p})")
            try:
                result = future.result(timeout=30)
                if result is not None:
                    if sym not in raw:
                        raw[sym] = {}
                    raw[sym][p] = result
            except Exception:
                pass

    if not raw:
        return pd.DataFrame()

    # Merge: for each symbol, combine results across timeframes
    merged = []
    period_labels = {"3mo": "3M", "6mo": "6M", "1y": "1Y"}

    for sym, period_results in raw.items():
        if not period_results:
            continue

        n_periods = len(period_results)
        # Use the middle timeframe (6mo) as primary, fallback to whatever exists
        primary_key = "6mo" if "6mo" in period_results else list(period_results.keys())[0]
        primary = period_results[primary_key].copy()

        # Score each timeframe
        tf_scores  = {p: r['fib_score']   for p, r in period_results.items()}
        tf_signals = {p: r['signal_score'] for p, r in period_results.items()}
        tf_rr      = {p: r['risk_reward']  for p, r in period_results.items()}
        tf_zones   = {p: r['zone']         for p, r in period_results.items()}

        # Confluence: average fib scores, boosted by agreement
        avg_fib    = sum(tf_scores.values()) / n_periods
        min_fib_tf = min(tf_scores.values())

        # Confluence bonus: all timeframes in golden zone
        all_in_golden = all(
            '38' in z or '50' in z or '61' in z or '🌟' in z
            for z in tf_zones.values()
        )
        confluence_bonus = 15 if (n_periods >= 3 and all_in_golden) else \
                           10 if (n_periods >= 2 and all_in_golden) else \
                            5 if n_periods >= 2 else 0

        mtf_score = min(100, round(avg_fib + confluence_bonus, 1))

        # Confluence label
        passed = sum(1 for s in tf_scores.values() if s >= min_fib_score)
        if passed == 3:   confluence = "🟢🟢🟢 ทั้ง 3 Timeframe"
        elif passed == 2: confluence = "🟢🟢⚪ 2 จาก 3 Timeframe"
        elif passed == 1: confluence = "🟢⚪⚪ 1 Timeframe"
        else:             confluence = "⚪⚪⚪ ไม่ผ่าน"

        # Build row
        row = primary.copy()
        row['mtf_score']      = mtf_score
        row['confluence']     = confluence
        row['passed_tfs']     = passed
        row['n_periods']      = n_periods
        row['score_3mo']      = tf_scores.get('3mo', None)
        row['score_6mo']      = tf_scores.get('6mo', None)
        row['score_1y']       = tf_scores.get('1y',  None)
        row['zone_3mo']       = tf_zones.get('3mo',  '—')
        row['zone_6mo']       = tf_zones.get('6mo',  '—')
        row['zone_1y']        = tf_zones.get('1y',   '—')

        # Re-grade based on mtf_score
        if   mtf_score >= 80: row['grade'] = 'A+'
        elif mtf_score >= 70: row['grade'] = 'A'
        elif mtf_score >= 58: row['grade'] = 'B+'
        elif mtf_score >= 45: row['grade'] = 'B'
        else:                 row['grade'] = 'C'

        merged.append(row)

    if not merged:
        return pd.DataFrame()

    df = pd.DataFrame(merged)
    df = df[df['mtf_score']   >= min_fib_score]
    df = df[df['risk_reward'] >= min_rr]
    df = df.sort_values(['passed_tfs', 'mtf_score'], ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# DAY TRADE SCANNER
# ═══════════════════════════════════════════════════════════════════════

# yfinance intraday limits:
# interval 1m  → period max 7d
# interval 5m  → period max 60d
# interval 15m → period max 60d
# interval 30m → period max 60d
# interval 1h  → period max 730d

DAYTRADE_INTERVALS = {
    "5m":  {"period": "5d",  "label": "5 นาที (Scalp)",    "min_bars": 30},
    "15m": {"period": "5d",  "label": "15 นาที (Day trade)","min_bars": 20},
    "30m": {"period": "10d", "label": "30 นาที (Intraday)", "min_bars": 15},
    "1h":  {"period": "20d", "label": "1 ชั่วโมง (Short swing)", "min_bars": 20},
}


def _add_intraday_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicators เฉพาะ intraday — lightweight เพราะข้อมูลน้อย
    ไม่ใช้ EMA200 หรือ ADX (ต้องการแท่งเยอะ)
    """
    from modules.indicators import _ema, _rsi, _macd, _bbands, _atr, _obv

    df = df.copy()
    df['EMA9']  = _ema(df['Close'], 9)
    df['EMA21'] = _ema(df['Close'], 21)
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['RSI']   = _rsi(df['Close'], 14)

    df['MACD'], df['MACD_signal'], df['MACD_hist'] = _macd(df['Close'], 8, 17, 9)
    df['BB_upper'], df['BB_middle'], df['BB_lower'], df['BB_width'] = _bbands(df['Close'], 10, 2)
    df['ATR']   = _atr(df['High'], df['Low'], df['Close'], 7)
    df['OBV']   = _obv(df['Close'], df['Volume'])

    df['Vol_SMA20'] = df['Volume'].rolling(10).mean()
    df['Vol_ratio'] = df['Volume'] / df['Vol_SMA20'].replace(0, np.nan)

    df = df.dropna(subset=['EMA9', 'RSI', 'ATR'])
    return df


def _intraday_signal_score(df: pd.DataFrame) -> tuple:
    """
    Signal score เฉพาะ intraday — weighted แตกต่างจาก daily
    เน้น momentum + volume มากกว่า trend
    returns: (score 0-100, signals list, regime str)
    """
    if df.empty or len(df) < 5:
        return 50, [], "SIDEWAYS"

    score   = 50
    signals = []
    last    = df.iloc[-1]
    prev    = df.iloc[-2] if len(df) >= 2 else last

    close = float(last['Close'])
    ema9  = float(last['EMA9'])
    ema21 = float(last['EMA21'])
    rsi   = float(last['RSI'])   if not pd.isna(last['RSI'])   else 50
    macd  = float(last['MACD'])  if not pd.isna(last['MACD'])  else 0
    msig  = float(last['MACD_signal']) if not pd.isna(last['MACD_signal']) else 0
    mhist = float(last['MACD_hist'])   if not pd.isna(last['MACD_hist'])   else 0
    vol_r = float(last['Vol_ratio'])   if not pd.isna(last['Vol_ratio'])   else 1.0

    bb_up = float(last['BB_upper']) if not pd.isna(last['BB_upper']) else close*1.02
    bb_lo = float(last['BB_lower']) if not pd.isna(last['BB_lower']) else close*0.98

    # ── EMA alignment (momentum) ──────────────────────────────────────
    if ema9 > ema21 and close > ema9:
        score += 12
        signals.append({"type":"BUY",  "reason":"EMA9 > EMA21 ราคาเหนือเส้น", "strength":"MEDIUM"})
    elif ema9 < ema21 and close < ema9:
        score -= 12
        signals.append({"type":"SELL", "reason":"EMA9 < EMA21 ราคาต่ำกว่าเส้น", "strength":"MEDIUM"})

    # ── MACD histogram direction ──────────────────────────────────────
    prev_hist = float(prev['MACD_hist']) if not pd.isna(prev['MACD_hist']) else 0
    if mhist > 0 and mhist > prev_hist:
        score += 10
        signals.append({"type":"BUY",  "reason":"MACD Histogram เพิ่มขึ้น", "strength":"MEDIUM"})
    elif mhist < 0 and mhist < prev_hist:
        score -= 10
        signals.append({"type":"SELL", "reason":"MACD Histogram ลดลง", "strength":"MEDIUM"})
    if macd > msig and float(prev.get('MACD', macd)) <= float(prev.get('MACD_signal', msig)):
        score += 8
        signals.append({"type":"BUY",  "reason":"MACD Cross Signal ขาขึ้น", "strength":"STRONG"})

    # ── RSI zones ─────────────────────────────────────────────────────
    if rsi < 30:
        score += 15
        signals.append({"type":"BUY",  "reason":f"RSI Oversold ({rsi:.0f})", "strength":"STRONG"})
    elif rsi > 70:
        score -= 15
        signals.append({"type":"SELL", "reason":f"RSI Overbought ({rsi:.0f})", "strength":"STRONG"})
    elif 40 <= rsi <= 60:
        score += 5

    # ── Bollinger Band bounce ─────────────────────────────────────────
    if close <= bb_lo * 1.005:
        score += 10
        signals.append({"type":"BUY",  "reason":"ราคาชน BB Lower — โอกาส Bounce", "strength":"MEDIUM"})
    elif close >= bb_up * 0.995:
        score -= 10
        signals.append({"type":"SELL", "reason":"ราคาชน BB Upper — โอกาส Reversal", "strength":"MEDIUM"})

    # ── Volume spike ──────────────────────────────────────────────────
    if vol_r >= 2.0:
        score += 10
        signals.append({"type":"BUY" if close >= float(last['Open']) else "SELL",
                        "reason":f"Volume Spike {vol_r:.1f}x", "strength":"STRONG"})
    elif vol_r >= 1.5:
        score += 5

    # ── Candle momentum ───────────────────────────────────────────────
    candle_body = close - float(last['Open'])
    candle_size = float(last['High']) - float(last['Low'])
    if candle_size > 0 and abs(candle_body) / candle_size > 0.6:
        if candle_body > 0:
            score += 5
            signals.append({"type":"BUY",  "reason":"แท่งเขียวแข็งแกร่ง", "strength":"WEAK"})
        else:
            score -= 5
            signals.append({"type":"SELL", "reason":"แท่งแดงแข็งแกร่ง", "strength":"WEAK"})

    score = max(0, min(100, score))

    if score >= 65:   regime = "BULL_MOMENTUM"
    elif score <= 35: regime = "BEAR_MOMENTUM"
    else:             regime = "SIDEWAYS"

    return score, signals, regime


def _scan_intraday(symbol: str, interval: str = "15m") -> Optional[dict]:
    """สแกน intraday — ใช้ Fib จาก Swing High/Low ของช่วง intraday"""
    cfg = DAYTRADE_INTERVALS.get(interval, DAYTRADE_INTERVALS["15m"])
    try:
        ticker = yf.Ticker(f"{symbol}.BK")
        df = ticker.history(period=cfg["period"], interval=interval, auto_adjust=True)

        if df is None or df.empty or len(df) < cfg["min_bars"]:
            return None

        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df[['Open','High','Low','Close','Volume']].copy()
        df = df.dropna()

        df = _add_intraday_indicators(df)
        if df.empty or len(df) < 10:
            return None

        current = float(df['Close'].iloc[-1])
        if current <= 0:
            return None

        # ── Intraday Fib: ใช้ swing จาก candles ล่าสุด ───────────────
        # ใช้ 2 วันล่าสุด หรือ 40 candles ล่าสุด
        lookback = min(len(df), 40)
        df_fib   = df.iloc[-lookback:]

        swing_high = float(df_fib['High'].max())
        swing_low  = float(df_fib['Low'].min())
        fib_range  = swing_high - swing_low
        if fib_range < current * 0.001:  # range น้อยกว่า 0.1% ของราคา
            return None

        # Trend: EMA slope ล่าสุด
        if 'EMA21' in df.columns and len(df) >= 5:
            is_uptrend = float(df['EMA21'].iloc[-1]) >= float(df['EMA21'].iloc[-5])
        else:
            is_uptrend = df['Close'].iloc[-10:].mean() >= df['Close'].iloc[-20:-10].mean()

        base      = swing_low  if is_uptrend else swing_high
        direction = 1          if is_uptrend else -1

        def fp(r): return base + direction * fib_range * r

        # Zone detection
        zone_ratio = None
        zone_name  = "นอกช่วง"
        for r0, r1 in [(0.0,0.236),(0.236,0.382),(0.382,0.5),(0.5,0.618),(0.618,0.786),(0.786,1.0)]:
            p0, p1 = fp(r0), fp(r1)
            lo, hi = min(p0, p1), max(p0, p1)
            if lo <= current <= hi:
                zone_ratio = (r0 + r1) / 2
                star = "🌟" if 0.382 <= zone_ratio <= 0.618 else ""
                zone_name = f"{r0*100:.1f}–{r1*100:.1f}% {star}".strip()
                break

        # Nearest key level
        key_prices = {"38.2%": fp(0.382), "50.0%": fp(0.500), "61.8%": fp(0.618)}
        nearest_level = min(key_prices, key=lambda k: abs(key_prices[k]-current))
        nearest_price = key_prices[nearest_level]
        dist_nearest  = abs((current - nearest_price) / nearest_price * 100)
        dist_golden   = abs((current - fp(0.618)) / fp(0.618) * 100)

        # Signal score (intraday version)
        score, signals, regime = _intraday_signal_score(df)

        rsi       = float(df['RSI'].iloc[-1])        if 'RSI' in df.columns else 50.0
        atr       = float(df['ATR'].iloc[-1])        if 'ATR' in df.columns else current * 0.01
        vol_ratio = float(df['Vol_ratio'].iloc[-1])  if 'Vol_ratio' in df.columns else 1.0
        if pd.isna(rsi): rsi = 50.0
        if pd.isna(atr): atr = current * 0.01
        if pd.isna(vol_ratio): vol_ratio = 1.0

        # Stop Loss — tighter for intraday (1.0 × ATR)
        if is_uptrend:
            stop_loss = round(max(current - atr * 1.0, current * 0.97), 2)
            stop_loss = min(stop_loss, current * 0.99)
        else:
            stop_loss = round(min(current + atr * 1.0, current * 1.03), 2)
            stop_loss = max(stop_loss, current * 1.01)

        risk_pct = max(abs((current - stop_loss) / current * 100), 0.1)

        # TP levels (tighter for intraday)
        tp_candidates = []
        for r in [0.236, 0.382, 0.5, 0.618, 0.786, 1.0]:
            tp = fp(r)
            if is_uptrend and tp > current * 1.002:
                tp_candidates.append(round(tp, 2))
            elif not is_uptrend and tp < current * 0.998:
                tp_candidates.append(round(tp, 2))

        tp1 = tp_candidates[0] if tp_candidates          else round(current * (1.02 if is_uptrend else 0.98), 2)
        tp2 = tp_candidates[1] if len(tp_candidates) > 1 else round(current * (1.04 if is_uptrend else 0.96), 2)
        rr  = abs(tp1 - current) / current * 100 / risk_pct

        # Fib Score
        fib_score = 0
        if zone_ratio is not None:
            if 0.382 <= zone_ratio <= 0.618:  fib_score += 35
            elif 0.236 <= zone_ratio <= 0.786: fib_score += 18
        fib_score += max(0.0, 30.0 - dist_nearest * 3.0)
        fib_score += (score / 100.0) * 20.0
        if vol_ratio >= 2.0:   fib_score += 10
        elif vol_ratio >= 1.5: fib_score += 7
        elif vol_ratio >= 1.0: fib_score += 4
        if 30 <= rsi <= 60:    fib_score += 5
        elif rsi < 30:         fib_score += 4

        fib_score = min(100.0, round(fib_score, 1))

        if   fib_score >= 75: grade = "A+"
        elif fib_score >= 65: grade = "A"
        elif fib_score >= 55: grade = "B+"
        elif fib_score >= 45: grade = "B"
        else:                 grade = "C"

        # Last candle info
        last_bar  = df.iloc[-1]
        open_price = float(last_bar['Open'])
        bar_change = (current - open_price) / open_price * 100

        # VWAP approximation
        vwap = float((df_fib['Close'] * df_fib['Volume']).sum() /
                     df_fib['Volume'].sum()) if df_fib['Volume'].sum() > 0 else current
        vs_vwap = (current - vwap) / vwap * 100

        return {
            "symbol":        symbol,
            "price":         round(current, 2),
            "fib_score":     fib_score,
            "grade":         grade,
            "zone":          zone_name,
            "dist_nearest":  round(dist_nearest, 1),
            "nearest_level": nearest_level,
            "dist_golden":   round(dist_golden, 1),
            "signal_score":  score,
            "regime":        regime,
            "rsi":           round(rsi, 1),
            "vol_ratio":     round(vol_ratio, 2),
            "buy_signals":   sum(1 for s in signals if s['type']=='BUY'),
            "sell_signals":  sum(1 for s in signals if s['type']=='SELL'),
            "stop_loss":     stop_loss,
            "tp1":           tp1,
            "tp2":           tp2,
            "risk_pct":      round(risk_pct, 2),
            "risk_reward":   round(rr, 2),
            "is_uptrend":    is_uptrend,
            "change_5d":     round(bar_change, 2),   # reuse field = bar change
            "swing_high":    round(swing_high, 2),
            "swing_low":     round(swing_low, 2),
            "golden_price":  round(fp(0.618), 2),
            "vwap":          round(vwap, 2),
            "vs_vwap_pct":   round(vs_vwap, 2),
            "interval":      interval,
            "atr":           round(atr, 3),
        }

    except Exception:
        return None


def run_daytrade_scan(
    symbols: Optional[List[str]] = None,
    interval: str = "15m",
    min_fib_score: int = 40,
    min_rr: float = 1.0,
    max_workers: int = 10,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Intraday Fibonacci Scanner
    ใช้ interval 5m / 15m / 30m / 1h
    """
    if symbols is None:
        # Day trade: ใช้ liquid stocks เท่านั้น (volume สูง)
        symbols = [
            "PTT","ADVANC","KBANK","SCB","KTB","BBL","AOT","CPALL","SCC","GULF",
            "PTTEP","BDMS","DELTA","TRUE","MTC","SAWAD","MINT","HMPRO","COM7",
            "TU","CPF","IVL","BANPU","EA","SPALI","LH","CPN","BJC","WHA","AMATA",
            "BH","CHG","CENTEL","ERW","BCH","TIDLOR","AEONTS","BAY","TISCO","KKP",
        ]

    results = []
    total   = len(symbols)
    done    = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_scan_intraday, sym, interval): sym for sym in symbols}
        for future in as_completed(future_map):
            done += 1
            sym = future_map[future]
            if progress_callback:
                progress_callback(done, total, sym)
            try:
                result = future.result(timeout=30)
                if result is not None:
                    results.append(result)
            except Exception:
                pass

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df[df['fib_score']   >= min_fib_score]
    df = df[df['risk_reward'] >= min_rr]
    df = df.sort_values('fib_score', ascending=False).reset_index(drop=True)
    return df
