"""
Candlestick Pattern Analysis + Bell Curve / Mean Reversion
──────────────────────────────────────────────────────────
Pattern detection จาก context หลายแท่ง พร้อม:
- Confidence score (0–100)
- Context: อยู่ที่แนวรับ/ต้านไหม?
- Bell Curve: Z-score, percentile, mean reversion probability
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: CANDLESTICK PATTERN DETECTOR
# ══════════════════════════════════════════════════════════════════════

def _body(c) -> float:
    return abs(float(c['Close']) - float(c['Open']))

def _upper_wick(c) -> float:
    return float(c['High']) - max(float(c['Close']), float(c['Open']))

def _lower_wick(c) -> float:
    return min(float(c['Close']), float(c['Open'])) - float(c['Low'])

def _range(c) -> float:
    return max(float(c['High']) - float(c['Low']), 1e-9)

def _is_bull(c) -> bool:
    return float(c['Close']) > float(c['Open'])

def _is_bear(c) -> bool:
    return float(c['Close']) < float(c['Open'])

def _body_pct(c) -> float:
    """สัดส่วนเนื้อเทียนต่อช่วงทั้งหมด"""
    return _body(c) / _range(c)


def detect_patterns_full(df: pd.DataFrame, lookback: int = 5) -> list:
    """
    ตรวจหา candlestick patterns จาก context หลายแท่ง
    lookback: จำนวนแท่งสูงสุดที่ใช้วิเคราะห์
    Returns list of pattern dicts, เรียงจากล่าสุด → เก่าสุด
    """
    if len(df) < 3:
        return []

    results = []

    # Precompute average body size (สำหรับ relative comparison)
    recent = df.iloc[-20:] if len(df) >= 20 else df
    avg_body  = recent.apply(_body, axis=1).mean()
    avg_range = recent.apply(_range, axis=1).mean()
    avg_vol   = float(df['Volume'].iloc[-20:].mean()) if 'Volume' in df.columns else 1

    def get_candle(i):
        """i=0 = ล่าสุด, i=1 = ก่อนหน้า 1, ..."""
        idx = len(df) - 1 - i
        return df.iloc[idx] if idx >= 0 else None

    c0 = get_candle(0)  # latest
    c1 = get_candle(1)
    c2 = get_candle(2)
    c3 = get_candle(3)
    c4 = get_candle(4)

    date0 = df.index[-1]
    vol0  = float(c0['Volume']) if 'Volume' in c0 else avg_vol

    # ── 1. Long Green Candle (แท่งเขียวยาว) ────────────────────────────
    if c0 is not None and _is_bull(c0) and _body(c0) > avg_body * 1.5 and _body_pct(c0) > 0.6:
        conf = min(100, 60 + int((_body_pct(c0) - 0.6) * 100))
        if vol0 > avg_vol * 1.2: conf = min(100, conf + 15)
        results.append({
            "pattern":     "Long Green Candle",
            "type":        "BUY",
            "strength":    "STRONG" if conf >= 75 else "MEDIUM",
            "confidence":  conf,
            "date":        date0,
            "bar_index":   len(df) - 1,
            "price":       float(c0['Close']),
            "description": "แท่งเขียวยาว เนื้อหนา — แรงซื้อคุมเกม",
            "tip":         "ยืนยันด้วย Volume สูง และไม่อยู่ใกล้แนวต้านสำคัญ",
            "emoji":       "🟢",
        })

    # ── 2. Long Red Candle (แท่งแดงยาว) ────────────────────────────────
    if c0 is not None and _is_bear(c0) and _body(c0) > avg_body * 1.5 and _body_pct(c0) > 0.6:
        conf = min(100, 60 + int((_body_pct(c0) - 0.6) * 100))
        if vol0 > avg_vol * 1.2: conf = min(100, conf + 15)
        results.append({
            "pattern":     "Long Red Candle",
            "type":        "SELL",
            "strength":    "STRONG" if conf >= 75 else "MEDIUM",
            "confidence":  conf,
            "date":        date0,
            "bar_index":   len(df) - 1,
            "price":       float(c0['Close']),
            "description": "แท่งแดงยาว เนื้อหนา — แรงขายรุนแรง",
            "tip":         "ระวังถ้าปริมาณซื้อขายสูง หมายถึง distribution",
            "emoji":       "🔴",
        })

    # ── 3. Hammer ──────────────────────────────────────────────────────
    if c0 is not None and c1 is not None:
        lw, uw, b = _lower_wick(c0), _upper_wick(c0), _body(c0)
        if lw > b * 2.0 and uw < b * 0.5 and _body_pct(c0) < 0.35:
            # Context: prior downtrend?
            prior_down = c1 is not None and float(c1['Close']) < float(c1['Open'])
            conf = 55 + (20 if prior_down else 0) + (10 if _is_bull(c0) else 0)
            results.append({
                "pattern":     "Hammer",
                "type":        "BUY",
                "strength":    "STRONG" if prior_down else "MEDIUM",
                "confidence":  min(100, conf),
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "ค้อน — ไส้ล่างยาว ราคากดลงแล้วสะท้อนกลับ",
                "tip":         "แรงที่สุดเมื่อเกิดที่แนวรับ + Volume สูง รอแท่งถัดไปยืนยัน",
                "emoji":       "🔨",
            })

    # ── 4. Shooting Star ───────────────────────────────────────────────
    if c0 is not None and c1 is not None:
        lw, uw, b = _lower_wick(c0), _upper_wick(c0), _body(c0)
        if uw > b * 2.0 and lw < b * 0.5 and _body_pct(c0) < 0.35:
            prior_up = c1 is not None and float(c1['Close']) > float(c1['Open'])
            conf = 55 + (20 if prior_up else 0) + (10 if _is_bear(c0) else 0)
            results.append({
                "pattern":     "Shooting Star",
                "type":        "SELL",
                "strength":    "STRONG" if prior_up else "MEDIUM",
                "confidence":  min(100, conf),
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['High']),
                "description": "ดาวตก — ไส้บนยาว ราคาพุ่งขึ้นแล้วถูกกด",
                "tip":         "อันตรายที่แนวต้านสำคัญ รอแท่งแดงยืนยัน",
                "emoji":       "🌠",
            })

    # ── 5. Inverted Hammer ─────────────────────────────────────────────
    if c0 is not None and c1 is not None:
        lw, uw, b = _lower_wick(c0), _upper_wick(c0), _body(c0)
        if uw > b * 2.0 and lw < b * 0.5 and _is_bear(c1):
            # Like Shooting Star but after downtrend = potential reversal up
            results.append({
                "pattern":     "Inverted Hammer",
                "type":        "BUY",
                "strength":    "WEAK",
                "confidence":  50,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "ค้อนกลับหัว — ต้องการการยืนยัน",
                "tip":         "รอแท่งเขียวถัดไปปิดเหนือ high ของแท่งนี้ก่อนซื้อ",
                "emoji":       "🔃",
            })

    # ── 6. Bullish Engulfing ───────────────────────────────────────────
    if c0 is not None and c1 is not None:
        if (_is_bear(c1) and _is_bull(c0) and
                float(c0['Open']) <= float(c1['Close']) and
                float(c0['Close']) >= float(c1['Open'])):
            size_ratio = _body(c0) / max(_body(c1), 1e-9)
            conf = min(100, 65 + int((size_ratio - 1) * 20))
            if vol0 > avg_vol * 1.3: conf = min(100, conf + 10)
            results.append({
                "pattern":     "Bullish Engulfing",
                "type":        "BUY",
                "strength":    "STRONG",
                "confidence":  conf,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "กลืนกินขาขึ้น — แท่งเขียวครอบแท่งแดงทั้งหมด",
                "tip":         "ยิ่งแท่งเขียวใหญ่กว่าแท่งแดงมากเท่าไหร่ ยิ่งแรง",
                "emoji":       "🌑➡🌕",
            })

    # ── 7. Bearish Engulfing ───────────────────────────────────────────
    if c0 is not None and c1 is not None:
        if (_is_bull(c1) and _is_bear(c0) and
                float(c0['Open']) >= float(c1['Close']) and
                float(c0['Close']) <= float(c1['Open'])):
            size_ratio = _body(c0) / max(_body(c1), 1e-9)
            conf = min(100, 65 + int((size_ratio - 1) * 20))
            if vol0 > avg_vol * 1.3: conf = min(100, conf + 10)
            results.append({
                "pattern":     "Bearish Engulfing",
                "type":        "SELL",
                "strength":    "STRONG",
                "confidence":  conf,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "กลืนกินขาลง — แท่งแดงครอบแท่งเขียวทั้งหมด",
                "tip":         "อันตรายมากในขาขึ้น บ่งชี้การเปลี่ยนแปลงของอารมณ์ตลาด",
                "emoji":       "🌕➡🌑",
            })

    # ── 8. Morning Star (3 แท่ง) ──────────────────────────────────────
    if c0 is not None and c1 is not None and c2 is not None:
        if (_is_bear(c2) and _body(c1) < _body(c2) * 0.35 and
                _is_bull(c0) and float(c0['Close']) > (float(c2['Open']) + float(c2['Close'])) / 2):
            conf = 75 + (10 if vol0 > avg_vol else 0)
            results.append({
                "pattern":     "Morning Star",
                "type":        "BUY",
                "strength":    "STRONG",
                "confidence":  min(100, conf),
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "ดาวรุ่ง (3 แท่ง) — กลับตัวขาขึ้นที่แนวรับ",
                "tip":         "pattern 3 แท่งที่เชื่อถือได้มาก เฉพาะเมื่ออยู่ที่แนวรับ",
                "emoji":       "🌅",
            })

    # ── 9. Evening Star (3 แท่ง) ──────────────────────────────────────
    if c0 is not None and c1 is not None and c2 is not None:
        if (_is_bull(c2) and _body(c1) < _body(c2) * 0.35 and
                _is_bear(c0) and float(c0['Close']) < (float(c2['Open']) + float(c2['Close'])) / 2):
            conf = 75 + (10 if vol0 > avg_vol else 0)
            results.append({
                "pattern":     "Evening Star",
                "type":        "SELL",
                "strength":    "STRONG",
                "confidence":  min(100, conf),
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "ดาวตอนเย็น (3 แท่ง) — กลับตัวขาลงที่แนวต้าน",
                "tip":         "เชื่อถือได้สูงเมื่ออยู่ที่แนวต้าน ควรขายทำกำไร",
                "emoji":       "🌇",
            })

    # ── 10. Doji (ลังเล) ───────────────────────────────────────────────
    if c0 is not None:
        if _body_pct(c0) < 0.08:
            # Context matters: doji after trend = stronger signal
            after_up   = c1 is not None and _is_bull(c1) and _body(c1) > avg_body
            after_down = c1 is not None and _is_bear(c1) and _body(c1) > avg_body
            sig_type = "SELL" if after_up else "BUY" if after_down else "NEUTRAL"
            conf = 60 if sig_type != "NEUTRAL" else 40
            results.append({
                "pattern":     "Doji",
                "type":        sig_type,
                "strength":    "MEDIUM" if sig_type != "NEUTRAL" else "WEAK",
                "confidence":  conf,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "โดจิ — ตลาดลังเล ดุลอำนาจซื้อ-ขายเท่ากัน",
                "tip":         "ดูบริบท: หลังขาขึ้นยาว = เตือนขาย / หลังขาลงยาว = โอกาสซื้อ",
                "emoji":       "⚖️",
            })

    # ── 11. Three White Soldiers (3 แท่งเขียวต่อเนื่อง) ───────────────
    if c0 is not None and c1 is not None and c2 is not None:
        if (_is_bull(c0) and _is_bull(c1) and _is_bull(c2) and
                _body(c0) > avg_body * 0.8 and _body(c1) > avg_body * 0.8 and
                float(c0['Close']) > float(c1['Close']) > float(c2['Close'])):
            results.append({
                "pattern":     "Three White Soldiers",
                "type":        "BUY",
                "strength":    "STRONG",
                "confidence":  82,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "ทหารเขียว 3 แถว — ขาขึ้นต่อเนื่อง แรงซื้อสม่ำเสมอ",
                "tip":         "Trend ขาขึ้นแข็งแกร่ง แต่ระวัง overbought หลังพุ่งยาว",
                "emoji":       "💪💪💪",
            })

    # ── 12. Three Black Crows (3 แท่งแดงต่อเนื่อง) ────────────────────
    if c0 is not None and c1 is not None and c2 is not None:
        if (_is_bear(c0) and _is_bear(c1) and _is_bear(c2) and
                _body(c0) > avg_body * 0.8 and _body(c1) > avg_body * 0.8 and
                float(c0['Close']) < float(c1['Close']) < float(c2['Close'])):
            results.append({
                "pattern":     "Three Black Crows",
                "type":        "SELL",
                "strength":    "STRONG",
                "confidence":  82,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Close']),
                "description": "อีกา 3 ตัว — ขาลงต่อเนื่อง แรงขายสม่ำเสมอ",
                "tip":         "ขาลงแข็งแกร่ง ควรหลีกเลี่ยงการซื้อจนกว่า pattern จะจบ",
                "emoji":       "🐦🐦🐦",
            })

    # ── 13. Upper Shadow Long (ไส้บนยาวในขาขึ้น) ──────────────────────
    if c0 is not None:
        uw = _upper_wick(c0)
        if uw > _body(c0) * 2.5 and uw > avg_range * 0.4:
            # Is there an uptrend? Check last 5 closes
            if len(df) >= 6:
                prev5 = df['Close'].iloc[-6:-1].mean()
                if float(c0['Close']) > prev5:  # in uptrend
                    results.append({
                        "pattern":     "Long Upper Shadow",
                        "type":        "SELL",
                        "strength":    "MEDIUM",
                        "confidence":  58,
                        "date":        date0,
                        "bar_index":   len(df) - 1,
                        "price":       float(c0['High']),
                        "description": "ไส้บนยาวในขาขึ้น — ฝั่งขายเริ่มต้านแรง",
                        "tip":         "ระวังการกลับตัว เฉพาะถ้าใกล้แนวต้านสำคัญ",
                        "emoji":       "⚠️",
                    })

    # ── 14. Tweezer Top (2 แท่ง high เท่ากัน) ─────────────────────────
    if c0 is not None and c1 is not None:
        hi_diff = abs(float(c0['High']) - float(c1['High'])) / max(float(c1['High']), 1e-9)
        if hi_diff < 0.003 and _is_bull(c1) and _is_bear(c0):
            results.append({
                "pattern":     "Tweezer Top",
                "type":        "SELL",
                "strength":    "MEDIUM",
                "confidence":  65,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['High']),
                "description": "แนวต้านคู่ (Tweezer Top) — ราคาขึ้นถึงจุดเดิมสองครั้ง",
                "tip":         "บ่งชี้แนวต้านแข็งแกร่ง โอกาสพักตัวหรือกลับทิศ",
                "emoji":       "🔱",
            })

    # ── 15. Tweezer Bottom ─────────────────────────────────────────────
    if c0 is not None and c1 is not None:
        lo_diff = abs(float(c0['Low']) - float(c1['Low'])) / max(float(c1['Low']), 1e-9)
        if lo_diff < 0.003 and _is_bear(c1) and _is_bull(c0):
            results.append({
                "pattern":     "Tweezer Bottom",
                "type":        "BUY",
                "strength":    "MEDIUM",
                "confidence":  65,
                "date":        date0,
                "bar_index":   len(df) - 1,
                "price":       float(c0['Low']),
                "description": "แนวรับคู่ (Tweezer Bottom) — ราคาลงถึงจุดเดิมสองครั้ง",
                "tip":         "แนวรับแข็งแกร่ง ถ้าปิดเหนือ high ของ c0 = สัญญาณซื้อ",
                "emoji":       "🧲",
            })

    return sorted(results, key=lambda x: x['confidence'], reverse=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: BELL CURVE / MEAN REVERSION ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyze_bell_curve(df: pd.DataFrame, window: int = 60) -> dict:
    """
    วิเคราะห์ Bell Curve + Mean Reversion
    ใช้ rolling window ล่าสุด เพื่อดูว่าราคาปัจจุบัน "ไกลจาก mean" แค่ไหน

    Returns dict:
        mean, std, z_score, percentile,
        reversion_probability, regime (STRETCHED/NORMAL/COMPRESSED),
        return_distribution (สำหรับ plot histogram),
        bb_position (ตำแหน่งใน Bollinger Band)
    """
    if df.empty or len(df) < 20:
        return {}

    closes = df['Close'].dropna()
    n = min(window, len(closes))
    recent = closes.iloc[-n:]

    current = float(recent.iloc[-1])
    mean    = float(recent.mean())
    std     = float(recent.std())
    if std == 0:
        return {}

    # ── Z-score (ห่างจาก mean กี่ sigma) ─────────────────────────────
    z_score = (current - mean) / std

    # ── Percentile ────────────────────────────────────────────────────
    percentile = float(stats.percentileofscore(recent, current))

    # ── Return distribution (% change day-over-day) ───────────────────
    returns = closes.pct_change().dropna().iloc[-(n-1):] * 100
    ret_mean = float(returns.mean())
    ret_std  = float(returns.std())
    ret_last = float(returns.iloc[-1]) if len(returns) > 0 else 0.0

    # Current return Z-score
    ret_z = (ret_last - ret_mean) / ret_std if ret_std > 0 else 0.0

    # ── Mean Reversion Probability ────────────────────────────────────
    # Based on: how often price returns to mean within 5 bars when |z| > threshold
    # Simple model: ยิ่ง |z| สูง โอกาส revert สูง (แต่ไม่ linear)
    abs_z = abs(z_score)
    if abs_z > 2.5:
        rev_prob = 0.85
        regime   = "STRETCHED_EXTREME"
        regime_th = "ยืดตัวมาก (Extreme)"
    elif abs_z > 2.0:
        rev_prob = 0.75
        regime   = "STRETCHED_HIGH"
        regime_th = "ยืดตัวสูง"
    elif abs_z > 1.5:
        rev_prob = 0.62
        regime   = "STRETCHED"
        regime_th = "ยืดตัว"
    elif abs_z > 1.0:
        rev_prob = 0.48
        regime   = "NORMAL"
        regime_th = "ปกติ"
    else:
        rev_prob = 0.35
        regime   = "COMPRESSED"
        regime_th = "หดตัว (Coiling)"

    direction = "กลับขึ้น" if z_score < 0 else "กลับลง"

    # ── Bollinger Band position ────────────────────────────────────────
    if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
        bb_up = float(df['BB_upper'].iloc[-1])
        bb_lo = float(df['BB_lower'].iloc[-1])
        bb_mid = float(df['BB_middle'].iloc[-1]) if 'BB_middle' in df.columns else mean
        bb_range = bb_up - bb_lo
        bb_pos = (current - bb_lo) / bb_range if bb_range > 0 else 0.5
        bb_label = (
            "เหนือ BB บน (Overbought)" if current > bb_up else
            "ใต้ BB ล่าง (Oversold)"   if current < bb_lo else
            f"ใน BB ({bb_pos*100:.0f}% จากล่าง)"
        )
    else:
        bb_pos, bb_label, bb_up, bb_lo, bb_mid = 0.5, "N/A", mean+2*std, mean-2*std, mean

    # ── Historical Z-scores (for chart) ────────────────────────────────
    rolling_mean = closes.rolling(n).mean()
    rolling_std  = closes.rolling(n).std()
    z_series = (closes - rolling_mean) / rolling_std.replace(0, np.nan)

    return {
        "current":          current,
        "mean":             mean,
        "std":              std,
        "z_score":          round(z_score, 3),
        "percentile":       round(percentile, 1),
        "regime":           regime,
        "regime_th":        regime_th,
        "reversion_prob":   round(rev_prob * 100, 1),
        "direction":        direction,
        "returns":          returns,
        "ret_mean":         round(ret_mean, 3),
        "ret_std":          round(ret_std, 3),
        "ret_last":         round(ret_last, 3),
        "ret_z":            round(ret_z, 3),
        "bb_pos":           round(bb_pos, 3),
        "bb_label":         bb_label,
        "bb_upper":         round(bb_up, 2),
        "bb_lower":         round(bb_lo, 2),
        "bb_middle":        round(bb_mid, 2),
        "z_series":         z_series,
        "price_series":     closes.iloc[-n:],
        "window":           n,
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: CHARTS
# ══════════════════════════════════════════════════════════════════════

def plot_candlestick_analysis(df: pd.DataFrame, patterns: list, symbol: str = "") -> go.Figure:
    """
    กราฟแท่งเทียนพร้อม annotation ทุก pattern
    """
    fig = go.Figure()

    # ── Candlestick ────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="OHLC",
        increasing_line_color='#00ff88', increasing_fillcolor='#00ff88',
        decreasing_line_color='#ff4444', decreasing_fillcolor='#ff4444',
        hovertext=[
            f"<b>{str(d)[:10]}</b><br>"
            f"O: {o:.2f}  H: {h:.2f}<br>"
            f"L: {l:.2f}  C: {c:.2f}<br>"
            f"Change: {((c-o)/o*100):+.2f}%"
            for d, o, h, l, c in zip(
                df.index, df['Open'], df['High'], df['Low'], df['Close']
            )
        ],
        hoverinfo="text",
    ))

    # ── Volume bars (small subplot-like at bottom) ─────────────────────
    if 'Volume' in df.columns:
        vol_max = df['Volume'].max()
        price_range = df['High'].max() - df['Low'].min()
        price_min = df['Low'].min()
        vol_scale = price_range * 0.12 / max(vol_max, 1)
        vol_colors = ['rgba(0,255,136,0.3)' if c >= o else 'rgba(255,68,68,0.3)'
                      for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(
            x=df.index, y=df['Volume'] * vol_scale,
            base=price_min - price_range * 0.02,
            marker_color=vol_colors, name="Volume",
            hovertemplate="Volume: %{customdata:,.0f}<extra></extra>",
            customdata=df['Volume'],
            showlegend=True,
        ))

    # ── EMA lines ─────────────────────────────────────────────────────
    ema_styles = [('EMA9','#FFD700',1),('EMA21','#00BFFF',1),('EMA50','#FF6B6B',1.5)]
    for col, color, width in ema_styles:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=col,
                line=dict(color=color, width=width),
                hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>",
            ))

    # ── Pattern annotations ────────────────────────────────────────────
    for p in patterns:
        is_buy  = p['type'] == 'BUY'
        color   = '#00ff88' if is_buy else '#ff4444' if p['type'] == 'SELL' else '#ffd700'
        ay      = -50  if is_buy else 50
        ay_side = 'below' if is_buy else 'above'

        # Arrow annotation
        bar_x = p['date']
        if is_buy:
            bar_y = float(df.loc[bar_x, 'Low']) * 0.998 if bar_x in df.index else p['price']
        else:
            bar_y = float(df.loc[bar_x, 'High']) * 1.002 if bar_x in df.index else p['price']

        fig.add_annotation(
            x=bar_x, y=bar_y,
            text=f"{p['emoji']} {p['pattern']}<br><small>{p['confidence']}%</small>",
            showarrow=True,
            arrowhead=2, arrowsize=1.2, arrowwidth=2,
            arrowcolor=color,
            ax=0, ay=ay,
            font=dict(color=color, size=10),
            bgcolor="rgba(10,10,20,0.85)",
            bordercolor=color,
            borderwidth=1,
        )

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"📊 Candlestick Analysis — {symbol}", font=dict(size=16)),
        template="plotly_dark",
        height=520,
        xaxis=dict(
            rangeslider=dict(visible=False),
            showspikes=True, spikemode='across', spikethickness=1,
            spikecolor='rgba(255,255,255,0.3)',
        ),
        yaxis=dict(showspikes=True, spikethickness=1, spikecolor='rgba(255,255,255,0.3)'),
        hovermode="x unified",
        hoverlabel=dict(bgcolor='rgba(20,22,35,0.95)', font=dict(family='monospace', size=11)),
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


def plot_bell_curve(bc: dict, symbol: str = "") -> go.Figure:
    """
    Bell Curve + Z-score chart แบบ 3 panel:
    1. Price Distribution (histogram + normal curve)
    2. Return Distribution
    3. Z-score over time
    """
    if not bc:
        return go.Figure()

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f"📊 Price Distribution (last {bc['window']} days)",
            "📈 Return Distribution (% daily change)",
            "📉 Z-score over time",
        ),
        vertical_spacing=0.10,
        row_heights=[0.4, 0.3, 0.3],
    )

    # ── Panel 1: Price histogram + Bell curve ─────────────────────────
    prices = bc['price_series']
    mean, std = bc['mean'], bc['std']
    current = bc['current']

    fig.add_trace(go.Histogram(
        x=prices, nbinsx=30,
        name="Price Distribution",
        marker_color='rgba(0,191,255,0.4)',
        marker_line_color='rgba(0,191,255,0.8)',
        marker_line_width=0.5,
        histnorm='probability density',
    ), row=1, col=1)

    # Normal curve overlay
    x_bell = np.linspace(mean - 4*std, mean + 4*std, 200)
    y_bell = stats.norm.pdf(x_bell, mean, std)
    fig.add_trace(go.Scatter(
        x=x_bell, y=y_bell, name="Normal Curve",
        line=dict(color='#ffd700', width=2),
    ), row=1, col=1)

    # Sigma lines
    sigma_styles = [
        (1,  'rgba(0,255,136,0.4)', '±1σ (68%)'),
        (2,  'rgba(255,165,0,0.4)', '±2σ (95%)'),
        (3,  'rgba(255,68,68,0.4)', '±3σ (99.7%)'),
    ]
    for mult, color, label in sigma_styles:
        for sign in [-1, 1]:
            fig.add_vline(
                x=mean + sign*mult*std, row=1, col=1,
                line=dict(color=color, width=1, dash='dash'),
                annotation_text=label if sign == 1 else "",
                annotation_font_size=9,
            )

    # Current price line
    z_color = '#ff4444' if abs(bc['z_score']) > 2 else '#ffd700' if abs(bc['z_score']) > 1 else '#00ff88'
    fig.add_vline(
        x=current, row=1, col=1,
        line=dict(color=z_color, width=2.5),
        annotation_text=f"ราคาปัจจุบัน<br>z={bc['z_score']:+.2f}",
        annotation_font_color=z_color,
        annotation_font_size=10,
    )
    fig.add_vline(
        x=mean, row=1, col=1,
        line=dict(color='white', width=1.5, dash='dot'),
        annotation_text=f"Mean\n{mean:.2f}",
        annotation_font_size=9,
    )

    # ── Panel 2: Return distribution ──────────────────────────────────
    rets = bc['returns']
    ret_mean, ret_std = bc['ret_mean'], bc['ret_std']
    ret_last = bc['ret_last']

    fig.add_trace(go.Histogram(
        x=rets, nbinsx=25,
        name="Daily Returns",
        marker_color='rgba(255,107,107,0.4)',
        marker_line_color='rgba(255,107,107,0.8)',
        marker_line_width=0.5,
        histnorm='probability density',
    ), row=2, col=1)

    x_ret = np.linspace(ret_mean - 4*ret_std, ret_mean + 4*ret_std, 200)
    y_ret = stats.norm.pdf(x_ret, ret_mean, ret_std)
    fig.add_trace(go.Scatter(
        x=x_ret, y=y_ret, name="Return Normal Curve",
        line=dict(color='#ffd700', width=2), showlegend=False,
    ), row=2, col=1)

    fig.add_vline(
        x=ret_last, row=2, col=1,
        line=dict(color='#00ff88' if ret_last >= 0 else '#ff4444', width=2),
        annotation_text=f"วันนี้ {ret_last:+.2f}%<br>z={bc['ret_z']:+.2f}",
        annotation_font_size=9,
    )
    fig.add_vline(
        x=0, row=2, col=1,
        line=dict(color='white', width=1, dash='dot'),
    )

    # ── Panel 3: Z-score time series ──────────────────────────────────
    z_series = bc['z_series'].dropna()
    z_colors = ['#ff4444' if z > 2 else '#00ff88' if z < -2 else
                '#ffa500' if abs(z) > 1 else '#888888'
                for z in z_series]

    fig.add_trace(go.Bar(
        x=z_series.index, y=z_series,
        name="Z-score",
        marker_color=z_colors,
        hovertemplate="Z-score: %{y:.2f}<extra></extra>",
    ), row=3, col=1)

    # Reference lines
    for level, color, label in [
        (2, 'rgba(255,68,68,0.6)', '+2σ'),
        (-2,'rgba(255,68,68,0.6)', '-2σ'),
        (1, 'rgba(255,165,0,0.4)',  '+1σ'),
        (-1,'rgba(255,165,0,0.4)',  '-1σ'),
        (0, 'rgba(255,255,255,0.3)','Mean'),
    ]:
        fig.add_hline(y=level, row=3, col=1,
                      line=dict(color=color, width=1, dash='dash'),
                      annotation_text=label, annotation_font_size=8)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=f"🔔 Bell Curve & Mean Reversion Analysis — {symbol}", font=dict(size=15)),
        template="plotly_dark",
        height=700,
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor='rgba(20,22,35,0.95)', font=dict(family='monospace', size=10)),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig
