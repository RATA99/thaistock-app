import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.indicators import find_support_resistance, detect_candlestick_patterns


def get_market_regime(df: pd.DataFrame) -> str:
    """ระบุ market regime จาก ADX + EMA200"""
    try:
        last = df.iloc[-1]
        adx  = last.get('ADX', 0) or 0
        price = last['Close']
        ema200 = last.get('EMA200', price) or price
        di_plus  = last.get('DI_plus', 0) or 0
        di_minus = last.get('DI_minus', 0) or 0

        if adx > 25:
            if price > ema200 and di_plus > di_minus:
                return "BULL_TREND"
            elif price < ema200 and di_minus > di_plus:
                return "BEAR_TREND"
            else:
                return "TRANSITION"
        else:
            # Low ADX = sideways
            return "SIDEWAYS"
    except:
        return "SIDEWAYS"


def calculate_signal_score(df: pd.DataFrame) -> tuple:
    """
    คำนวณ signal score 0-100 พร้อมรายการสัญญาณ
    Returns: (score: int, signals: list[dict], regime: str)
    """
    signals = []
    score   = 50  # neutral start
    regime  = get_market_regime(df)

    if df.empty or len(df) < 5:
        return score, signals, regime

    last  = df.iloc[-1]
    prev  = df.iloc[-2]

    # ── TREND SIGNALS (max ±40 pts) ───────────────────────────────────

    # EMA Alignment
    try:
        ema9  = last.get('EMA9', 0) or 0
        ema21 = last.get('EMA21', 0) or 0
        ema50 = last.get('EMA50', 0) or 0
        ema200= last.get('EMA200', 0) or 0
        close = last['Close']

        if ema9 > ema21 > ema50 > ema200 and close > ema9:
            score += 15
            signals.append({
                "type": "BUY", "strength": "STRONG",
                "reason": "EMA เรียงตัวสมบูรณ์ (9>21>50>200) — แนวโน้มขาขึ้นแรง"
            })
        elif ema9 < ema21 < ema50 < ema200 and close < ema9:
            score -= 15
            signals.append({
                "type": "SELL", "strength": "STRONG",
                "reason": "EMA เรียงตัวลง (9<21<50<200) — แนวโน้มขาลงแรง"
            })
        elif close > ema50:
            score += 7
            signals.append({
                "type": "BUY", "strength": "MEDIUM",
                "reason": "ราคาอยู่เหนือ EMA50 — แนวโน้มระยะกลางขาขึ้น"
            })
        elif close < ema50:
            score -= 7
            signals.append({
                "type": "SELL", "strength": "MEDIUM",
                "reason": "ราคาต่ำกว่า EMA50 — แนวโน้มระยะกลางขาลง"
            })
    except:
        pass

    # Golden/Death Cross (EMA9 vs EMA21)
    try:
        prev_ema9  = prev.get('EMA9', 0) or 0
        prev_ema21 = prev.get('EMA21', 0) or 0
        ema9  = last.get('EMA9', 0) or 0
        ema21 = last.get('EMA21', 0) or 0

        if prev_ema9 < prev_ema21 and ema9 > ema21:
            score += 12
            signals.append({
                "type": "BUY", "strength": "STRONG",
                "reason": "Golden Cross EMA9 ตัด EMA21 ขึ้น — สัญญาณซื้อ"
            })
        elif prev_ema9 > prev_ema21 and ema9 < ema21:
            score -= 12
            signals.append({
                "type": "SELL", "strength": "STRONG",
                "reason": "Death Cross EMA9 ตัด EMA21 ลง — สัญญาณขาย"
            })
    except:
        pass

    # Price vs EMA200 (long-term trend)
    try:
        ema200 = last.get('EMA200', 0) or last['Close']
        if ema200 > 0:
            diff_pct = (last['Close'] - ema200) / ema200 * 100
            if diff_pct > 5:
                score += 8
                signals.append({
                    "type": "BUY", "strength": "MEDIUM",
                    "reason": f"ราคาอยู่เหนือ EMA200 (+{diff_pct:.1f}%) — long-term uptrend"
                })
            elif diff_pct < -5:
                score -= 8
                signals.append({
                    "type": "SELL", "strength": "MEDIUM",
                    "reason": f"ราคาต่ำกว่า EMA200 ({diff_pct:.1f}%) — long-term downtrend"
                })
    except:
        pass

    # ── MOMENTUM SIGNALS (max ±30 pts) ────────────────────────────────

    # RSI
    try:
        rsi = last.get('RSI', 50) or 50
        if rsi < 30:
            score += 12
            signals.append({
                "type": "BUY", "strength": "STRONG",
                "reason": f"RSI={rsi:.1f} — Oversold อาจเด้งกลับ"
            })
        elif rsi > 70:
            score -= 12
            signals.append({
                "type": "SELL", "strength": "STRONG",
                "reason": f"RSI={rsi:.1f} — Overbought อาจปรับลง"
            })
        elif 40 <= rsi <= 60:
            signals.append({
                "type": "NEUTRAL", "strength": "WEAK",
                "reason": f"RSI={rsi:.1f} — อยู่ในโซนกลาง"
            })
    except:
        pass

    # MACD Crossover
    try:
        macd     = last.get('MACD', 0) or 0
        macd_sig = last.get('MACD_signal', 0) or 0
        p_macd   = prev.get('MACD', 0) or 0
        p_sig    = prev.get('MACD_signal', 0) or 0

        if p_macd < p_sig and macd > macd_sig:
            score += 10
            signals.append({
                "type": "BUY", "strength": "STRONG",
                "reason": "MACD ตัด Signal line ขึ้น — สัญญาณซื้อ momentum"
            })
        elif p_macd > p_sig and macd < macd_sig:
            score -= 10
            signals.append({
                "type": "SELL", "strength": "STRONG",
                "reason": "MACD ตัด Signal line ลง — สัญญาณขาย momentum"
            })
        elif macd > macd_sig and macd > 0:
            score += 5
            signals.append({
                "type": "BUY", "strength": "WEAK",
                "reason": "MACD > Signal และ > 0 — momentum เป็นบวก"
            })
        elif macd < macd_sig and macd < 0:
            score -= 5
            signals.append({
                "type": "SELL", "strength": "WEAK",
                "reason": "MACD < Signal และ < 0 — momentum เป็นลบ"
            })
    except:
        pass

    # StochRSI
    try:
        k = last.get('StochRSI_k', 50) or 50
        d = last.get('StochRSI_d', 50) or 50
        p_k = prev.get('StochRSI_k', 50) or 50
        p_d = prev.get('StochRSI_d', 50) or 50

        if p_k < p_d and k > d and k < 30:
            score += 8
            signals.append({
                "type": "BUY", "strength": "MEDIUM",
                "reason": f"StochRSI ตัดขึ้น ({k:.1f}) ในโซน Oversold"
            })
        elif p_k > p_d and k < d and k > 70:
            score -= 8
            signals.append({
                "type": "SELL", "strength": "MEDIUM",
                "reason": f"StochRSI ตัดลง ({k:.1f}) ในโซน Overbought"
            })
    except:
        pass

    # ── VOLUME SIGNALS (max ±20 pts) ─────────────────────────────────

    try:
        vol_ratio = last.get('Vol_ratio', 1) or 1
        obv_now   = last.get('OBV', 0) or 0
        obv_prev  = df['OBV'].iloc[-10] if len(df) >= 10 else obv_now

        if vol_ratio > 2.0 and last['Close'] > prev['Close']:
            score += 10
            signals.append({
                "type": "BUY", "strength": "STRONG",
                "reason": f"Volume สูงผิดปกติ ({vol_ratio:.1f}x) ขณะราคาขึ้น — Breakout แรง"
            })
        elif vol_ratio > 2.0 and last['Close'] < prev['Close']:
            score -= 10
            signals.append({
                "type": "SELL", "strength": "STRONG",
                "reason": f"Volume สูงผิดปกติ ({vol_ratio:.1f}x) ขณะราคาลง — Breakdown แรง"
            })
        elif vol_ratio < 0.5:
            signals.append({
                "type": "NEUTRAL", "strength": "WEAK",
                "reason": f"Volume ต่ำผิดปกติ ({vol_ratio:.1f}x) — การเคลื่อนไหวอ่อนแอ"
            })

        # OBV trend
        if obv_now > obv_prev and last['Close'] > prev['Close']:
            score += 7
            signals.append({
                "type": "BUY", "strength": "MEDIUM",
                "reason": "OBV เพิ่มขึ้นพร้อมราคา — แรงซื้อสะสม (Accumulation)"
            })
        elif obv_now < obv_prev and last['Close'] < prev['Close']:
            score -= 7
            signals.append({
                "type": "SELL", "strength": "MEDIUM",
                "reason": "OBV ลดลงพร้อมราคา — แรงขายกระจาย (Distribution)"
            })
    except:
        pass

    # Bollinger Band signals
    try:
        bb_lower = last.get('BB_lower', 0) or 0
        bb_upper = last.get('BB_upper', 0) or 0
        close    = last['Close']

        if bb_lower > 0 and close <= bb_lower * 1.01:
            score += 5
            signals.append({
                "type": "BUY", "strength": "MEDIUM",
                "reason": "ราคาแตะ Bollinger Band ล่าง — โอกาสเด้งกลับ"
            })
        elif bb_upper > 0 and close >= bb_upper * 0.99:
            score -= 5
            signals.append({
                "type": "SELL", "strength": "MEDIUM",
                "reason": "ราคาแตะ Bollinger Band บน — อาจปรับลง"
            })
    except:
        pass

    # ── PATTERN SIGNALS (10 pts) ─────────────────────────────────────
    try:
        patterns = detect_candlestick_patterns(df)
        for p in patterns:
            if p['type'] == 'BUY':
                score += 5
            elif p['type'] == 'SELL':
                score -= 5
            signals.append({
                "type": p['type'],
                "strength": "MEDIUM",
                "reason": f"{p['pattern']}: {p['description_th']}"
            })
    except:
        pass

    # Clamp to 0-100
    score = max(0, min(100, score))

    return int(score), signals, regime


def calculate_price_targets(df: pd.DataFrame, current_price: float) -> dict:
    """คำนวณจุดซื้อ-ขาย, stop loss, target prices"""
    supports, resistances = find_support_resistance(df)

    # ATR for dynamic SL
    atr = df['ATR'].iloc[-1] if 'ATR' in df.columns and not pd.isna(df['ATR'].iloc[-1]) else current_price * 0.02

    # ── Fibonacci Retracement ────────────────────────────────────────
    period_high = df['High'].rolling(60).max().iloc[-1]
    period_low  = df['Low'].rolling(60).min().iloc[-1]
    fib_range   = period_high - period_low

    fibonacci = {
        "0.0 (Low)":   round(period_low, 2),
        "0.236":       round(period_low + fib_range * 0.236, 2),
        "0.382":       round(period_low + fib_range * 0.382, 2),
        "0.500":       round(period_low + fib_range * 0.500, 2),
        "0.618":       round(period_low + fib_range * 0.618, 2),
        "0.786":       round(period_low + fib_range * 0.786, 2),
        "1.0 (High)":  round(period_high, 2),
    }

    # ── Buy Zone ─────────────────────────────────────────────────────
    fib_618 = fibonacci["0.618"]
    if supports:
        nearest_sup = supports[0]
    else:
        nearest_sup = current_price * 0.95

    buy_low  = round(min(nearest_sup, fib_618) * 0.99, 2)
    buy_high = round(max(nearest_sup, fib_618) * 1.01, 2)

    # ── Stop Loss ────────────────────────────────────────────────────
    stop_loss    = round(buy_low - atr * 1.5, 2)
    risk_amount  = current_price - stop_loss
    risk_pct     = (risk_amount / current_price * 100) if current_price > 0 else 5.0

    # ── Targets ──────────────────────────────────────────────────────
    tp1 = round(current_price + risk_amount * 2, 2)   # R:R 1:2
    tp2 = round(current_price + risk_amount * 3, 2)   # R:R 1:3
    if resistances:
        tp3 = round(resistances[0], 2)
    else:
        tp3 = round(current_price + risk_amount * 4, 2)

    risk_reward = risk_amount / (tp1 - current_price) if (tp1 - current_price) > 0 else 0.5
    trailing_stop = round(current_price - atr * 2, 2)

    return {
        "buy_zone":          {"low": buy_low, "high": buy_high},
        "stop_loss":         stop_loss,
        "targets":           [tp1, tp2, tp3],
        "trailing_stop":     trailing_stop,
        "risk_amount_pct":   round(abs(risk_pct), 2),
        "risk_reward":       round(risk_reward, 3),
        "fibonacci":         fibonacci,
        "support_levels":    [round(s, 2) for s in supports],
        "resistance_levels": [round(r, 2) for r in resistances],
    }


def run_backtest(df: pd.DataFrame, strategy: str, capital: float, sl_pct: float) -> dict:
    """Backtest trading strategy บน historical data"""
    import plotly.graph_objects as go

    equity      = capital
    trades      = []
    in_position = False
    entry_price = 0.0
    entry_date  = None
    equity_curve_x = [df.index[0]]
    equity_curve_y = [capital]

    closes = df['Close'].values
    dates  = df.index

    for i in range(2, len(df)):
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        pp   = df.iloc[i - 2]
        close = closes[i]

        # ── Generate Entry Signal ─────────────────────────────────────
        entry_signal = False
        exit_signal  = False

        if strategy == "EMA Crossover (9/21)":
            ema9_now  = row.get('EMA9', 0) or 0
            ema21_now = row.get('EMA21', 0) or 0
            ema9_prev = prev.get('EMA9', 0) or 0
            ema21_prev= prev.get('EMA21', 0) or 0
            entry_signal = (ema9_prev < ema21_prev and ema9_now > ema21_now)
            exit_signal  = (ema9_prev > ema21_prev and ema9_now < ema21_now)

        elif strategy == "RSI Oversold/Overbought":
            rsi_now  = row.get('RSI', 50) or 50
            rsi_prev = prev.get('RSI', 50) or 50
            entry_signal = (rsi_prev < 30 and rsi_now >= 30)
            exit_signal  = (rsi_prev < 70 and rsi_now >= 70)

        elif strategy == "MACD Crossover":
            macd_now  = row.get('MACD', 0) or 0
            sig_now   = row.get('MACD_signal', 0) or 0
            macd_prev = prev.get('MACD', 0) or 0
            sig_prev  = prev.get('MACD_signal', 0) or 0
            entry_signal = (macd_prev < sig_prev and macd_now > sig_now)
            exit_signal  = (macd_prev > sig_prev and macd_now < sig_now)

        elif strategy == "Bollinger Band Bounce":
            bb_lower  = row.get('BB_lower', 0) or 0
            bb_upper  = row.get('BB_upper', 0) or 0
            entry_signal = (bb_lower > 0 and close <= bb_lower * 1.005)
            exit_signal  = (bb_upper > 0 and close >= bb_upper * 0.995)

        elif strategy == "Combined Signal Score > 65":
            sc, _, _ = calculate_signal_score(df.iloc[:i + 1])
            entry_signal = sc >= 65
            exit_signal  = sc <= 35

        # ── Execute Trades ────────────────────────────────────────────
        if not in_position and entry_signal:
            in_position = True
            entry_price = close
            entry_date  = dates[i]

        elif in_position:
            sl_price = entry_price * (1 - sl_pct)
            if exit_signal or close <= sl_price:
                pnl_pct = (close - entry_price) / entry_price * 100
                pnl_thb = equity * (close - entry_price) / entry_price
                equity  += pnl_thb
                trades.append({
                    "วันที่เข้า": entry_date.strftime("%Y-%m-%d"),
                    "วันที่ออก": dates[i].strftime("%Y-%m-%d"),
                    "ราคาเข้า": round(entry_price, 2),
                    "ราคาออก": round(close, 2),
                    "กำไร/ขาดทุน %": round(pnl_pct, 2),
                    "กำไร/ขาดทุน THB": round(pnl_thb, 2),
                    "ผลลัพธ์": "✅ กำไร" if pnl_pct > 0 else "❌ ขาดทุน"
                })
                in_position = False

        equity_curve_x.append(dates[i])
        equity_curve_y.append(round(equity, 2))

    # ── Compute Stats ─────────────────────────────────────────────────
    if not trades:
        total_return = 0.0
        win_rate     = 0.0
        max_dd       = 0.0
    else:
        total_return = (equity - capital) / capital * 100
        wins    = sum(1 for t in trades if t["กำไร/ขาดทุน %"] > 0)
        win_rate = wins / len(trades) * 100

        # Max drawdown from equity curve
        eq_arr   = np.array(equity_curve_y)
        peak     = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak * 100
        max_dd   = float(drawdown.min())

    # ── Equity Curve Chart ────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_curve_x, y=equity_curve_y,
        mode='lines', name='Equity',
        line=dict(color='#00ff88', width=2),
        fill='tozeroy', fillcolor='rgba(0,255,136,0.1)'
    ))
    fig.update_layout(
        title="📈 Equity Curve",
        template='plotly_dark',
        height=350,
        xaxis_title="วันที่",
        yaxis_title="มูลค่าพอร์ต (THB)",
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
    )

    return {
        "total_return":  round(total_return, 2),
        "win_rate":      round(win_rate, 2),
        "max_drawdown":  round(max_dd, 2),
        "total_trades":  len(trades),
        "equity_curve":  fig,
        "trade_log":     pd.DataFrame(trades) if trades else pd.DataFrame(),
    }


# ═══════════════════════════════════════════════════════════════════════
# TRADE RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════

def generate_recommendation(
    df: pd.DataFrame,
    score: int,
    signals: list,
    regime: str,
    current_price: float,
    timeframe: str = "1D",
) -> dict:
    """
    สรุปคำแนะนำการเทรดจากการวิเคราะห์ทางเทคนิคทั้งหมด
    
    Returns dict:
        action:     "BUY" | "ACCUMULATE" | "HOLD" | "REDUCE" | "SELL" | "WAIT"
        confidence: "HIGH" | "MEDIUM" | "LOW"
        title_th:   ชื่อสรุปภาษาไทย
        summary:    อธิบายสั้นๆ
        reasons:    list[str]  เหตุผลหลัก (max 5)
        cautions:   list[str]  ปัจจัยเสี่ยง
        entry_zone: (low, high) หรือ None
        stop_loss:  float หรือ None
        targets:    [tp1, tp2] หรือ []
        score:      int 0-100
        color:      hex color
        emoji:      emoji
    """
    if df.empty or len(df) < 5:
        return _neutral_rec(current_price, score)

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    # ── Extract indicators ────────────────────────────────────────────
    close   = float(last['Close'])
    rsi     = float(last.get('RSI', 50) or 50)
    macd    = float(last.get('MACD', 0) or 0)
    macd_sig= float(last.get('MACD_signal', 0) or 0)
    macd_h  = float(last.get('MACD_hist', 0) or 0)
    ema9    = float(last.get('EMA9',   close) or close)
    ema21   = float(last.get('EMA21',  close) or close)
    ema50   = float(last.get('EMA50',  close) or close)
    ema200  = float(last.get('EMA200', close) or close)
    atr     = float(last.get('ATR', close*0.02) or close*0.02)
    bb_up   = float(last.get('BB_upper', close*1.02) or close*1.02)
    bb_lo   = float(last.get('BB_lower', close*0.98) or close*0.98)
    vol_r   = float(last.get('Vol_ratio', 1) or 1)
    adx     = float(last.get('ADX', 0) or 0)
    di_p    = float(last.get('DI_plus', 0) or 0)
    di_m    = float(last.get('DI_minus', 0) or 0)

    prev_macd_h = float(prev.get('MACD_hist', 0) or 0)

    # ── Count buy/sell signals ────────────────────────────────────────
    buy_strong  = sum(1 for s in signals if s['type']=='BUY'  and s['strength']=='STRONG')
    buy_medium  = sum(1 for s in signals if s['type']=='BUY'  and s['strength']=='MEDIUM')
    sell_strong = sum(1 for s in signals if s['type']=='SELL' and s['strength']=='STRONG')
    sell_medium = sum(1 for s in signals if s['type']=='SELL' and s['strength']=='MEDIUM')

    # ── Key conditions ────────────────────────────────────────────────
    is_uptrend      = close > ema50 > 0
    is_strong_trend = adx > 25
    above_ema200    = close > ema200 if ema200 > 0 else True
    ema_aligned_up  = ema9 > ema21 > ema50
    ema_aligned_dn  = ema9 < ema21 < ema50
    macd_bullish    = macd > macd_sig and macd_h > 0
    macd_improving  = macd_h > prev_macd_h
    rsi_oversold    = rsi < 35
    rsi_overbought  = rsi > 70
    rsi_healthy     = 40 <= rsi <= 65
    near_bb_lower   = close <= bb_lo * 1.02
    near_bb_upper   = close >= bb_up * 0.98
    vol_confirm     = vol_r >= 1.5

    # ── Determine ACTION ─────────────────────────────────────────────
    reasons  = []
    cautions = []

    # === STRONG BUY CONDITIONS ===
    if (score >= 72 and ema_aligned_up and macd_bullish
            and above_ema200 and not rsi_overbought):
        action     = "BUY"
        title_th   = "🟢 ซื้อได้เลย"
        summary    = "สัญญาณเป็นบวกแข็งแกร่ง หลายตัวชี้พร้อมกัน โมเมนตัมดี"
        color      = "#00cc44"
        emoji      = "🟢"
        confidence = "HIGH" if score >= 80 else "MEDIUM"
        if ema_aligned_up:
            reasons.append(f"EMA เรียงตัวขาขึ้น (9 > 21 > 50) — trend ชัดเจน")
        if macd_bullish:
            reasons.append(f"MACD อยู่เหนือ Signal line ใน territory บวก")
        if above_ema200:
            reasons.append(f"ราคาอยู่เหนือ EMA200 — อยู่ใน Long-term uptrend")
        if vol_confirm:
            reasons.append(f"Volume สูง {vol_r:.1f}x — มีแรงซื้อหนุน")

    # === ACCUMULATE (ทยอยสะสม) ===
    elif (score >= 60 and (is_uptrend or rsi_oversold or near_bb_lower)
          and sell_strong == 0):
        action     = "ACCUMULATE"
        title_th   = "🔵 ทยอยสะสม"
        summary    = "สัญญาณเป็นบวกแต่ยังไม่แข็งแกร่งพอ แนะนำซื้อเป็นงวด"
        color      = "#4488ff"
        emoji      = "🔵"
        confidence = "MEDIUM"
        if rsi_oversold:
            reasons.append(f"RSI {rsi:.0f} — Oversold โอกาสเด้งกลับสูง")
        if near_bb_lower:
            reasons.append(f"ราคาแตะ Bollinger Band ล่าง — แนวรับแข็งแกร่ง")
        if is_uptrend:
            reasons.append(f"ราคายังอยู่เหนือ EMA50 — trend ใหญ่ยังขาขึ้น")
        if macd_improving:
            reasons.append(f"MACD Histogram กำลังดีขึ้น — momentum กลับมา")
        cautions.append("ยังไม่มี signal แตกหัก — ซื้อเป็นงวดดีกว่า all-in")

    # === HOLD ===
    elif 45 <= score < 60 and not rsi_overbought and not ema_aligned_dn:
        action     = "HOLD"
        title_th   = "⚪ ถือได้ / รอดูก่อน"
        summary    = "สัญญาณยังไม่ชัดเจน ทั้งขาขึ้นและขาลงยังเป็นไปได้"
        color      = "#aaaaaa"
        emoji      = "⚪"
        confidence = "LOW"
        reasons.append(f"Signal score {score}/100 — อยู่ใน neutral zone")
        if not is_strong_trend:
            reasons.append(f"ADX {adx:.0f} — ยังไม่มี trend แรงๆ (ต้องการ > 25)")
        cautions.append("รอให้ราคายืนยันทิศทางก่อนค่อยตัดสินใจ")
        cautions.append("ติดตาม volume — ถ้า volume เพิ่มขึ้นพร้อมราคาค่อยตัดสินใจ")

    # === REDUCE (ทยอยขาย) ===
    elif (40 <= score < 55 and (rsi_overbought or near_bb_upper or ema_aligned_dn)
          and sell_medium >= 1):
        action     = "REDUCE"
        title_th   = "🟡 ทยอยขาย / ลดน้ำหนัก"
        summary    = "สัญญาณเริ่มอ่อนแอ แนะนำลดสัดส่วนหรือขายบางส่วน"
        color      = "#ffaa00"
        emoji      = "🟡"
        confidence = "MEDIUM"
        if rsi_overbought:
            reasons.append(f"RSI {rsi:.0f} — Overbought โอกาสปรับลงสูง")
        if near_bb_upper:
            reasons.append(f"ราคาแตะ Bollinger Band บน — แนวต้านแข็งแกร่ง")
        if ema_aligned_dn:
            reasons.append(f"EMA เรียงตัวขาลง (9 < 21 < 50) — trend กลับด้าน")
        cautions.append("อย่าขายทั้งหมดทันที — อาจทยอยขาย 30-50% ก่อน")

    # === SELL ===
    elif (score < 40 and sell_strong >= 1
          and (ema_aligned_dn or (not above_ema200 and regime == "BEAR_TREND"))):
        action     = "SELL"
        title_th   = "🔴 ขาย / หลีกเลี่ยง"
        summary    = "สัญญาณลบหลายตัวพร้อมกัน แนวโน้มขาลงชัดเจน"
        color      = "#ff4444"
        emoji      = "🔴"
        confidence = "HIGH" if score < 30 else "MEDIUM"
        if ema_aligned_dn:
            reasons.append(f"EMA เรียงตัวขาลงสมบูรณ์ — downtrend ยืนยัน")
        if not above_ema200:
            reasons.append(f"ราคาต่ำกว่า EMA200 — Long-term trend เป็นลบ")
        if sell_strong > 0:
            reasons.append(f"มี {sell_strong} STRONG sell signal — แรงขายหนัก")
        cautions.append("ถ้ายังถือหุ้นอยู่ ตั้ง Stop Loss ทันที")

    # === WAIT ===
    else:
        action     = "WAIT"
        title_th   = "🟠 รอจังหวะ"
        summary    = "สัญญาณขัดแย้งกันอยู่ ยังไม่ใช่จังหวะที่ดีพอ"
        color      = "#ff8800"
        emoji      = "🟠"
        confidence = "LOW"
        reasons.append(f"Score {score}/100 — ต่ำกว่าเกณฑ์เข้าซื้อ (>60)")
        if sell_strong > 0 and buy_strong > 0:
            reasons.append("มีทั้ง buy และ sell signal แรง — สัญญาณขัดแย้ง")
        cautions.append("อย่าฝืนเข้า — รอให้สัญญาณชัดขึ้น")

    # ── Add regime context ────────────────────────────────────────────
    regime_notes = {
        "BULL_TREND":  "📈 อยู่ใน Bull Trend — โอกาสขาขึ้นมากกว่า",
        "BEAR_TREND":  "📉 อยู่ใน Bear Trend — ระวังการขาดทุน",
        "SIDEWAYS":    "↔️ ตลาด Sideways — ซื้อที่แนวรับ ขายที่แนวต้าน",
        "TRANSITION":  "⏳ กำลัง Transition — รอ breakout ยืนยัน",
    }
    if regime in regime_notes:
        reasons.append(regime_notes[regime])

    # ── Entry / SL / TP ───────────────────────────────────────────────
    entry_zone = stop_loss = None
    targets    = []

    if action in ("BUY", "ACCUMULATE"):
        entry_low  = round(close * 0.99, 2)
        entry_high = round(close * 1.005, 2)
        entry_zone = (entry_low, entry_high)
        stop_loss  = round(close - atr * 2.0, 2)
        tp1 = round(close + atr * 2.5, 2)
        tp2 = round(close + atr * 4.0, 2)
        targets = [tp1, tp2]
        risk_pct   = abs(close - stop_loss) / close * 100
        reward_pct = abs(tp1 - close) / close * 100
        rr = reward_pct / risk_pct if risk_pct > 0 else 0
        if rr < 1.5:
            cautions.append(f"R:R = 1:{rr:.1f} — ต่ำกว่าเกณฑ์ (ควร > 1:2)")
        else:
            reasons.append(f"R:R = 1:{rr:.1f} — คุ้มค่าความเสี่ยง")

    elif action in ("REDUCE", "SELL"):
        stop_loss = round(close + atr * 1.5, 2)  # for short / exit trigger

    # ── RSI context note ─────────────────────────────────────────────
    if rsi > 70 and action not in ("SELL", "REDUCE"):
        cautions.append(f"RSI {rsi:.0f} สูง — อาจมีการปรับฐานระยะสั้น")
    elif rsi < 30 and action not in ("BUY", "ACCUMULATE"):
        cautions.append(f"RSI {rsi:.0f} ต่ำมาก — oversold อาจเด้งได้")

    # Limit reasons/cautions
    reasons  = reasons[:5]
    cautions = cautions[:3]

    return {
        "action":     action,
        "confidence": confidence,
        "title_th":   title_th,
        "summary":    summary,
        "reasons":    reasons,
        "cautions":   cautions,
        "entry_zone": entry_zone,
        "stop_loss":  stop_loss,
        "targets":    targets,
        "score":      score,
        "color":      color,
        "emoji":      emoji,
        "regime":     regime,
        "rsi":        round(rsi, 1),
        "timeframe":  timeframe,
    }


def _neutral_rec(price: float, score: int) -> dict:
    return {
        "action": "WAIT", "confidence": "LOW",
        "title_th": "⚪ ข้อมูลไม่เพียงพอ",
        "summary": "ข้อมูลน้อยเกินไป ไม่สามารถวิเคราะห์ได้",
        "reasons": [], "cautions": ["กรุณาเลือก timeframe ที่ยาวขึ้น"],
        "entry_zone": None, "stop_loss": None, "targets": [],
        "score": score, "color": "#666666", "emoji": "⚪",
        "regime": "UNKNOWN", "rsi": 50, "timeframe": "?",
    }
