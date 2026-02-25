import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import os

from modules.data_fetcher import (
    get_historical_data, get_realtime_quote,
    get_dividend_history, get_stock_info,
    is_market_open, calculate_dividend_cagr,
    search_stocks, validate_symbol,
    resample_4h,
)
from modules.indicators import add_all_indicators
from modules.signals import calculate_signal_score, calculate_price_targets, run_backtest, generate_recommendation
from modules.charts import (
    plot_candlestick, plot_macd, plot_rsi,
    plot_dividend_chart, plot_fibonacci, plot_fibonacci_table
)
from modules.scanner import run_fibonacci_scan, run_multi_timeframe_scan, run_daytrade_scan, SET_UNIVERSE, DAYTRADE_INTERVALS
from modules.candle_analysis import detect_patterns_full, analyze_bell_curve, plot_candlestick_analysis, plot_bell_curve

# ─── PAGE CONFIG ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thai Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130; border-radius: 10px;
        padding: 15px; margin: 5px;
        border: 1px solid #2d3250;
    }
    .signal-buy {
        background: #0d3320; border: 1px solid #00ff88;
        border-radius: 8px; padding: 10px; margin-bottom: 8px;
    }
    .signal-sell {
        background: #330d0d; border: 1px solid #ff4444;
        border-radius: 8px; padding: 10px; margin-bottom: 8px;
    }
    .signal-neutral {
        background: #1e1e2e; border: 1px solid #666;
        border-radius: 8px; padding: 10px; margin-bottom: 8px;
    }
    .target-box {
        background: #1a1a2e; border-radius: 8px;
        padding: 12px; margin: 6px 0;
    }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: bold; }
    .regime-box {
        text-align: center; padding: 18px;
        border-radius: 10px; margin-top: 8px;
    }
    stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── SEARCH CACHE ─────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def cached_search(query: str) -> list:
    return search_stocks(query)

@st.cache_data(ttl=60)
def cached_validate(sym: str) -> bool:
    return validate_symbol(sym)

# ─── SMART CACHE TTL ──────────────────────────────────────────────────
# ตลาดเปิด: quote cache 60s, history cache 3min
# ตลาดปิด:  quote cache 15min, history cache 1hr
_mkt_open = is_market_open()
_quote_ttl   = 60    if _mkt_open else 900
_history_ttl = 180   if _mkt_open else 3600

@st.cache_data(ttl=_quote_ttl)
def fetch_realtime(sym: str) -> dict:
    return get_realtime_quote(sym)

@st.cache_data(ttl=_history_ttl)
def fetch_historical(sym: str, period: str, interval: str = "1d") -> pd.DataFrame:
    return get_historical_data(sym, period, interval)

@st.cache_data(ttl=86400)
def fetch_dividends(sym: str) -> pd.DataFrame:
    return get_dividend_history(sym)

@st.cache_data(ttl=3600)
def fetch_info(sym: str) -> dict:
    return get_stock_info(sym)

# ─── SIDEBAR ──────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 Thai Stock Analyzer")
    st.caption("ระบบวิเคราะห์หุ้นไทย SET / MAI")
    st.divider()

    # ── Free-text stock search ────────────────────────────────────────
    st.markdown("**🔍 ค้นหาหุ้น**")
    search_query = st.text_input(
        label="search_input",
        placeholder="พิมพ์ชื่อ เช่น PTT, KBANK, AOT...",
        label_visibility="collapsed",
        key="stock_search"
    )

    # Session state: remember selected symbol
    if "symbol" not in st.session_state:
        st.session_state.symbol = "PTT"

    # Show search results when user types
    if search_query and len(search_query.strip()) >= 1:
        with st.spinner("กำลังค้นหา..."):
            results = cached_search(search_query.strip())

        if results:
            st.markdown(f"<small style='color:#aaa'>พบ {len(results)} รายการ</small>",
                        unsafe_allow_html=True)
            for r in results:
                sym   = r['symbol']
                name  = r.get('name', sym)
                label = f"**{sym}** — {name[:35]}"
                if st.button(label, key=f"btn_{sym}", use_container_width=True):
                    st.session_state.symbol = sym
                    st.rerun()
        else:
            st.warning("ไม่พบหุ้นที่ตรงกัน")
            st.caption("ลองพิมพ์ชื่อใหม่ เช่น 'PTT' หรือ 'Kasikorn'")

    # Manual symbol entry
    st.markdown("**หรือใส่ Symbol ตรงๆ**")
    manual_sym = st.text_input(
        label="manual_sym",
        placeholder="เช่น PTT, ADVANC, BBL",
        label_visibility="collapsed",
        key="manual_symbol"
    ).strip().upper()

    if manual_sym:
        if st.button(f"✅ ใช้ {manual_sym}", type="primary", use_container_width=True):
            st.session_state.symbol = manual_sym
            st.rerun()

    # Current selection badge
    symbol = st.session_state.symbol
    st.markdown(f"""
    <div style='background:#1a1a2e; border:1px solid #ffd700; border-radius:8px;
         padding:10px; margin:8px 0; text-align:center'>
    <span style='color:#aaa; font-size:0.8rem'>หุ้นที่เลือก</span><br>
    <b style='color:#ffd700; font-size:1.4rem'>{symbol}</b>
    </div>
    """, unsafe_allow_html=True)

    # Quick access: Popular stocks
    st.markdown("**⭐ หุ้นยอดนิยม**")
    popular = [
        ["PTT", "ADVANC", "KBANK", "SCB"],
        ["AOT", "CPALL", "BDMS", "GULF"],
        ["DELTA", "MTC", "MINT", "SCC"],
    ]
    for row in popular:
        cols = st.columns(4)
        for col, sym_btn in zip(cols, row):
            if col.button(sym_btn, key=f"pop_{sym_btn}",
                          use_container_width=True,
                          type="primary" if sym_btn == symbol else "secondary"):
                st.session_state.symbol = sym_btn
                st.rerun()

    st.divider()

    # ── Timeframe selector ────────────────────────────────────────────
    st.markdown("**⏱ Timeframe**")

    # Group 1: Intraday
    st.caption("📍 Intraday")
    tf_intraday = st.radio(
        "intraday", ["5M","15M","30M","1H","4H"],
        index=None, horizontal=True, label_visibility="collapsed",
        key="tf_intra"
    )

    # Group 2: Daily/Weekly/Monthly
    st.caption("📅 Swing / Position")
    tf_swing = st.radio(
        "swing", ["1W","1M","3M","6M","1Y","3Y","5Y"],
        index=3, horizontal=True, label_visibility="collapsed",
        key="tf_swing"
    )

    # Determine active timeframe
    if tf_intraday:
        timeframe = tf_intraday
        # Clear swing selection via session state trick
        if st.session_state.get("_last_tf_group") != "intra":
            st.session_state["_last_tf_group"] = "intra"
    else:
        timeframe = tf_swing if tf_swing else "1M"
        st.session_state["_last_tf_group"] = "swing"

    # Map to yfinance period + interval
    TF_CONFIG = {
        # intraday: (period, interval, label)
        "5M":  ("5d",   "5m",  "5 นาที"),
        "15M": ("5d",   "15m", "15 นาที"),
        "30M": ("10d",  "30m", "30 นาที"),
        "1H":  ("30d",  "60m", "1 ชั่วโมง"),
        "4H":  ("60d",  "60m", "4 ชั่วโมง"),   # resample from 1H
        # daily+
        "1W":  ("1mo",  "1d",  "1 สัปดาห์"),
        "1M":  ("3mo",  "1d",  "1 เดือน"),
        "3M":  ("6mo",  "1d",  "3 เดือน"),
        "6M":  ("1y",   "1d",  "6 เดือน"),
        "1Y":  ("2y",   "1d",  "1 ปี"),
        "3Y":  ("5y",   "1d",  "3 ปี"),
        "5Y":  ("10y",  "1d",  "5 ปี"),
    }
    tf_cfg = TF_CONFIG.get(timeframe, TF_CONFIG["1M"])
    tf_period   = tf_cfg[0]
    tf_interval = tf_cfg[1]
    tf_label    = tf_cfg[2]
    is_intraday = tf_interval in ("5m","15m","30m","60m")

    # ── Timeframe description ─────────────────────────────────────────
    TF_DESC = {
        "5M":  "⚡ Scalp — เข้า-ออกภายใน 15–60 นาที",
        "15M": "⚡ Day trade — ปิดภายในวัน (แนะนำ)",
        "30M": "📊 Intraday swing — ข้ามคืนได้",
        "1H":  "📊 Short swing — 1–3 วัน",
        "4H":  "📊 Swing — 3–7 วัน (ยอดฮิต TA)",
        "1W":  "📈 Swing — 1–2 สัปดาห์",
        "1M":  "📈 Swing — 2–4 สัปดาห์",
        "3M":  "📈 Position — 1–3 เดือน ✨",
        "6M":  "📈 Position — 3–6 เดือน",
        "1Y":  "📈 Trend — Major S/R levels",
        "3Y":  "🏦 Long-term trend",
        "5Y":  "🏦 Macro / Investment grade",
    }
    st.caption(TF_DESC.get(timeframe, ""))

    st.divider()
    st.subheader("📊 Indicators")
    show_ema      = st.checkbox("EMA (9/21/50/200)", value=True)
    show_bb       = st.checkbox("Bollinger Bands",   value=True)
    show_ichimoku = st.checkbox("Ichimoku Cloud",    value=False)
    show_fibonacci= st.checkbox("📐 Fibonacci Retracement", value=False,
                                help="แสดง Fibonacci levels บนกราฟ พร้อม Golden Ratio 61.8%")
    show_volume   = st.checkbox("Volume Profile",    value=True)

    st.divider()

    # ── Smart Auto-Refresh ────────────────────────────────────────────
    st.subheader("🔄 Live Data")

    from datetime import timezone, timedelta
    _tz_bkk = timezone(timedelta(hours=7))
    _now = datetime.now(_tz_bkk)
    _h, _m = _now.hour, _now.minute
    _is_open = (
        _now.weekday() < 5 and
        ((10, 0) <= (_h, _m) <= (12, 30) or (14, 30) <= (_h, _m) <= (17, 0))
    )

    # Market status badge
    if _is_open:
        st.markdown("🟢 **ตลาดเปิดอยู่** — Live data")
    else:
        # Compute next open
        if _now.weekday() >= 5:
            days_to_mon = 7 - _now.weekday()
            _next = _now.replace(hour=10, minute=0, second=0) + timedelta(days=days_to_mon)
        elif (_h, _m) < (10, 0):
            _next = _now.replace(hour=10, minute=0, second=0)
        elif (12, 30) < (_h, _m) < (14, 30):
            _next = _now.replace(hour=14, minute=30, second=0)
        else:
            _next = (_now + timedelta(days=1)).replace(hour=10, minute=0, second=0)
            if _next.weekday() == 5: _next += timedelta(days=2)
            elif _next.weekday() == 6: _next += timedelta(days=1)
        _diff = int((_next - _now).total_seconds())
        _hh, _rem = divmod(_diff, 3600)
        _mm2, _ss = divmod(_rem, 60)
        st.markdown(f"🔴 **ตลาดปิด** — เปิดใน {_hh:02d}:{_mm2:02d}:{_ss:02d}")

    # yfinance delay warning
    st.caption("⚠️ yfinance: delay ~15 นาที\nข้อมูล realtime ต้องมี broker API")

    auto_refresh = st.toggle("🔄 Auto Refresh", value=False, key="auto_refresh_toggle")

    if auto_refresh:
        refresh_interval = st.select_slider(
            "ความถี่ refresh",
            options=[15, 30, 60, 120, 300],
            value=60 if _is_open else 300,
            format_func=lambda x: {
                15: "15 วิ ⚡", 30: "30 วิ", 60: "1 นาที",
                120: "2 นาที", 300: "5 นาที"
            }[x],
        )
        if _is_open:
            st.success(f"✅ Refresh ทุก {refresh_interval} วิ (ตลาดเปิด)")
        else:
            st.info(f"💤 Refresh ทุก {refresh_interval} วิ (ตลาดปิด — ข้อมูลไม่เปลี่ยน)")
    else:
        refresh_interval = 60

    st.divider()
    # API Status
    _st_id  = os.getenv("SETTRADE_APP_ID", "")
    _st_sb  = os.getenv("SETTRADE_SANDBOX", "true").lower() == "true"
    if _st_id:
        mode = "🧪 Sandbox" if _st_sb else "Production"
        st.success(f"⚡ SETTRADE Realtime\n{mode} mode")
    elif os.getenv("FINNHUB_API_KEY"):
        st.info("🟢 Finnhub mode")
    else:
        st.warning("🟡 yfinance mode\n(delay ~15 นาที)")

    st.caption("⚠️ ไม่ใช่คำแนะนำการลงทุน")


# ─── FETCH DATA ───────────────────────────────────────────────────────
with st.spinner(f"กำลังโหลดข้อมูล {symbol} ({tf_label})..."):
    quote = fetch_realtime(symbol)
    _fetch_interval = "60m" if timeframe == "4H" else tf_interval
    df    = fetch_historical(symbol, tf_period, _fetch_interval)
    # 4H = resample from 1H
    if timeframe == "4H" and not df.empty:
        df = resample_4h(df)
    divs  = fetch_dividends(symbol)
    info  = fetch_info(symbol)

if df is None or df.empty:
    if is_intraday:
        st.error(f"❌ ไม่พบข้อมูล intraday สำหรับ **{symbol}** ({timeframe})\n\n"
                 f"yfinance มีข้อจำกัด: 5M/15M/30M ดึงได้แค่ 5–10 วันล่าสุด, 1H ดึงได้ 30 วัน\n"
                 f"ลองเปลี่ยนเป็น 1M หรือ 3M แทน")
    else:
        st.error(f"❌ ไม่พบข้อมูลหุ้น **{symbol}** กรุณาลองหุ้นอื่น หรือตรวจสอบ internet connection")
    st.stop()

# Intraday banner
if is_intraday:
    st.info(f"⚡ **Intraday Mode: {timeframe}** ({tf_label}) — "
            f"{'delay ~15 นาที (yfinance)' if not os.getenv('SETTRADE_APP_ID') else 'SETTRADE realtime ✅'} · "
            f"แสดง {len(df)} candles")

# Add technical indicators
try:
    df = add_all_indicators(df)
except Exception as e:
    st.warning(f"⚠️ เพิ่ม indicators บางตัวไม่สำเร็จ: {e}")

if df.empty:
    st.error("ข้อมูลไม่เพียงพอสำหรับการวิเคราะห์ กรุณาเลือก timeframe ที่ยาวขึ้น")
    st.stop()

# Calculate signals and targets
try:
    score, signals, regime = calculate_signal_score(df)
except Exception as e:
    score, signals, regime = 50, [], "SIDEWAYS"

current_price = quote.get('price', float(df['Close'].iloc[-1]))
if current_price == 0:
    current_price = float(df['Close'].iloc[-1])

try:
    targets = calculate_price_targets(df, current_price)
except Exception as e:
    targets = {
        "buy_zone": {"low": current_price * 0.95, "high": current_price * 0.98},
        "stop_loss": current_price * 0.93,
        "targets": [current_price * 1.05, current_price * 1.10, current_price * 1.15],
        "trailing_stop": current_price * 0.95,
        "risk_amount_pct": 5.0,
        "risk_reward": 0.5,
        "fibonacci": {},
        "support_levels": [],
        "resistance_levels": [],
    }


# ─── HEADER ──────────────────────────────────────────────────────────
col_title, col_regime = st.columns([3, 1])

with col_title:
    change      = quote.get('change', 0) or 0
    pct_change  = quote.get('pct_change', 0) or 0
    price_color = "green" if change >= 0 else "red"
    change_icon = "▲" if change >= 0 else "▼"
    market_status = "🟢 เปิด" if is_market_open() else "🔴 ปิด"

    # ── Live ticker bar ───────────────────────────────────────────────
    ticker_symbols = ["PTT","ADVANC","KBANK","AOT","CPALL","SCB","GULF","BDMS","DELTA","MTC"]
    @st.cache_data(ttl=60)
    def _ticker_quotes(syms: tuple) -> list:
        out = []
        for s in syms:
            try:
                q = get_realtime_quote(s)
                p = q.get('price', 0)
                c = q.get('pct_change', 0)
                if p > 0:
                    out.append((s, p, c))
            except Exception:
                pass
        return out

    ticker_data = _ticker_quotes(tuple(ticker_symbols))
    if ticker_data:
        parts = []
        for s, p, c in ticker_data:
            clr  = "#00ff88" if c >= 0 else "#ff4444"
            icon = "▲" if c >= 0 else "▼"
            parts.append(
                f"<span style='margin:0 18px; white-space:nowrap'>"
                f"<b style='color:#ffd700'>{s}</b> "
                f"<span style='color:{clr}'>{icon} {p:.2f} ({c:+.2f}%)</span>"
                f"</span>"
            )
        ticker_html = "".join(parts * 3)  # repeat 3x for scroll effect
        st.markdown(f"""
        <div style='background:#0a0a14; border:1px solid #222; border-radius:6px;
             overflow:hidden; padding:6px 0; margin-bottom:10px'>
        <div style='display:flex; animation:scroll 40s linear infinite; width:max-content'>
            {ticker_html}
        </div>
        </div>
        <style>
        @keyframes scroll {{
            0%   {{ transform: translateX(0); }}
            100% {{ transform: translateX(-33.33%); }}
        }}
        </style>
        """, unsafe_allow_html=True)

    # ── Last updated indicator ────────────────────────────────────────
    now_str  = datetime.now().strftime('%H:%M:%S')
    _st_configured = bool(os.getenv("SETTRADE_APP_ID", ""))
    _st_sandbox    = os.getenv("SETTRADE_SANDBOX", "true").lower() == "true"
    if _st_configured:
        delay_note = "⚡ SETTRADE Realtime" + (" (Sandbox)" if _st_sandbox else "")
    elif os.getenv("FINNHUB_API_KEY"):
        delay_note = "Finnhub (~1 นาที)"
    else:
        delay_note = "⚠️ yfinance delay ~15 นาที"
    ttl_note = f"Cache: {_quote_ttl}s (ตลาด{'เปิด' if _mkt_open else 'ปิด'})"
    st.markdown(f"""
    <div style='display:flex; gap:12px; align-items:center; margin-bottom:8px; font-size:0.82rem; color:#666'>
        <span>🕐 {now_str}</span>
        <span>·</span>
        <span>ตลาด: {market_status}</span>
        <span>·</span>
        <span>⚠️ {delay_note}</span>
        <span>·</span>
        <span>{ttl_note}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <h1 style='margin-bottom:0'>{symbol}
    <span style='background:#1a1a2e; color:#ffd700; font-size:0.75rem;
         border:1px solid #ffd700; border-radius:4px; padding:2px 8px; margin-left:8px;
         vertical-align:middle'>{timeframe}</span>
    <span style='color:{price_color}; font-size:1.1rem'>
    &nbsp;{change_icon} {current_price:.2f} THB &nbsp;
    ({change:+.2f} / {pct_change:+.2f}%)
    </span></h1>
    """, unsafe_allow_html=True)
    st.caption(f"{info.get('name', symbol)} · {tf_label}")

with col_regime:
    regime_config = {
        "BULL_TREND": ("🐂 Uptrend",    "green"),
        "BEAR_TREND": ("🐻 Downtrend",  "red"),
        "SIDEWAYS":   ("↔️ Sideways",   "orange"),
        "TRANSITION": ("⏳ Transition", "gray"),
    }
    r_label, r_color = regime_config.get(regime, ("Unknown", "gray"))
    st.markdown(f"""
    <div style='text-align:center; padding:16px;
        border:2px solid {r_color}; border-radius:10px; color:{r_color}'>
        <h3 style='margin:0'>{r_label}</h3>
        <small>Market Regime</small>
    </div>
    """, unsafe_allow_html=True)


# ─── METRICS ROW ─────────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

def safe_fmt(val, fmt=".2f", fallback="N/A"):
    try:
        return format(float(val), fmt) if val and float(val) != 0 else fallback
    except:
        return fallback

m1.metric("Open",     safe_fmt(quote.get('open', 0)))
m2.metric("High",     safe_fmt(quote.get('high', 0)))
m3.metric("Low",      safe_fmt(quote.get('low', 0)))
vol = quote.get('volume', 0) or 0
m4.metric("Volume",   f"{vol/1e6:.2f}M" if vol > 0 else "N/A")
m5.metric("P/E",      safe_fmt(info.get('pe_ratio', 0)))
m6.metric("P/BV",     safe_fmt(info.get('pbv', 0)))
m7.metric("Div Yield",f"{safe_fmt(info.get('div_yield', 0))}%")


# ─── TRADE RECOMMENDATION PANEL ───────────────────────────────────────
st.divider()
rec = generate_recommendation(
    df=df, score=score, signals=signals,
    regime=regime, current_price=current_price,
    timeframe=timeframe,
)

_conf_badge = {
    "HIGH":   "<span style='background:#1a4a1a; color:#00ff88; padding:2px 8px; border-radius:4px; font-size:0.8rem'>HIGH</span>",
    "MEDIUM": "<span style='background:#2a2a0a; color:#ffd700; padding:2px 8px; border-radius:4px; font-size:0.8rem'>MEDIUM</span>",
    "LOW":    "<span style='background:#2a1a0a; color:#ff8800; padding:2px 8px; border-radius:4px; font-size:0.8rem'>LOW</span>",
}.get(rec['confidence'], "")

rc1, rc2 = st.columns([2, 3])

with rc1:
    # Main verdict box
    st.markdown(f"""
    <div style='background:linear-gradient(135deg, #0a0a14 0%, #1a1a2e 100%);
         border:2px solid {rec["color"]}; border-radius:12px;
         padding:20px; text-align:center'>
        <div style='font-size:3rem; margin-bottom:4px'>{rec["emoji"]}</div>
        <div style='font-size:1.5rem; font-weight:bold; color:{rec["color"]}'>
            {rec["title_th"]}
        </div>
        <div style='color:#ccc; font-size:0.9rem; margin:8px 0'>
            {rec["summary"]}
        </div>
        <div style='margin-top:10px'>
            ความน่าเชื่อถือ: {_conf_badge}
        </div>
        <div style='margin-top:10px; font-size:0.82rem; color:#888'>
            Signal Score: <b style='color:#ffd700'>{rec["score"]}</b>/100 ·
            RSI: <b style='color:{"#ff4444" if rec["rsi"]>70 else "#00ff88" if rec["rsi"]<30 else "white"}'>{rec["rsi"]}</b> ·
            TF: <b>{rec["timeframe"]}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Entry / SL / TP row
    if rec['entry_zone'] or rec['stop_loss'] or rec['targets']:
        st.markdown("")
        ep1, ep2, ep3 = st.columns(3)
        if rec['entry_zone']:
            ep1.markdown(f"""
            <div style='background:#0d2d0d; border:1px solid #00ff88;
                 border-radius:8px; padding:10px; text-align:center'>
            <div style='color:#888; font-size:0.75rem'>🎯 จุดเข้า</div>
            <div style='color:#00ff88; font-weight:bold; font-size:0.9rem'>
                {rec['entry_zone'][0]:.2f}–{rec['entry_zone'][1]:.2f}
            </div></div>
            """, unsafe_allow_html=True)
        if rec['stop_loss']:
            ep2.markdown(f"""
            <div style='background:#2d0d0d; border:1px solid #ff4444;
                 border-radius:8px; padding:10px; text-align:center'>
            <div style='color:#888; font-size:0.75rem'>🛑 Stop Loss</div>
            <div style='color:#ff4444; font-weight:bold; font-size:0.9rem'>
                {rec['stop_loss']:.2f}
            </div></div>
            """, unsafe_allow_html=True)
        if rec['targets']:
            ep3.markdown(f"""
            <div style='background:#1a1a0d; border:1px solid #ffd700;
                 border-radius:8px; padding:10px; text-align:center'>
            <div style='color:#888; font-size:0.75rem'>🏆 Target</div>
            <div style='color:#ffd700; font-weight:bold; font-size:0.9rem'>
                {rec['targets'][0]:.2f}
                {'/ '+str(rec['targets'][1]) if len(rec['targets'])>1 else ''}
            </div></div>
            """, unsafe_allow_html=True)

with rc2:
    # Reasons + Cautions
    if rec['reasons']:
        st.markdown("**✅ เหตุผลสนับสนุน**")
        for r in rec['reasons']:
            st.markdown(f"""
            <div style='background:#0d1a0d; border-left:3px solid #00ff88;
                 padding:6px 10px; margin:4px 0; border-radius:0 6px 6px 0;
                 font-size:0.85rem; color:#ccc'>{r}</div>
            """, unsafe_allow_html=True)

    if rec['cautions']:
        st.markdown("**⚠️ ปัจจัยเสี่ยง / ข้อควรระวัง**")
        for c in rec['cautions']:
            st.markdown(f"""
            <div style='background:#1a1500; border-left:3px solid #ffaa00;
                 padding:6px 10px; margin:4px 0; border-radius:0 6px 6px 0;
                 font-size:0.85rem; color:#ccc'>{c}</div>
            """, unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div style='background:#1a0505; border:1px solid #440000; border-radius:6px;
     padding:8px 12px; margin-top:4px; font-size:0.78rem; color:#aa6666'>
⚠️ <b>คำเตือน:</b> ผลการวิเคราะห์นี้เป็นเพียงเครื่องมือช่วยประกอบการพิจารณาจากข้อมูลทางเทคนิคเท่านั้น
ไม่ใช่คำแนะนำการลงทุน การลงทุนมีความเสี่ยง ผู้ลงทุนควรศึกษาข้อมูลและตัดสินใจด้วยตนเอง
</div>
""", unsafe_allow_html=True)


# ─── MAIN TABS ────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Chart & Signals",
    "🎯 จุดซื้อ-ขาย",
    "💰 ปันผล",
    "📋 ข้อมูลบริษัท",
    "⚙️ Backtest",
    "🔭 Fibo Scanner",
    "🕯️ แท่งเทียน & Bell Curve",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — CHART & SIGNALS
# ══════════════════════════════════════════════════════════════════════
with tab1:
    try:
        fig_candle = plot_candlestick(
            df, symbol,
            show_ema=show_ema,
            show_bb=show_bb,
            show_ichimoku=show_ichimoku,
            show_vwap=is_intraday,
            targets=targets,
            signals_list=signals
        )
        st.plotly_chart(fig_candle, use_container_width=True)
    except Exception as e:
        st.error(f"ไม่สามารถแสดงกราฟได้: {e}")

    # Sub-charts: MACD + RSI
    sub1, sub2 = st.columns(2)
    with sub1:
        try:
            st.plotly_chart(plot_macd(df), use_container_width=True)
        except Exception as e:
            st.warning(f"MACD error: {e}")
    with sub2:
        try:
            st.plotly_chart(plot_rsi(df), use_container_width=True)
        except Exception as e:
            st.warning(f"RSI error: {e}")

    # Fibonacci Retracement Chart (toggle)
    if show_fibonacci:
        st.divider()
        st.subheader("📐 Fibonacci Retracement")
        fib_col1, fib_col2 = st.columns([2, 1])
        with fib_col1:
            try:
                fig_fib = plot_fibonacci(df, symbol, current_price)
                st.plotly_chart(fig_fib, use_container_width=True)
            except Exception as e:
                st.error(f"Fibonacci chart error: {e}")
        with fib_col2:
            st.markdown("""
            <div style='background:#1a1a2e; border-radius:10px; padding:15px; border:1px solid #2d3250'>
            <h4 style='color:#ffd700'>📐 Fibonacci Retracement</h4>
            <p style='color:#aaa; font-size:0.85rem'>
            Fibonacci คือ indicator ที่ใช้หา<br>
            <b style='color:#00ff88'>แนวรับ</b> และ <b style='color:#ff4444'>แนวต้าน</b><br>
            จากอัตราส่วน Golden Ratio
            </p>
            <hr style='border-color:#333'>
            <table style='width:100%; font-size:0.82rem'>
            <tr><td style='color:#00bfff'>23.6%</td><td style='color:#aaa'>จุดพักตัวแรก</td></tr>
            <tr><td style='color:#00ff88'>38.2%</td><td style='color:#aaa'>จุดพักตัวที่ดี</td></tr>
            <tr><td style='color:#ffd700'>50.0%</td><td style='color:#aaa'>จุดจิตวิทยา</td></tr>
            <tr><td style='color:#ff8800; font-weight:bold'>61.8% 🌟</td><td style='color:#aaa'><b>Golden Ratio</b></td></tr>
            <tr><td style='color:#ff4488'>78.6%</td><td style='color:#aaa'>แนวรับ/ต้านแข็ง</td></tr>
            <tr><td style='color:#cc88ff'>127.2%</td><td style='color:#aaa'>เป้าหมายขยาย</td></tr>
            <tr><td style='color:#aa44ff; font-weight:bold'>161.8% 🌟</td><td style='color:#aaa'><b>Golden Extension</b></td></tr>
            </table>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Compact table
            try:
                fib_tbl = plot_fibonacci_table(df, current_price)
                # Highlight golden ratio row
                st.dataframe(
                    fib_tbl[['Level', 'ราคา (THB)', 'ห่างจากราคา', 'สถานะ']],
                    use_container_width=True, hide_index=True
                )
            except Exception as e:
                st.warning(f"Fibonacci table error: {e}")

    # Signal Cards
    st.subheader(f"📡 สัญญาณทั้งหมด ({len(signals)} สัญญาณ)")
    if signals:
        sig_cols = st.columns(3)
        for i, sig in enumerate(signals):
            css_class = (
                "signal-buy"     if sig['type'] == "BUY"  else
                "signal-sell"    if sig['type'] == "SELL" else
                "signal-neutral"
            )
            icon = "✅" if sig['type'] == "BUY" else "❌" if sig['type'] == "SELL" else "⚠️"
            strength = sig.get('strength', '')
            with sig_cols[i % 3]:
                st.markdown(f"""
                <div class="{css_class}">
                {icon} <b>{sig['type']}</b>
                {f'<span style="opacity:0.7"> — {strength}</span>' if strength else ''}<br>
                <small>{sig.get('reason', '')}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("ไม่มีสัญญาณที่ชัดเจนในขณะนี้")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — จุดซื้อ-ขาย
# ══════════════════════════════════════════════════════════════════════
with tab2:
    score_color = (
        "#00ff88" if score >= 60 else
        "#ff4444" if score <= 40 else
        "#ffd700"
    )
    score_label = (
        "BUY 🟢"     if score >= 65 else
        "SELL 🔴"    if score <= 35 else
        "NEUTRAL ⚪"
    )

    col_score, col_targets = st.columns([1, 2])

    with col_score:
        st.markdown(f"""
        <div style='text-align:center; padding:25px;
             border:2px solid {score_color}; border-radius:15px; margin-bottom:10px'>
            <p style='color:#aaa; margin:0'>Signal Score</p>
            <h1 style='color:{score_color}; font-size:3.5rem; margin:0'>{score}</h1>
            <p style='color:#aaa; margin:0'>/ 100</p>
            <h2 style='color:{score_color}; margin:8px 0 0 0'>{score_label}</h2>
        </div>
        """, unsafe_allow_html=True)

        st.progress(score / 100)

        st.info(f"""
        📊 **Market Regime:** {regime}  
        🕯 **ราคาปัจจุบัน:** {current_price:.2f} THB
        """)

    with col_targets:
        buy_low  = targets['buy_zone']['low']
        buy_high = targets['buy_zone']['high']
        sl       = targets['stop_loss']
        tp1, tp2, tp3 = targets['targets']
        risk_pct = targets['risk_amount_pct']
        rr       = targets['risk_reward']

        def pct_from(price):
            return ((price - current_price) / current_price * 100) if current_price > 0 else 0

        st.markdown(f"""
        <div class='target-box' style='border-left:4px solid #00ff88'>
        💚 <b>โซนซื้อ (Buy Zone)</b><br>
        <h3 style='margin:4px 0'>{buy_low:.2f} — {buy_high:.2f} THB</h3>
        </div>

        <div class='target-box' style='border-left:4px solid #ff4444'>
        🛑 <b>Stop Loss</b><br>
        <h3 style='margin:4px 0'>{sl:.2f} THB
        <small style='color:#ff4444'>({pct_from(sl):+.1f}%)</small></h3>
        </div>

        <div class='target-box' style='border-left:4px solid #ffd700'>
        🎯 <b>Target 1</b> (R:R 1:2)<br>
        <h3 style='margin:4px 0'>{tp1:.2f} THB
        <small style='color:#00ff88'>({pct_from(tp1):+.1f}%)</small></h3>
        </div>

        <div class='target-box' style='border-left:4px solid #ffd700'>
        🎯 <b>Target 2</b> (R:R 1:3)<br>
        <h3 style='margin:4px 0'>{tp2:.2f} THB
        <small style='color:#00ff88'>({pct_from(tp2):+.1f}%)</small></h3>
        </div>

        <div class='target-box' style='border-left:4px solid #ff8800'>
        🎯 <b>Target 3</b> (Resistance)<br>
        <h3 style='margin:4px 0'>{tp3:.2f} THB
        <small style='color:#00ff88'>({pct_from(tp3):+.1f}%)</small></h3>
        </div>
        """, unsafe_allow_html=True)

        rr_display = (1 / rr) if rr > 0 else 0
        st.warning(f"⚠️ ความเสี่ยงต่อเทรด: **{risk_pct:.1f}%** | Risk/Reward: **1:{rr_display:.1f}**")

    # Fibonacci Analysis — Full Panel
    st.divider()
    st.subheader("📐 Fibonacci Retracement Analysis")

    fib_main, fib_side = st.columns([3, 1])

    with fib_main:
        try:
            fig_fib2 = plot_fibonacci(df, symbol, current_price)
            st.plotly_chart(fig_fib2, use_container_width=True)
        except Exception as e:
            st.error(f"Fibonacci chart error: {e}")

    with fib_side:
        # Explanation card
        st.markdown("""
        <div style='background:#1a1a2e; border-radius:10px; padding:14px;
             border:1px solid #ffd700; margin-bottom:10px'>
        <h4 style='color:#ffd700; margin:0 0 8px 0'>🌟 วิธีใช้ Fibonacci</h4>
        <p style='color:#ccc; font-size:0.83rem; margin:0'>
        <b style='color:#00ff88'>จุดเข้าซื้อ:</b><br>
        ราคาพักตัวที่ 38.2%, 50%, 61.8%<br><br>
        <b style='color:#ffd700'>จุดทำกำไร:</b><br>
        TP ที่ 100%, 127.2%, 161.8%<br><br>
        <b style='color:#ff4444'>Stop Loss:</b><br>
        ต่ำกว่า 78.6% ในทิศทางขาขึ้น<br><br>
        <b style='color:#ff8800'>Golden Ratio 61.8%</b> คือระดับสำคัญที่สุด<br>
        ราคามักพักตัวและกลับตัวที่จุดนี้
        </p>
        </div>
        """, unsafe_allow_html=True)

    # Full Fibonacci Table
    st.subheader("📋 ตาราง Fibonacci Levels ทั้งหมด")
    try:
        fib_tbl = plot_fibonacci_table(df, current_price)
        # Style the dataframe display
        st.dataframe(
            fib_tbl,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Level":        st.column_config.TextColumn("Level", width="small"),
                "ราคา (THB)":  st.column_config.NumberColumn("ราคา (THB)", format="%.2f"),
                "ห่างจากราคา": st.column_config.TextColumn("ห่างจากราคา", width="medium"),
                "ความสำคัญ":   st.column_config.TextColumn("ความสำคัญ", width="large"),
                "สถานะ":       st.column_config.TextColumn("สถานะ", width="medium"),
            }
        )

        # Highlight the zone current price is in
        swing_high = float(df['High'].max())
        swing_low  = float(df['Low'].min())
        fib_range  = swing_high - swing_low
        mid_idx = len(df) // 2
        is_up = df['Close'].iloc[mid_idx:].mean() >= df['Close'].iloc[:mid_idx].mean()
        base = swing_low if is_up else swing_high
        dirn = 1 if is_up else -1

        zone_name = "ยังไม่ชัดเจน"
        zone_color = "#666"
        fib_check = [
            (0.000, 0.236, "0.0% – 23.6%",  "#888"),
            (0.236, 0.382, "23.6% – 38.2%", "#00bfff"),
            (0.382, 0.500, "38.2% – 50.0%", "#00ff88"),
            (0.500, 0.618, "50.0% – 61.8%", "#ffd700"),
            (0.618, 0.786, "61.8% – 78.6% 🌟", "#ff8800"),
            (0.786, 1.000, "78.6% – 100%",  "#ff4488"),
            (1.000, 1.272, "100% – 127.2%", "#cc88ff"),
            (1.272, 1.618, "127.2% – 161.8% 🌟", "#aa44ff"),
        ]
        for r0, r1, lbl, col in fib_check:
            p0 = base + dirn * fib_range * r0
            p1 = base + dirn * fib_range * r1
            if min(p0, p1) <= current_price <= max(p0, p1):
                zone_name = lbl
                zone_color = col
                break

        st.markdown(f"""
        <div style='background:#1a1a2e; border-left:4px solid {zone_color};
             border-radius:8px; padding:12px; margin-top:8px'>
        <b style='color:{zone_color}'>📍 ราคาปัจจุบัน {current_price:.2f} THB</b><br>
        <span style='color:#ccc'>อยู่ใน Fibonacci Zone: </span>
        <b style='color:{zone_color}'>{zone_name}</b>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Fibonacci table error: {e}")

    # Support / Resistance
    col_sup, col_res = st.columns(2)
    with col_sup:
        st.subheader("🔵 แนวรับ (Support)")
        if targets['support_levels']:
            sup_data = [
                {"ระดับ (THB)": f"{s:.2f}",
                 "ห่างจากราคา": f"{((current_price - s) / current_price * 100):.1f}%"}
                for s in targets['support_levels'][:5]
            ]
            st.dataframe(pd.DataFrame(sup_data), hide_index=True, use_container_width=True)
        else:
            st.info("ไม่พบแนวรับที่ชัดเจน")

    with col_res:
        st.subheader("🟠 แนวต้าน (Resistance)")
        if targets['resistance_levels']:
            res_data = [
                {"ระดับ (THB)": f"{r:.2f}",
                 "ห่างจากราคา": f"{((r - current_price) / current_price * 100):.1f}%"}
                for r in targets['resistance_levels'][:5]
            ]
            st.dataframe(pd.DataFrame(res_data), hide_index=True, use_container_width=True)
        else:
            st.info("ไม่พบแนวต้านที่ชัดเจน")

    st.error("""
    ⚠️ **DISCLAIMER:** การวิเคราะห์ทางเทคนิคนี้เป็นเพียงเครื่องมือช่วยประกอบการตัดสินใจเท่านั้น 
    มิใช่คำแนะนำการลงทุน ผู้ลงทุนควรศึกษาข้อมูลเพิ่มเติมและรับผิดชอบการตัดสินใจด้วยตนเอง  
    **การลงทุนมีความเสี่ยง ผู้ลงทุนอาจสูญเสียเงินลงทุนทั้งหมดหรือบางส่วนได้**
    """)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — ปันผล
# ══════════════════════════════════════════════════════════════════════
with tab3:
    if divs is not None and not divs.empty:
        col_chart, col_stats = st.columns([2, 1])

        with col_chart:
            try:
                st.plotly_chart(plot_dividend_chart(divs), use_container_width=True)
            except Exception as e:
                st.warning(f"ไม่สามารถแสดงกราฟปันผล: {e}")

        with col_stats:
            total_5y = divs['amount'].sum()
            try:
                annual_by_year = divs.groupby('year')['amount'].sum()
                avg_annual = annual_by_year.mean()
            except:
                avg_annual = total_5y / 5

            div_cagr = calculate_dividend_cagr(divs)
            count = len(divs)

            st.metric("💰 ปันผลรวม 5 ปี", f"{total_5y:.4f} THB")
            st.metric("📅 เฉลี่ยต่อปี",    f"{avg_annual:.4f} THB")
            st.metric("📈 CAGR ปันผล",     f"{div_cagr:.1f}%")
            st.metric("🔁 จำนวนครั้งที่จ่าย", f"{count} ครั้ง")

        st.subheader("📋 ประวัติปันผลทั้งหมด")
        display_divs = divs.copy()
        display_divs['ex_date'] = display_divs['ex_date'].dt.strftime('%Y-%m-%d')
        st.dataframe(
            display_divs.sort_values('ex_date', ascending=False),
            use_container_width=True, hide_index=True
        )
    else:
        st.info(f"ℹ️ ไม่พบข้อมูลปันผลของ **{symbol}** ในช่วง 5 ปีที่ผ่านมา หรือบริษัทนี้ไม่มีนโยบายจ่ายปันผล")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — ข้อมูลบริษัท
# ══════════════════════════════════════════════════════════════════════
with tab4:
    col_info1, col_info2 = st.columns(2)

    with col_info1:
        st.subheader("📌 ข้อมูลพื้นฐาน")
        mktcap = info.get('market_cap', 0) or 0
        shares = info.get('shares_outstanding', 0) or 0
        st.json({
            "ชื่อบริษัท":   info.get('name', symbol),
            "Sector":      info.get('sector', 'N/A'),
            "Industry":    info.get('industry', 'N/A'),
            "Market Cap":  f"{mktcap/1e9:.2f}B THB" if mktcap else "N/A",
            "Shares Out":  f"{shares/1e6:.2f}M หุ้น" if shares else "N/A",
            "52W High":    str(info.get('52w_high', 'N/A')),
            "52W Low":     str(info.get('52w_low', 'N/A')),
            "Beta":        str(info.get('beta', 'N/A')),
        })

        # 52-week range progress bar
        try:
            high_52w = float(info.get('52w_high', 0)) or 0
            low_52w  = float(info.get('52w_low', 0)) or 0
            if high_52w > low_52w > 0:
                pos = (current_price - low_52w) / (high_52w - low_52w)
                st.write("**📊 ตำแหน่งใน 52-Week Range**")
                st.caption(f"Low: {low_52w:.2f}  ←  {current_price:.2f}  →  High: {high_52w:.2f}")
                st.progress(float(np.clip(pos, 0, 1)))
        except:
            pass

    with col_info2:
        st.subheader("📊 อัตราส่วนทางการเงิน")
        ratios = [
            ("P/E Ratio",     info.get('pe_ratio', 0)),
            ("P/BV Ratio",    info.get('pbv', 0)),
            ("EPS (TTM)",     info.get('eps', 0)),
            ("ROE",           f"{info.get('roe', 0):.1f}%"),
            ("ROA",           f"{info.get('roa', 0):.1f}%"),
            ("Debt/Equity",   info.get('debt_equity', 0)),
            ("Current Ratio", info.get('current_ratio', 0)),
            ("Div Yield",     f"{info.get('div_yield', 0):.2f}%"),
            ("Beta",          info.get('beta', 0)),
        ]
        for label, val in ratios:
            if val and val != 0:
                st.metric(label, val)
            else:
                st.metric(label, "N/A")


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — BACKTEST
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("⚙️ Backtest Strategy")
    st.caption("ทดสอบ strategy บนข้อมูลย้อนหลัง")

    bc1, bc2, bc3 = st.columns(3)
    strategy = bc1.selectbox("📋 Strategy", [
        "EMA Crossover (9/21)",
        "RSI Oversold/Overbought",
        "MACD Crossover",
        "Bollinger Band Bounce",
        "Combined Signal Score > 65"
    ])
    bt_capital = bc2.number_input("💵 เงินทุน (THB)", value=100000, step=10000, min_value=10000)
    bt_sl_pct  = bc3.slider("🛑 Stop Loss %", 3.0, 20.0, 7.0, 0.5)

    if st.button("🚀 Run Backtest", type="primary"):
        if len(df) < 50:
            st.warning("ข้อมูลน้อยเกินไปสำหรับ backtest กรุณาเลือก timeframe ที่ยาวขึ้น (1Y หรือมากกว่า)")
        else:
            with st.spinner("กำลัง backtest กรุณารอสักครู่..."):
                try:
                    bt_results = run_backtest(df, strategy, float(bt_capital), bt_sl_pct / 100)

                    r1, r2, r3, r4 = st.columns(4)
                    tr = bt_results['total_return']
                    r1.metric(
                        "📈 Total Return",
                        f"{tr:.1f}%",
                        delta=f"{tr:.1f}%"
                    )
                    r2.metric("✅ Win Rate",     f"{bt_results['win_rate']:.1f}%")
                    r3.metric("📉 Max Drawdown", f"{bt_results['max_drawdown']:.1f}%")
                    r4.metric("🔄 Total Trades", bt_results['total_trades'])

                    final_equity = bt_capital * (1 + tr / 100)
                    st.info(f"💰 เงินเริ่มต้น: {bt_capital:,.0f} THB → เงินสุดท้าย: {final_equity:,.0f} THB")

                    st.plotly_chart(bt_results['equity_curve'], use_container_width=True)

                    if not bt_results['trade_log'].empty:
                        st.subheader("📋 Trade Log")
                        st.dataframe(bt_results['trade_log'], use_container_width=True, hide_index=True)
                    else:
                        st.info("ไม่มี trade ที่เกิดขึ้นใน period นี้ ลอง strategy อื่นหรือ timeframe ที่ยาวขึ้น")

                except Exception as e:
                    st.error(f"Backtest error: {e}")


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — FIBONACCI SCANNER
# ══════════════════════════════════════════════════════════════════════
with tab6:
    st.subheader("🔭 Fibonacci Scanner — หาหุ้นน่าลงทุน")
    st.caption("สแกนหุ้น SET ที่ราคาอยู่ใน Golden Zone (38.2%–61.8%) พร้อม signal แข็งแกร่ง")

    # ── Scan Mode ─────────────────────────────────────────────────────
    mode_col1, mode_col2 = st.columns([2, 3])
    scan_mode = mode_col1.radio(
        "🔍 โหมดสแกน",
        ["Multi-Timeframe (แนะนำ)", "Single Timeframe", "⚡ Day Trade (Intraday)"],
        index=0, horizontal=False,
    )
    with mode_col2:
        if "Multi" in scan_mode:
            st.markdown("""
            <div style='background:#0d2d1a; border:1px solid #00ff88; border-radius:8px; padding:12px; margin-top:4px'>
            <b style='color:#00ff88'>🌟 Multi-Timeframe Confluence</b><br>
            <span style='color:#ccc; font-size:0.85rem'>
            สแกนทั้ง <b>3 เดือน + 6 เดือน + 1 ปี</b> พร้อมกัน<br>
            หุ้นที่อยู่ใน Golden Zone ทั้ง 3 TF = <b style='color:#ffd700'>Confluence สูงมาก</b><br>
            โอกาสกำไรสูง เพราะ Fib level "ซ้อนกัน" หลายชั้น
            </span>
            </div>
            """, unsafe_allow_html=True)
        elif "Day Trade" in scan_mode:
            st.markdown("""
            <div style='background:#1a0d2e; border:1px solid #aa44ff; border-radius:8px; padding:12px; margin-top:4px'>
            <b style='color:#cc88ff'>⚡ Day Trade / Intraday Fibonacci</b><br>
            <span style='color:#ccc; font-size:0.85rem'>
            ใช้ข้อมูล <b>Intraday (5m/15m/30m/1h)</b><br>
            หา Fib จาก <b>Swing High/Low ของ 2–10 วันล่าสุด</b><br>
            เหมาะกับการเทรด <b style='color:#ffd700'>ภายในวัน หรือ 1–3 วัน</b><br>
            <small>⚠️ ควรใช้ระหว่างตลาดเปิด (10:00–17:00) เพื่อข้อมูล real-time</small>
            </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#1a1a0d; border:1px solid #ffd700; border-radius:8px; padding:12px; margin-top:4px'>
            <b style='color:#ffd700'>📅 Single Timeframe</b><br>
            <span style='color:#ccc; font-size:0.85rem'>
            เลือก timeframe เดียว เหมาะกับ swing/position trade<br>
            <b>3 เดือน</b> = แนะนำสำหรับ swing 1–4 สัปดาห์
            </span>
            </div>
            """, unsafe_allow_html=True)

    # ── Scanner Settings ──────────────────────────────────────────────
    with st.expander("⚙️ ตั้งค่าการสแกน", expanded=True):
        sc1, sc2, sc3, sc4 = st.columns(4)

        if "Day Trade" in scan_mode:
            scan_interval = sc1.selectbox(
                "⏱ Interval",
                ["5m", "15m", "30m", "1h"],
                index=1,  # default 15m
                format_func=lambda x: {
                    "5m":  "5 นาที (Scalp) ⚡",
                    "15m": "15 นาที (Day trade) ✨",
                    "30m": "30 นาที (Intraday swing)",
                    "1h":  "1 ชั่วโมง (Short swing)",
                }[x],
                help="ยิ่งเล็ก = เร็ว แต่ noise มากขึ้น | แนะนำ 15 นาที"
            )
            scan_period = "daytrade"
        elif "Single" in scan_mode:
            scan_interval = "15m"
            scan_period = sc1.selectbox(
                "📅 ช่วงเวลา",
                ["1mo", "3mo", "6mo", "1y", "2y"],
                index=2,
                format_func=lambda x: {
                    "1mo": "1 เดือน",
                    "3mo": "3 เดือน (Swing) ✨",
                    "6mo": "6 เดือน (Position)",
                    "1y":  "1 ปี (Major level)",
                    "2y":  "2 ปี (Long-term)",
                }[x],
            )
        else:
            scan_interval = "15m"
            sc1.markdown("""
            <div style='padding:8px 0'>
            <small style='color:#aaa'>Timeframes</small><br>
            <b style='color:#00ff88'>3M + 6M + 1Y</b>
            </div>
            """, unsafe_allow_html=True)
            scan_period = "multi"

        min_fib = sc2.slider("🎯 Fib Score ขั้นต่ำ", 30, 80, 45, 5)
        min_rr  = sc3.slider("⚖️ Risk/Reward ขั้นต่ำ", 1.0, 4.0, 1.0, 0.5)
        max_w   = sc4.slider("⚡ Workers", 3, 15, 8, 1,
                              help="มากขึ้น = เร็วขึ้น แต่ใช้ internet มากขึ้น")

        # Preset universe
        sc5, sc6 = st.columns(2)
        use_custom = sc5.toggle("กำหนด watchlist เอง", value=False)

        if use_custom:
            custom_input = sc6.text_area(
                "ใส่ symbols คั่นด้วย comma หรือ newline",
                value="PTT, KBANK, AOT, CPALL, BDMS, ADVANC",
                height=80,
            )
            scan_symbols = [
                s.strip().upper()
                for s in custom_input.replace('\n', ',').split(',')
                if s.strip()
            ]
            st.caption(f"จะสแกน {len(scan_symbols)} หุ้น: {', '.join(scan_symbols[:10])}{'...' if len(scan_symbols)>10 else ''}")
        else:
            st.caption(f"จะสแกน **{len(SET_UNIVERSE)} หุ้น** จาก SET Universe (Large+Mid cap)")
            scan_symbols = None

    # ── Run Button ────────────────────────────────────────────────────
    run_col, est_col = st.columns([1, 3])
    run_scan = run_col.button("🚀 เริ่มสแกน", type="primary", use_container_width=True)
    n_stocks   = len(scan_symbols) if scan_symbols else len(SET_UNIVERSE)
    if scan_period == "daytrade":
        n_tasks = n_stocks
        mode_label = f"Intraday {scan_interval}"
    elif scan_period == "multi":
        n_tasks = n_stocks * 3
        mode_label = "Multi-TF (3×)"
    else:
        n_tasks = n_stocks
        mode_label = scan_period
    est_sec = max(15, n_tasks // max_w * 4)
    est_col.info(
        f"เวลาโดยประมาณ: **{est_sec}–{est_sec+20} วินาที** "
        f"({n_stocks} หุ้น × {mode_label}, {max_w} workers)",
        icon="ℹ️"
    )

    if run_scan:
        progress_bar  = st.progress(0.0, text="กำลังเริ่มต้น...")
        status_text   = st.empty()
        result_holder = st.empty()

        scan_state = {"done": 0, "found": 0, "raw": 0}

        def update_progress(done, total, current_sym):
            scan_state["done"] = done
            pct = done / total
            progress_bar.progress(
                pct,
                text=f"สแกน {done}/{total} · {current_sym} · พบ {scan_state['found']} หุ้น"
            )

        scan_df     = pd.DataFrame()
        scan_df_raw = pd.DataFrame()

        with st.spinner(""):
            try:
                if scan_period == "daytrade":
                    # ── Day Trade / Intraday scan ─────────────────────
                    scan_df = run_daytrade_scan(
                        symbols=scan_symbols,
                        interval=scan_interval,
                        min_fib_score=min_fib,
                        min_rr=min_rr,
                        max_workers=max_w,
                        progress_callback=update_progress,
                    )
                    scan_df_raw = scan_df
                elif scan_period == "multi":
                    # ── Multi-Timeframe scan ──────────────────────────
                    scan_df = run_multi_timeframe_scan(
                        symbols=scan_symbols,
                        periods=["3mo", "6mo", "1y"],
                        min_fib_score=min_fib,
                        min_rr=min_rr,
                        max_workers=max_w,
                        progress_callback=update_progress,
                    )
                    scan_df_raw = scan_df
                else:
                    # ── Single Timeframe ──────────────────────────────
                    scan_df_raw = run_fibonacci_scan(
                        symbols=scan_symbols,
                        period=scan_period,
                        min_fib_score=0,
                        min_rr=0,
                        max_workers=max_w,
                        progress_callback=update_progress,
                    )
                    if not scan_df_raw.empty and 'fib_score' in scan_df_raw.columns:
                        scan_df = scan_df_raw[
                            (scan_df_raw['fib_score']   >= min_fib) &
                            (scan_df_raw['risk_reward'] >= min_rr)
                        ].reset_index(drop=True)
                    else:
                        scan_df = pd.DataFrame()

                scan_state["found"] = len(scan_df)
            except Exception as e:
                st.error(f"Scanner error: {e}")

        progress_bar.progress(1.0, text=f"✅ สแกนเสร็จ — พบ {len(scan_df)} หุ้นผ่านเกณฑ์")

        if scan_df.empty:
            raw_count = len(scan_df_raw) if 'scan_df_raw' in dir() and not scan_df_raw.empty else 0
            if raw_count > 0:
                st.warning(f"พบ **{raw_count} หุ้น** แต่ไม่ผ่านเกณฑ์ที่ตั้งไว้ — ลองลด Fib Score หรือ R:R")
                with st.expander(f"🔍 ดูผลดิบทั้ง {raw_count} หุ้น (ก่อน filter)"):
                    show_raw = scan_df_raw[['symbol','price','fib_score','grade',
                                           'zone','signal_score','rsi','risk_reward','is_uptrend']].copy()
                    show_raw.columns = ['Symbol','ราคา','Fib Score','เกรด',
                                        'Zone','Signal','RSI','R:R','Uptrend']
                    st.dataframe(show_raw, use_container_width=True, hide_index=True)
            else:
                st.warning("ไม่สามารถดึงข้อมูลหุ้นได้ กรุณาตรวจสอบ internet connection")
        else:
            # ── Summary metrics ───────────────────────────────────────
            st.divider()
            is_mtf = scan_period == "multi"
            score_col = 'mtf_score' if (is_mtf and 'mtf_score' in scan_df.columns) else 'fib_score'

            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("🎯 หุ้นผ่านเกณฑ์", len(scan_df))
            sm2.metric("⭐ เกรด A+/A",
                       len(scan_df[scan_df['grade'].isin(['A+','A'])]))
            if is_mtf and 'passed_tfs' in scan_df.columns:
                sm3.metric("🟢🟢🟢 ผ่านทั้ง 3 TF",
                           len(scan_df[scan_df['passed_tfs'] >= 3]))
            else:
                sm3.metric("📈 Uptrend",
                           len(scan_df[scan_df['is_uptrend'] == True]))
            sm4.metric("🌟 ใกล้ Golden 61.8%",
                       len(scan_df[scan_df['dist_golden'] <= 5.0]))
            avg_rr = scan_df['risk_reward'].mean()
            sm5.metric("⚖️ Avg R:R", f"1:{avg_rr:.1f}")

            # ── Grade color helper ────────────────────────────────────
            grade_colors = {
                "A+": "🟢", "A": "🟢", "B+": "🟡", "B": "🟡"
            }

            # ── Top picks highlight cards ─────────────────────────────
            st.subheader("🏆 Top Picks")
            is_mtf      = scan_period == "multi"
            is_daytrade = scan_period == "daytrade"
            top_n = min(6, len(scan_df))
            card_cols = st.columns(min(3, top_n))
            for i in range(top_n):
                row = scan_df.iloc[i]
                col = card_cols[i % 3]
                trend_icon = "📈" if row['is_uptrend'] else "📉"
                grade_icon = grade_colors.get(row['grade'], "⚪")
                rr_str = f"1:{row['risk_reward']:.1f}"
                score_display = row.get('mtf_score', row['fib_score']) if is_mtf else row['fib_score']
                score_label   = "MTF Score" if is_mtf else "Fib Score"
                confluence_row = f"<tr><td>🔗 Confluence</td><td><b style='color:#00ff88'>{row.get('confluence','—')}</b></td></tr>" if is_mtf and 'confluence' in row else ""
                # Day trade extras: VWAP, ATR, bar change
                if is_daytrade and 'vwap' in row:
                    vwap_color = "#00ff88" if row['vs_vwap_pct'] >= 0 else "#ff4444"
                    extra_rows = f"""
                  <tr><td>📊 VWAP</td><td><b>{row['vwap']:.2f}</b>
                    <span style='color:{vwap_color}'>({row['vs_vwap_pct']:+.1f}%)</span></td></tr>
                  <tr><td>📏 ATR ({row['interval']})</td><td><b>{row['atr']:.3f}</b></td></tr>
                  <tr><td>🕐 Bar Change</td>
                    <td><b style='color:{"#00ff88" if row["change_5d"]>=0 else "#ff4444"}'>{row['change_5d']:+.2f}%</b></td></tr>
                """
                    bar_label = f"<span style='background:#1a0d2e; color:#cc88ff; padding:2px 6px; border-radius:4px; font-size:0.75rem'>⚡ {row['interval']}</span>"
                else:
                    extra_rows = f"""
                  <tr><td>📈 5D Change</td>
                    <td><b style='color:{"#00ff88" if row["change_5d"]>=0 else "#ff4444"}'>{row['change_5d']:+.1f}%</b></td></tr>
                """
                    bar_label = ""
                col.markdown(f"""
                <div style='background:#1a1a2e; border:1px solid #ffd700;
                     border-radius:10px; padding:14px; margin-bottom:8px'>
                <div style='display:flex; justify-content:space-between; align-items:center'>
                  <h3 style='margin:0; color:#ffd700'>{row['symbol']} {bar_label}</h3>
                  <span style='font-size:1.3rem'>{grade_icon} {row['grade']}</span>
                </div>
                <div style='font-size:1.1rem; color:white; margin:6px 0'>
                  <b>{row['price']:.2f} THB</b>
                </div>
                <hr style='border-color:#333; margin:8px 0'>
                <table style='width:100%; font-size:0.82rem; color:#ccc'>
                  {confluence_row}
                  <tr>
                    <td>🎯 {score_label}</td>
                    <td><b style='color:#ffd700'>{score_display:.0f}</b>/100</td>
                  </tr>
                  <tr>
                    <td>📐 Zone</td>
                    <td><b style='color:#00bfff'>{row['zone']}</b></td>
                  </tr>
                  <tr>
                    <td>🌟 ห่างจาก 61.8%</td>
                    <td><b>{row['dist_golden']:.1f}%</b></td>
                  </tr>
                  <tr>
                    <td>📊 Signal</td>
                    <td><b style='color:#00ff88'>{row['signal_score']}/100</b></td>
                  </tr>
                  <tr>
                    <td>📉 RSI</td>
                    <td><b style='color:{"#ff4444" if row["rsi"]>70 else "#00ff88" if row["rsi"]<30 else "white"}'>{row['rsi']}</b></td>
                  </tr>
                  {extra_rows}
                  <tr>
                    <td>⚖️ R:R</td>
                    <td><b style='color:#00ff88'>{rr_str}</b></td>
                  </tr>
                  <tr>
                    <td>🛑 Stop Loss</td>
                    <td><b style='color:#ff4444'>{row['stop_loss']:.2f}</b></td>
                  </tr>
                  <tr>
                    <td>🎯 TP1 / TP2</td>
                    <td><b>{row['tp1']:.2f} / {row['tp2']:.2f}</b></td>
                  </tr>
                  <tr>
                    <td>{trend_icon} Trend</td>
                    <td><b>{"Uptrend" if row["is_uptrend"] else "Downtrend"}</b></td>
                  </tr>
                </table>
                </div>
                """, unsafe_allow_html=True)

                # ปุ่มดูกราฟหุ้นนี้
                if col.button(f"📊 ดูกราฟ {row['symbol']}",
                               key=f"scan_view_{row['symbol']}_{i}",
                               use_container_width=True):
                    st.session_state.symbol = row['symbol']
                    st.rerun()

            # ── Full results table ────────────────────────────────────
            st.divider()
            st.subheader(f"📋 ผลการสแกนทั้งหมด ({len(scan_df)} หุ้น)")

            is_mtf      = scan_period == "multi"
            is_daytrade = scan_period == "daytrade"

            if is_daytrade and 'vwap' in scan_df.columns:
                display_df = scan_df[[
                    'symbol','price','fib_score','grade','zone',
                    'signal_score','rsi','vol_ratio',
                    'vwap','vs_vwap_pct','atr',
                    'stop_loss','tp1','tp2','risk_pct','risk_reward','change_5d'
                ]].copy()
                display_df.columns = [
                    'Symbol','ราคา','Fib Score','เกรด','Zone',
                    'Signal','RSI','Vol Ratio',
                    'VWAP','vs VWAP%','ATR',
                    'Stop Loss','TP1','TP2','Risk%','R:R','Bar Change%'
                ]
                col_cfg = {
                    "Symbol":     st.column_config.TextColumn("Symbol", width="small"),
                    "ราคา":       st.column_config.NumberColumn("ราคา", format="%.2f"),
                    "Fib Score":  st.column_config.ProgressColumn("Fib Score", min_value=0, max_value=100, format="%d"),
                    "Signal":     st.column_config.ProgressColumn("Signal",    min_value=0, max_value=100, format="%d"),
                    "RSI":        st.column_config.NumberColumn("RSI",         format="%.1f"),
                    "Vol Ratio":  st.column_config.NumberColumn("Vol Ratio",   format="%.1fx"),
                    "VWAP":       st.column_config.NumberColumn("VWAP",        format="%.2f"),
                    "vs VWAP%":   st.column_config.NumberColumn("vs VWAP%",    format="%.2f%%"),
                    "ATR":        st.column_config.NumberColumn("ATR",         format="%.3f"),
                    "R:R":        st.column_config.NumberColumn("R:R",         format="1:%.1f"),
                    "Risk%":      st.column_config.NumberColumn("Risk%",       format="%.2f%%"),
                    "Bar Change%":st.column_config.NumberColumn("Bar%",        format="%.2f%%"),
                }
            elif is_mtf and 'mtf_score' in scan_df.columns:
                display_df = scan_df[[
                    'symbol','price','mtf_score','grade','confluence',
                    'zone_3mo','zone_6mo','zone_1y',
                    'score_3mo','score_6mo','score_1y',
                    'signal_score','rsi','stop_loss','tp1','tp2','risk_reward','change_5d'
                ]].copy()
                display_df.columns = [
                    'Symbol','ราคา','MTF Score','เกรด','Confluence',
                    'Zone 3M','Zone 6M','Zone 1Y',
                    'Fib 3M','Fib 6M','Fib 1Y',
                    'Signal','RSI','Stop Loss','TP1','TP2','R:R','5D%'
                ]
                col_cfg = {
                    "Symbol":     st.column_config.TextColumn("Symbol", width="small"),
                    "ราคา":       st.column_config.NumberColumn("ราคา", format="%.2f"),
                    "MTF Score":  st.column_config.ProgressColumn("MTF Score", min_value=0, max_value=100, format="%d"),
                    "Fib 3M":     st.column_config.ProgressColumn("Fib 3M",  min_value=0, max_value=100, format="%d"),
                    "Fib 6M":     st.column_config.ProgressColumn("Fib 6M",  min_value=0, max_value=100, format="%d"),
                    "Fib 1Y":     st.column_config.ProgressColumn("Fib 1Y",  min_value=0, max_value=100, format="%d"),
                    "Signal":     st.column_config.ProgressColumn("Signal",  min_value=0, max_value=100, format="%d"),
                    "RSI":        st.column_config.NumberColumn("RSI", format="%.1f"),
                    "R:R":        st.column_config.NumberColumn("R:R", format="1:%.1f"),
                    "5D%":        st.column_config.NumberColumn("5D%", format="%.2f%%"),
                }
            else:
                display_df = scan_df[[
                    'symbol','price','fib_score','grade','zone',
                    'dist_golden','signal_score','regime',
                    'rsi','vol_ratio','buy_signals',
                    'stop_loss','tp1','tp2','risk_pct','risk_reward','change_5d'
                ]].copy()
                display_df.columns = [
                    'Symbol','ราคา','Fib Score','เกรด','Fib Zone',
                    'ห่าง 61.8%','Signal','Regime',
                    'RSI','Vol Ratio','Buy Signals',
                    'Stop Loss','TP1','TP2','Risk %','R:R','5D Change%'
                ]
                col_cfg = {
                    "Symbol":      st.column_config.TextColumn("Symbol", width="small"),
                    "ราคา":        st.column_config.NumberColumn("ราคา (THB)", format="%.2f"),
                    "Fib Score":   st.column_config.ProgressColumn("Fib Score", min_value=0, max_value=100, format="%d"),
                    "Signal":      st.column_config.ProgressColumn("Signal",    min_value=0, max_value=100, format="%d"),
                    "RSI":         st.column_config.NumberColumn("RSI",         format="%.1f"),
                    "Vol Ratio":   st.column_config.NumberColumn("Vol Ratio",   format="%.2fx"),
                    "R:R":         st.column_config.NumberColumn("R:R",         format="1:%.1f"),
                    "5D Change%":  st.column_config.NumberColumn("5D %",        format="%.2f%%"),
                    "Risk %":      st.column_config.NumberColumn("Risk %",       format="%.1f%%"),
                }

            st.dataframe(display_df, use_container_width=True, hide_index=True, column_config=col_cfg)

            # ── Export ────────────────────────────────────────────────
            csv = display_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "⬇️ Download ผลการสแกน (.csv)",
                data=csv,
                file_name=f"fibo_scan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )

            # ── Scatter plot: Fib Score vs Signal Score ───────────────
            st.subheader("📊 Fib Score vs Signal Score")
            try:
                import plotly.express as px
                fig_scatter = px.scatter(
                    scan_df,
                    x='signal_score', y='fib_score',
                    text='symbol', color='grade',
                    size='vol_ratio',
                    color_discrete_map={
                        'A+': '#00ff88', 'A': '#88ff44',
                        'B+': '#ffd700', 'B': '#ff8800'
                    },
                    labels={
                        'signal_score': 'Signal Score',
                        'fib_score':    'Fib Score',
                        'grade':        'เกรด',
                    },
                    template='plotly_dark',
                    height=400,
                )
                fig_scatter.update_traces(
                    textposition='top center',
                    textfont=dict(size=10, color='white'),
                    marker=dict(opacity=0.85),
                )
                fig_scatter.update_layout(
                    paper_bgcolor='#0e1117',
                    plot_bgcolor='#0e1117',
                    font=dict(color='white'),
                )
                # Add quadrant lines
                fig_scatter.add_hline(y=60, line=dict(color='#555', dash='dot'))
                fig_scatter.add_vline(x=55, line=dict(color='#555', dash='dot'))
                fig_scatter.add_annotation(
                    x=75, y=85, text="🎯 Best Zone",
                    font=dict(color='#00ff88', size=12),
                    showarrow=False
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            except Exception as e:
                st.warning(f"Scatter plot error: {e}")

    else:
        # Placeholder before scan
        st.markdown("""
        <div style='text-align:center; padding:60px 20px; background:#1a1a2e;
             border-radius:15px; border:2px dashed #333; margin:20px 0'>
        <h2 style='color:#ffd700'>🔭 Fibonacci Scanner</h2>
        <p style='color:#aaa; font-size:1.1rem'>
        สแกนหุ้น SET อัตโนมัติ หาตัวที่
        </p>
        <div style='display:flex; justify-content:center; gap:30px; flex-wrap:wrap; margin:20px 0'>
          <div style='background:#0d3320; border:1px solid #00ff88; border-radius:8px; padding:12px 20px'>
            <b style='color:#00ff88'>📐 อยู่ใน Golden Zone</b><br>
            <small style='color:#aaa'>Fib 38.2%–61.8%</small>
          </div>
          <div style='background:#1a1a00; border:1px solid #ffd700; border-radius:8px; padding:12px 20px'>
            <b style='color:#ffd700'>📊 Signal Score สูง</b><br>
            <small style='color:#aaa'>Trend + Momentum + Volume</small>
          </div>
          <div style='background:#0d0d33; border:1px solid #4488ff; border-radius:8px; padding:12px 20px'>
            <b style='color:#4488ff'>⚖️ R:R ดี ≥ 1:1.5</b><br>
            <small style='color:#aaa'>กำไรมากกว่าความเสี่ยง</small>
          </div>
        </div>
        <p style='color:#666'>กด <b style='color:white'>🚀 เริ่มสแกน</b> ด้านบนเพื่อเริ่มต้น</p>
        </div>
        """, unsafe_allow_html=True)

    st.error("⚠️ ผลการสแกนเป็นเพียงเครื่องมือช่วยคัดกรองเบื้องต้นเท่านั้น ไม่ใช่คำแนะนำการลงทุน")


# ══════════════════════════════════════════════════════════════════════
# TAB 7 — CANDLESTICK PATTERNS + BELL CURVE
# ══════════════════════════════════════════════════════════════════════
with tab7:
    st.subheader(f"🕯️ Candlestick Pattern Analysis — {symbol}")

    # ── Settings bar ──────────────────────────────────────────────────
    cfg1, cfg2, cfg3 = st.columns(3)
    candle_period = cfg1.selectbox(
        "📅 ช่วงเวลากราฟ",
        ["1mo","3mo","6mo","1y"],
        index=1,
        format_func=lambda x: {"1mo":"1 เดือน","3mo":"3 เดือน","6mo":"6 เดือน","1y":"1 ปี"}[x],
        key="candle_period_sel",
    )
    bell_window = cfg2.slider("🔔 Bell Curve Window (วัน)", 20, 120, 60, 10)
    min_conf = cfg3.slider("🎯 Confidence ขั้นต่ำ", 40, 90, 55, 5,
                           help="กรอง pattern ที่มี confidence ต่ำออก")

    # ── Load data ─────────────────────────────────────────────────────
    @st.cache_data(ttl=300)
    def _load_candle_df(sym, period):
        from modules.data_fetcher import get_historical_data
        from modules.indicators import add_all_indicators
        d = get_historical_data(sym, period)
        if not d.empty:
            try: d = add_all_indicators(d)
            except: pass
        return d

    candle_df = _load_candle_df(symbol, candle_period)

    if candle_df.empty:
        st.warning("ไม่สามารถโหลดข้อมูลได้")
    else:
        # ── Detect patterns ───────────────────────────────────────────
        patterns_all  = detect_patterns_full(candle_df)
        patterns_show = [p for p in patterns_all if p['confidence'] >= min_conf]
        bell          = analyze_bell_curve(candle_df, window=bell_window)

        # ── Summary metrics ───────────────────────────────────────────
        buy_p  = [p for p in patterns_show if p['type'] == 'BUY']
        sell_p = [p for p in patterns_show if p['type'] == 'SELL']
        neu_p  = [p for p in patterns_show if p['type'] == 'NEUTRAL']

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🟢 Buy Patterns",  len(buy_p))
        m2.metric("🔴 Sell Patterns", len(sell_p))
        m3.metric("⚖️ Neutral",       len(neu_p))
        if bell:
            z_color = "🔴" if abs(bell['z_score']) > 2 else "🟡" if abs(bell['z_score']) > 1 else "🟢"
            m4.metric(f"{z_color} Z-score", f"{bell['z_score']:+.2f}",
                      delta=f"Percentile {bell['percentile']:.0f}%")
            m5.metric("🔔 Revert Prob",
                      f"{bell['reversion_prob']:.0f}%",
                      delta=bell['direction'])

        st.divider()

        # ── Left: Candle chart | Right: Pattern cards ─────────────────
        chart_col, card_col = st.columns([3, 1])

        with chart_col:
            fig_candle = plot_candlestick_analysis(candle_df, patterns_show, symbol)
            st.plotly_chart(fig_candle, use_container_width=True)

        with card_col:
            st.markdown("### 🎯 Patterns พบล่าสุด")
            if not patterns_show:
                st.info(f"ไม่พบ pattern ที่ confidence ≥ {min_conf}%\nลอง ลด confidence ขั้นต่ำ")
            else:
                for p in patterns_show[:8]:
                    type_color = {"BUY":"#00ff88","SELL":"#ff4444","NEUTRAL":"#ffd700"}.get(p['type'],'#888')
                    strength_bg = {"STRONG":"rgba(255,215,0,0.15)","MEDIUM":"rgba(100,100,100,0.15)","WEAK":"rgba(50,50,50,0.1)"}.get(p['strength'],'')
                    st.markdown(f"""
                    <div style='background:{strength_bg}; border-left:3px solid {type_color};
                         border-radius:6px; padding:8px 10px; margin-bottom:8px'>
                    <div style='display:flex; justify-content:space-between'>
                        <b style='color:{type_color}'>{p['emoji']} {p['pattern']}</b>
                        <span style='color:#ffd700; font-size:0.85rem'>{p['confidence']}%</span>
                    </div>
                    <div style='color:#aaa; font-size:0.8rem; margin-top:4px'>{p['description']}</div>
                    <div style='color:#666; font-size:0.75rem; margin-top:3px'>💡 {p['tip']}</div>
                    </div>
                    """, unsafe_allow_html=True)

        st.divider()

        # ── Bell Curve section ────────────────────────────────────────
        st.subheader(f"🔔 Bell Curve & Mean Reversion — {symbol}")

        if bell:
            # Key stats row
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("📍 ราคาปัจจุบัน", f"{bell['current']:.2f}")
            b2.metric("📊 Mean (Avg)", f"{bell['mean']:.2f}")
            b3.metric("📐 Std Dev", f"{bell['std']:.2f}")
            b4.metric("📏 Z-score", f"{bell['z_score']:+.2f}",
                      delta=f"Percentile {bell['percentile']:.0f}%")
            b5.metric("🔁 Revert Prob", f"{bell['reversion_prob']:.0f}%",
                      delta=bell['direction'])
            b6.metric("📈 BB Position", bell['bb_label'][:20])

            # Regime box
            regime_color = {
                "STRETCHED_EXTREME": "#ff2244",
                "STRETCHED_HIGH":    "#ff6644",
                "STRETCHED":         "#ffa500",
                "NORMAL":            "#00cc66",
                "COMPRESSED":        "#00bfff",
            }.get(bell['regime'], '#888')

            z = bell['z_score']
            regime_msg = (
                f"ราคาอยู่ **{abs(z):.1f} sigma** {'สูงกว่า' if z > 0 else 'ต่ำกว่า'} ค่าเฉลี่ย "
                f"— โอกาสกลับ{'ลง' if z > 0 else 'ขึ้น'}สู่ mean: **{bell['reversion_prob']:.0f}%**"
            )

            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.3); border:2px solid {regime_color};
                 border-radius:10px; padding:14px; margin:10px 0'>
            <div style='display:flex; align-items:center; gap:12px'>
                <span style='font-size:1.8rem'>🔔</span>
                <div>
                    <b style='color:{regime_color}; font-size:1.1rem'>{bell['regime_th']}</b><br>
                    <span style='color:#ccc'>{regime_msg}</span>
                </div>
            </div>
            </div>
            """, unsafe_allow_html=True)

            # Bell Curve chart
            fig_bell = plot_bell_curve(bell, symbol)
            st.plotly_chart(fig_bell, use_container_width=True)

            # Reading guide
            with st.expander("📖 วิธีอ่าน Bell Curve & Z-score"):
                st.markdown("""
**Z-score คืออะไร?**
> Z-score บอกว่าราคาปัจจุบันห่างจากค่าเฉลี่ยกี่ "ส่วนเบี่ยงเบนมาตรฐาน (Sigma)"

| Z-score | ความหมาย | โอกาส Mean Reversion |
|---------|----------|---------------------|
| 0 ถึง ±1σ | ปกติ — ราคาอยู่ในช่วงปกติ | ~35% |
| ±1σ ถึง ±2σ | ยืดตัว — เริ่มออกไปไกล | ~48–62% |
| ±2σ ถึง ±2.5σ | Stretched — ออกไปไกลมาก | ~75% |
| > ±2.5σ | Extreme — หายากมาก | ~85% |

**Bell Curve บอกอะไร?**
- **Histogram** = distribution ของราคาจริงในช่วงที่เลือก
- **Curve เหลือง** = ideal normal distribution
- **เส้นแดงแนวตั้ง** = ราคาปัจจุบัน
- ถ้าราคาอยู่นอก ±2σ = "ผิดปกติ" → โอกาส revert สูง

**Return Distribution**
- แสดงการกระจาย %change รายวัน
- Fat tail = หุ้นผันผวนสูง มี extreme move บ่อย

**ใช้ร่วมกับ Fibonacci อย่างไร?**
- ราคาที่อยู่ใน Golden Zone (38.2–61.8%) + Z-score > 2 ลงมา = โอกาสซื้อดีมาก
- ราคาที่อยู่ใกล้ Fib 1.618 + Z-score > 2.5 = take profit zone
                """)

        st.error("⚠️ ข้อมูลนี้เป็นการวิเคราะห์ทางสถิติเท่านั้น ไม่ใช่คำแนะนำการลงทุน")


# ─── SMART AUTO REFRESH ───────────────────────────────────────────────
if auto_refresh:
    if is_market_open():
        time.sleep(refresh_interval)
        st.rerun()
    else:
        # ตลาดปิด: refresh ช้าลง 5 นาที
        time.sleep(300)
        st.rerun()
