import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

DARK_BG    = '#0e1117'
DARK_PANEL = '#1e2130'
GREEN      = '#00ff88'
RED        = '#ff4444'
YELLOW     = '#ffd700'
BLUE       = '#4488ff'
ORANGE     = '#ff8800'
PURPLE     = '#cc88ff'


def _base_layout(title: str, height: int = 500, margin: dict = None) -> dict:
    return dict(
        title=dict(text=title, font=dict(color='white', size=14)),
        template='plotly_dark',
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        height=height,
        margin=margin or dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='#333', borderwidth=1),
        xaxis=dict(gridcolor='#1e2130', zerolinecolor='#333'),
        yaxis=dict(gridcolor='#1e2130', zerolinecolor='#333'),
    )


def plot_candlestick(df: pd.DataFrame, symbol: str,
                     show_ema: bool = True,
                     show_bb: bool = True,
                     show_ichimoku: bool = False,
                     show_vwap: bool = False,
                     targets: dict = None,
                     signals_list: list = None) -> go.Figure:
    """Full dark-theme candlestick chart with indicators and targets"""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=[None, None]
    )

    # ── VWAP (intraday) ───────────────────────────────────────────────
    if show_vwap and 'Volume' in df.columns:
        typical = (df['High'] + df['Low'] + df['Close']) / 3
        cum_tp_vol = (typical * df['Volume']).cumsum()
        cum_vol    = df['Volume'].cumsum().replace(0, np.nan)
        vwap_line  = cum_tp_vol / cum_vol
        fig.add_trace(go.Scatter(
            x=df.index, y=vwap_line,
            name='VWAP', line=dict(color='#ff9900', width=1.5, dash='dot'),
            hovertemplate="<b>VWAP</b>: %{y:.2f}<extra></extra>",
        ), row=1, col=1)

    # ── Candlesticks ─────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        name='Price',
        increasing_line_color=GREEN,
        decreasing_line_color=RED,
        increasing_fillcolor=GREEN,
        decreasing_fillcolor=RED,
        hovertext=[
            f"<b>{idx.strftime('%d %b %Y')}</b><br>"
            f"Open:  <b>{o:.2f}</b><br>"
            f"High:  <b style='color:#00ff88'>{h:.2f}</b><br>"
            f"Low:   <b style='color:#ff4444'>{l:.2f}</b><br>"
            f"Close: <b>{c:.2f}</b><br>"
            f"Change: <b style='color:{'#00ff88' if c>=o else '#ff4444'}'>{((c-o)/o*100):+.2f}%</b>"
            for idx, o, h, l, c in zip(
                df.index, df['Open'], df['High'], df['Low'], df['Close']
            )
        ],
        hoverinfo='text',
    ), row=1, col=1)

    # ── EMA Lines ────────────────────────────────────────────────────
    if show_ema:
        ema_configs = [
            ('EMA9',  '#ff9900', 1,   'EMA 9'),
            ('EMA21', '#00ccff', 1.5, 'EMA 21'),
            ('EMA50', '#ff44cc', 2,   'EMA 50'),
            ('EMA200','#ffff44', 2.5, 'EMA 200'),
        ]
        for col, color, width, name in ema_configs:
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col],
                    name=name, line=dict(color=color, width=width),
                    opacity=0.85,
                    hovertemplate=f"<b>{name}</b>: %{{y:.2f}} THB<extra></extra>",
                ), row=1, col=1)

    # ── Bollinger Bands ───────────────────────────────────────────────
    if show_bb and 'BB_upper' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_upper'],
            name='BB Upper', line=dict(color='rgba(100,100,255,0.6)', width=1, dash='dot'),
            hovertemplate="<b>BB Upper</b>: %{y:.2f} THB<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_lower'],
            name='BB Lower', line=dict(color='rgba(100,100,255,0.6)', width=1, dash='dot'),
            fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
            hovertemplate="<b>BB Lower</b>: %{y:.2f} THB<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_middle'],
            name='BB Mid', line=dict(color='rgba(100,100,255,0.4)', width=1, dash='dash'),
            hovertemplate="<b>BB Mid</b>: %{y:.2f} THB<extra></extra>",
        ), row=1, col=1)

    # ── Ichimoku Cloud ────────────────────────────────────────────────
    if show_ichimoku:
        if 'Tenkan' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Tenkan'],
                name='Tenkan', line=dict(color='#ff6688', width=1)
            ), row=1, col=1)
        if 'Kijun' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Kijun'],
                name='Kijun', line=dict(color='#6688ff', width=1)
            ), row=1, col=1)
        if 'Senkou_A' in df.columns and 'Senkou_B' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Senkou_A'],
                name='Senkou A', line=dict(color='rgba(0,200,100,0.3)', width=1),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Senkou_B'],
                name='Senkou B', line=dict(color='rgba(255,100,100,0.3)', width=1),
                fill='tonexty', fillcolor='rgba(100,200,100,0.08)'
            ), row=1, col=1)

    # ── Price Targets ────────────────────────────────────────────────
    if targets:
        last_date = df.index[-1]
        start_date = df.index[max(0, len(df) - 60)]

        # Buy Zone shading
        bz_low  = targets['buy_zone']['low']
        bz_high = targets['buy_zone']['high']
        fig.add_hrect(
            y0=bz_low, y1=bz_high,
            fillcolor='rgba(0,255,136,0.08)',
            line=dict(color='rgba(0,255,136,0.4)', width=1, dash='dot'),
            annotation_text="Buy Zone", annotation_position="right",
            row=1, col=1
        )

        # Stop Loss line
        sl = targets['stop_loss']
        fig.add_hline(
            y=sl, line=dict(color=RED, width=1.5, dash='dash'),
            annotation_text=f"SL: {sl:.2f}",
            annotation_font_color=RED,
            row=1, col=1
        )

        # Target lines
        colors_tp = [GREEN, YELLOW, ORANGE]
        for idx, tp in enumerate(targets['targets']):
            fig.add_hline(
                y=tp, line=dict(color=colors_tp[idx], width=1, dash='dash'),
                annotation_text=f"TP{idx+1}: {tp:.2f}",
                annotation_font_color=colors_tp[idx],
                row=1, col=1
            )

    # ── Volume Bars ───────────────────────────────────────────────────
    colors = [GREEN if df['Close'].iloc[i] >= df['Open'].iloc[i] else RED
              for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume', marker_color=colors,
        opacity=0.7, showlegend=False,
        hovertemplate="<b>Volume</b>: %{y:,.0f}<extra></extra>",
    ), row=2, col=1)

    if 'Vol_SMA20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Vol_SMA20'],
            name='Vol MA20', line=dict(color=YELLOW, width=1),
            hovertemplate="<b>Vol MA20</b>: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        **_base_layout(f"{symbol} — Price Chart", height=680),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(20,22,35,0.95)',
            bordercolor='#444',
            font=dict(color='white', size=12, family='monospace'),
        ),
    )
    fig.update_xaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='#555', spikethickness=1,
    )
    fig.update_yaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='#555', spikethickness=1,
        title_text="ราคา (THB)", row=1, col=1,
    )
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_macd(df: pd.DataFrame) -> go.Figure:
    """MACD chart with histogram"""
    fig = go.Figure()

    if 'MACD_hist' in df.columns:
        colors = [GREEN if v >= 0 else RED for v in df['MACD_hist'].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df['MACD_hist'],
            name='Histogram', marker_color=colors, opacity=0.7,
            hovertemplate="<b>Histogram</b>: %{y:.4f}<extra></extra>",
        ))

    if 'MACD' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD'],
            name='MACD', line=dict(color=BLUE, width=1.5),
            hovertemplate="<b>MACD</b>: %{y:.4f}<extra></extra>",
        ))

    if 'MACD_signal' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['MACD_signal'],
            name='Signal', line=dict(color=ORANGE, width=1.5, dash='dot'),
            hovertemplate="<b>Signal</b>: %{y:.4f}<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color='#555', width=1))
    fig.update_layout(
        **_base_layout("MACD (12,26,9)", height=250),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(20,22,35,0.95)', bordercolor='#444',
            font=dict(color='white', size=11, family='monospace'),
        ),
    )
    fig.update_xaxes(showspikes=True, spikemode='across', spikecolor='#555', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode='across', spikecolor='#555', spikethickness=1)
    return fig


def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold zones"""
    fig = go.Figure()

    if 'RSI' in df.columns:
        rsi = df['RSI']

        # Oversold zone fill
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            name='RSI', line=dict(color=PURPLE, width=2),
            fill=None,
            hovertemplate="<b>RSI</b>: %{y:.2f}<extra></extra>",
        ))

        # Reference lines
        for level, color, label in [(70, RED, 'Overbought'), (30, GREEN, 'Oversold'), (50, '#555', '')]:
            fig.add_hline(
                y=level,
                line=dict(color=color, width=1, dash='dash'),
                annotation_text=label if label else None,
                annotation_font_color=color,
            )

        # Shading
        fig.add_hrect(y0=70, y1=100, fillcolor='rgba(255,68,68,0.07)',
                      line_width=0)
        fig.add_hrect(y0=0,  y1=30, fillcolor='rgba(0,255,136,0.07)',
                      line_width=0)

    fig.update_layout(
        **_base_layout("RSI (14)", height=250),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(20,22,35,0.95)', bordercolor='#444',
            font=dict(color='white', size=11, family='monospace'),
        ),
    )
    fig.update_yaxes(range=[0, 100], gridcolor='#1e2130')
    fig.update_xaxes(showspikes=True, spikemode='across', spikecolor='#555', spikethickness=1)
    fig.update_yaxes(showspikes=True, spikemode='across', spikecolor='#555', spikethickness=1)
    return fig


def plot_dividend_chart(divs: pd.DataFrame) -> go.Figure:
    """Annual dividend bar chart with yield line"""
    if divs is None or divs.empty:
        return go.Figure()

    annual = divs.groupby('year')['amount'].sum().reset_index()
    annual.columns = ['ปี', 'ปันผลรวม']

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=annual['ปี'].astype(str),
        y=annual['ปันผลรวม'],
        name='ปันผล/หุ้น (THB)',
        marker_color=GREEN,
        opacity=0.8,
        text=annual['ปันผลรวม'].round(2),
        textposition='outside',
        textfont=dict(color='white')
    ), secondary_y=False)

    # Yield line (mock: we don't have price at xd here, show relative change)
    if len(annual) > 1:
        yoy_change = annual['ปันผลรวม'].pct_change() * 100
        fig.add_trace(go.Scatter(
            x=annual['ปี'].astype(str),
            y=yoy_change,
            name='การเปลี่ยนแปลง YoY (%)',
            line=dict(color=YELLOW, width=2),
            mode='lines+markers',
            marker=dict(size=8)
        ), secondary_y=True)

    fig.update_layout(
        **_base_layout("💰 ประวัติปันผล 5 ปี", height=350),
        barmode='group',
    )
    fig.update_yaxes(title_text="ปันผล (THB/หุ้น)", secondary_y=False)
    fig.update_yaxes(title_text="YoY Change (%)", secondary_y=True)

    return fig


def plot_fibonacci(df: "pd.DataFrame", symbol: str, current_price: float) -> "go.Figure":
    """
    Interactive Fibonacci Retracement Chart
    - หา Swing High / Swing Low อัตโนมัติ
    - แสดง Fib levels 0.0 → 1.618 พร้อม zone shading
    - บอก zone ที่ราคาอยู่ปัจจุบัน
    """
    swing_high = float(df['High'].max())
    swing_low  = float(df['Low'].min())
    fib_range  = swing_high - swing_low

    mid = len(df) // 2
    is_uptrend = df['Close'].iloc[mid:].mean() >= df['Close'].iloc[:mid].mean()
    base = swing_low if is_uptrend else swing_high
    direction = 1 if is_uptrend else -1

    FIB_LEVELS = [
        (0.000, "0.0%",   '#888888', 'rgba(136,136,136,0.05)'),
        (0.236, "23.6%",  '#00bfff', 'rgba(0,191,255,0.06)'),
        (0.382, "38.2%",  '#00ff88', 'rgba(0,255,136,0.08)'),
        (0.500, "50.0%",  '#ffd700', 'rgba(255,215,0,0.08)'),
        (0.618, "61.8%",  '#ff8800', 'rgba(255,136,0,0.10)'),
        (0.786, "78.6%",  '#ff4488', 'rgba(255,68,136,0.07)'),
        (1.000, "100%",   '#ff4444', 'rgba(255,68,68,0.05)'),
    ]
    FIB_EXTENSIONS = [
        (1.272, "127.2%", '#cc88ff'),
        (1.618, "161.8%", '#aa44ff'),
    ]

    def fib_price(ratio):
        return base + direction * fib_range * ratio

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.80, 0.20],
    )

    # Candlestick with rich hover
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'],   close=df['Close'],
        name='Price',
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444',
        increasing_fillcolor='rgba(0,255,136,0.3)',
        decreasing_fillcolor='rgba(255,68,68,0.3)',
        hovertext=[
            f"<b>{idx.strftime('%d %b %Y')}</b><br>"
            f"Open:  <b>{o:.2f}</b><br>"
            f"High:  <b style='color:#00ff88'>{h:.2f}</b><br>"
            f"Low:   <b style='color:#ff4444'>{l:.2f}</b><br>"
            f"Close: <b>{c:.2f}</b><br>"
            f"Change: <b>{((c-o)/o*100):+.2f}%</b>"
            for idx, o, h, l, c in zip(
                df.index, df['Open'], df['High'], df['Low'], df['Close']
            )
        ],
        hoverinfo='text',
    ), row=1, col=1)

    # Zone shading between adjacent levels
    for i in range(len(FIB_LEVELS) - 1):
        r0, lbl0, c0, fill0 = FIB_LEVELS[i]
        r1, lbl1, c1, fill1 = FIB_LEVELS[i + 1]
        p0, p1 = fib_price(r0), fib_price(r1)
        if fill0:
            fig.add_hrect(
                y0=min(p0, p1), y1=max(p0, p1),
                fillcolor=fill0, line_width=0,
                row=1, col=1
            )

    # Fib lines — use Scatter instead of hline so hover works
    all_x = [df.index[0], df.index[-1]]
    for ratio, label, color, _ in FIB_LEVELS:
        price = fib_price(ratio)
        is_golden = (ratio == 0.618)
        desc_map = {
            0.000: "จุดเริ่มต้น",
            0.236: "แนวรับ/ต้านอ่อน",
            0.382: "แนวรับ/ต้านปานกลาง",
            0.500: "กึ่งกลาง — จิตวิทยา",
            0.618: "🌟 Golden Ratio",
            0.786: "แนวรับ/ต้านแข็ง",
            1.000: "จุดสิ้นสุด",
        }
        desc = desc_map.get(ratio, label)
        fig.add_trace(go.Scatter(
            x=all_x, y=[price, price],
            mode='lines', name=f"Fib {label}",
            line=dict(color=color, width=2.5 if is_golden else 1.0,
                      dash='solid' if is_golden else 'dot'),
            showlegend=False,
            hovertemplate=f"<b>Fib {label}</b>  {price:.2f} THB<br><i>{desc}</i><extra></extra>",
        ), row=1, col=1)
        fig.add_annotation(
            x=1.01, xref='paper', y=price, yref='y',
            text=f"<b>{label}</b>  {price:.2f}",
            showarrow=False, font=dict(color=color, size=10),
            xanchor='left', bgcolor='rgba(14,17,23,0.8)',
        )

    # Extension lines
    ext_desc = {1.272: "Extension เป้าหมายแรก", 1.618: "🌟 Golden Extension"}
    for ratio, label, color in FIB_EXTENSIONS:
        price = fib_price(ratio)
        fig.add_trace(go.Scatter(
            x=all_x, y=[price, price],
            mode='lines', name=f"Fib {label}",
            line=dict(color=color, width=2.0 if ratio == 1.618 else 1.2, dash='dash'),
            showlegend=False,
            hovertemplate=f"<b>Fib {label}</b>  {price:.2f} THB<br><i>{ext_desc.get(ratio,'')}</i><extra></extra>",
        ), row=1, col=1)
        fig.add_annotation(
            x=1.01, xref='paper', y=price, yref='y',
            text=f"<b>{label}</b>  {price:.2f}",
            showarrow=False, font=dict(color=color, size=10),
            xanchor='left', bgcolor='rgba(14,17,23,0.8)',
        )

    # Current price line
    fig.add_trace(go.Scatter(
        x=all_x, y=[current_price, current_price],
        mode='lines', name='Current',
        line=dict(color='white', width=1.5, dash='dash'),
        showlegend=False,
        hovertemplate=f"<b>ราคาปัจจุบัน</b>  {current_price:.2f} THB<extra></extra>",
    ), row=1, col=1)
    fig.add_annotation(
        x=1.01, xref='paper', y=current_price, yref='y',
        text=f"▶ {current_price:.2f}",
        showarrow=False, font=dict(color='white', size=11),
        xanchor='left', bgcolor='rgba(60,60,90,0.9)',
    )

    # Swing High/Low markers
    high_idx = df['High'].idxmax()
    low_idx  = df['Low'].idxmin()
    fig.add_trace(go.Scatter(
        x=[high_idx], y=[swing_high], mode='markers+text',
        marker=dict(symbol='triangle-down', size=14, color='#ff4444'),
        text=[f"H {swing_high:.2f}"], textposition='top center',
        textfont=dict(color='#ff4444', size=10),
        name='Swing High',
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[low_idx], y=[swing_low], mode='markers+text',
        marker=dict(symbol='triangle-up', size=14, color='#00ff88'),
        text=[f"L {swing_low:.2f}"], textposition='bottom center',
        textfont=dict(color='#00ff88', size=10),
        name='Swing Low',
    ), row=1, col=1)

    # Volume
    vol_colors = ['#00ff88' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4444'
                  for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'], name='Volume',
        marker_color=vol_colors, opacity=0.6, showlegend=False
    ), row=2, col=1)

    # Find current zone
    zone_label = "นอกช่วง Fibonacci"
    for i in range(len(FIB_LEVELS) - 1):
        r0, lbl0, c0, _ = FIB_LEVELS[i]
        r1, lbl1, c1, _ = FIB_LEVELS[i + 1]
        p0, p1 = fib_price(r0), fib_price(r1)
        if min(p0, p1) <= current_price <= max(p0, p1):
            zone_label = f"ราคาอยู่ใน Zone {lbl0} – {lbl1}"
            break

    trend_th = "ขาขึ้น 📈" if is_uptrend else "ขาลง 📉"
    fig.update_layout(
        **_base_layout(
            f"{symbol} — Fibonacci Retracement | {trend_th} | {zone_label}",
            height=700,
            margin=dict(l=10, r=160, t=50, b=10),
        ),
        xaxis_rangeslider_visible=False,
        hovermode='closest',
        hoverlabel=dict(
            bgcolor='rgba(20,22,35,0.95)',
            bordercolor='#444',
            font=dict(color='white', size=12, family='monospace'),
        ),
    )
    fig.update_xaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='#666', spikethickness=1,
    )
    fig.update_yaxes(
        showspikes=True, spikemode='across', spikesnap='cursor',
        spikecolor='#666', spikethickness=1,
    )
    fig.update_yaxes(title_text="ราคา (THB)", row=1, col=1)
    fig.update_yaxes(title_text="Volume",     row=2, col=1)

    return fig


def plot_fibonacci_table(df: "pd.DataFrame", current_price: float) -> "pd.DataFrame":
    """สร้างตาราง Fibonacci levels พร้อมคำอธิบาย"""
    swing_high = float(df['High'].max())
    swing_low  = float(df['Low'].min())
    fib_range  = swing_high - swing_low
    mid = len(df) // 2
    is_uptrend = df['Close'].iloc[mid:].mean() >= df['Close'].iloc[:mid].mean()
    base = swing_low if is_uptrend else swing_high
    direction = 1 if is_uptrend else -1

    levels = [
        (0.000, "0.0%",   "จุดเริ่มต้น (Swing Low/High)"),
        (0.236, "23.6%",  "แนวรับ/ต้านอ่อน — จุดพักตัวแรก"),
        (0.382, "38.2%",  "แนวรับ/ต้านปานกลาง — จุดพักตัวที่ดี"),
        (0.500, "50.0%",  "กึ่งกลาง — จุดสำคัญทางจิตวิทยา"),
        (0.618, "61.8%",  "🌟 Golden Ratio — แนวรับ/ต้านแข็งแกร่งที่สุด"),
        (0.786, "78.6%",  "แนวรับ/ต้านแข็ง — ก่อนกลับ Swing เดิม"),
        (1.000, "100%",   "จุดสิ้นสุด (Swing High/Low เดิม)"),
        (1.272, "127.2%", "Extension — เป้าหมายแรก"),
        (1.618, "161.8%", "🌟 Golden Extension — เป้าหมายสูงสุด"),
    ]

    rows = []
    for ratio, label, desc in levels:
        price = base + direction * fib_range * ratio
        dist_pct = ((price - current_price) / current_price * 100) if current_price > 0 else 0
        rows.append({
            "Level":       label,
            "ราคา (THB)": round(price, 2),
            "ห่างจากราคา": f"{dist_pct:+.2f}%",
            "ความสำคัญ":   desc,
            "สถานะ":       "📍 ใกล้ราคาปัจจุบัน" if abs(dist_pct) < 3.0 else "",
        })
    import pandas as pd
    return pd.DataFrame(rows)
