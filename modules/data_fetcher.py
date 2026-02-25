import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
FINNHUB_KEY       = os.getenv("FINNHUB_API_KEY", "")
SETTRADE_APP_ID   = os.getenv("SETTRADE_APP_ID", "")
SETTRADE_SECRET   = os.getenv("SETTRADE_APP_SECRET", "")
SETTRADE_SANDBOX  = os.getenv("SETTRADE_SANDBOX", "true").lower() == "true"

# Init SETTRADE client ถ้ามี credentials
_st_client = None
def _get_st_client():
    global _st_client
    if _st_client is None and SETTRADE_APP_ID and SETTRADE_SECRET:
        try:
            from modules.settrade_realtime import SettradeClient
            _st_client = SettradeClient(SETTRADE_APP_ID, SETTRADE_SECRET, SETTRADE_SANDBOX)
        except Exception as e:
            print(f"[SETTRADE] init error: {e}")
    return _st_client


def get_historical_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    ดึงข้อมูลย้อนหลัง — รองรับทั้ง daily และ intraday
    interval: "1d","5m","15m","30m","60m"
    4H = ส่ง interval="60m" แล้ว resample ทีหลัง
    """
    try:
        ticker = yf.Ticker(f"{symbol}.BK")
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                return pd.DataFrame()
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()


def resample_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1H data → 4H candles"""
    if df.empty:
        return df
    try:
        df4 = df.resample('4h').agg({
            'Open':   'first',
            'High':   'max',
            'Low':    'min',
            'Close':  'last',
            'Volume': 'sum',
        }).dropna()
        return df4
    except Exception:
        return df


def get_realtime_quote(symbol: str) -> dict:
    """
    ดึงราคาปัจจุบัน — priority:
    1. SETTRADE OpenAPI (realtime, ถ้า configured)
    2. Finnhub (ถ้ามี API key)
    3. yfinance (fallback, delay ~15 นาที)
    """
    default = {
        "price": 0.0, "change": 0.0, "pct_change": 0.0,
        "high": 0.0, "low": 0.0, "open": 0.0,
        "prev_close": 0.0, "volume": 0, "source": "unknown"
    }

    # ── Priority 1: SETTRADE (realtime) ─────────────────────────────
    try:
        st = _get_st_client()
        if st:
            q = st.get_quote(symbol)
            if q and q.get("price", 0) > 0:
                return q
    except Exception as e:
        print(f"[SETTRADE] quote error: {e}")

    # ── Priority 2: Finnhub ──────────────────────────────────────────
    if FINNHUB_KEY:
        try:
            url = "https://finnhub.io/api/v1/quote"
            params = {"symbol": f"{symbol}.BK", "token": FINNHUB_KEY}
            r = requests.get(url, params=params, timeout=5)
            data = r.json()
            if data.get("c", 0) > 0:
                return {
                    "price":      float(data.get("c", 0)),
                    "change":     float(data.get("d", 0)),
                    "pct_change": float(data.get("dp", 0)),
                    "high":       float(data.get("h", 0)),
                    "low":        float(data.get("l", 0)),
                    "open":       float(data.get("o", 0)),
                    "prev_close": float(data.get("pc", 0)),
                    "volume":     int(data.get("v", 0)),
                    "timestamp":  datetime.fromtimestamp(data.get("t", 0)),
                    "source":     "finnhub",
                }
        except Exception as e:
            print(f"Finnhub error: {e}")

    # ── Priority 3: yfinance (delay ~15 min) ─────────────────────────
    try:
        ticker = yf.Ticker(f"{symbol}.BK")
        fi = ticker.fast_info
        price = float(fi.last_price or 0)
        prev  = float(fi.previous_close or price)
        change = price - prev
        pct    = (change / prev * 100) if prev != 0 else 0.0
        return {
            "price":      price,
            "change":     change,
            "pct_change": pct,
            "high":       float(fi.day_high or price),
            "low":        float(fi.day_low or price),
            "open":       float(fi.open or price),
            "prev_close": prev,
            "volume":     int(fi.three_month_average_volume or 0),
            "source":     "yfinance_delayed",
        }
    except Exception as e:
        print(f"yfinance quote error: {e}")
        return default


def get_dividend_history(symbol: str) -> pd.DataFrame:
    """ดึงประวัติปันผล 5 ปีจาก yfinance"""
    try:
        ticker = yf.Ticker(f"{symbol}.BK")
        divs = ticker.dividends
        if divs is None or divs.empty:
            return pd.DataFrame()

        divs = divs.reset_index()
        divs.columns = ['ex_date', 'amount']
        divs['ex_date'] = pd.to_datetime(divs['ex_date'])
        if divs['ex_date'].dt.tz is not None:
            divs['ex_date'] = divs['ex_date'].dt.tz_localize(None)

        cutoff = datetime.now() - pd.DateOffset(years=5)
        divs = divs[divs['ex_date'] >= cutoff].copy()
        divs['year'] = divs['ex_date'].dt.year
        divs['amount'] = divs['amount'].round(4)

        return divs.sort_values('ex_date', ascending=False).reset_index(drop=True)
    except Exception as e:
        print(f"Dividend error for {symbol}: {e}")
        return pd.DataFrame()


def get_stock_info(symbol: str) -> dict:
    """ดึงข้อมูลพื้นฐานบริษัทจาก yfinance"""
    try:
        ticker = yf.Ticker(f"{symbol}.BK")
        info = ticker.info
        def safe_round(val, dec=2):
            try:
                return round(float(val), dec) if val else 0
            except:
                return 0

        return {
            "name":               info.get("longName", symbol),
            "sector":             info.get("sector", "N/A"),
            "industry":           info.get("industry", "N/A"),
            "market_cap":         info.get("marketCap", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "pe_ratio":           safe_round(info.get("trailingPE")),
            "pbv":                safe_round(info.get("priceToBook")),
            "eps":                safe_round(info.get("trailingEps")),
            "roe":                safe_round(info.get("returnOnEquity", 0) * 100),
            "roa":                safe_round(info.get("returnOnAssets", 0) * 100),
            "debt_equity":        safe_round(info.get("debtToEquity")),
            "current_ratio":      safe_round(info.get("currentRatio")),
            "div_yield":          safe_round(info.get("dividendYield", 0) * 100),
            "52w_high":           info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low":            info.get("fiftyTwoWeekLow", "N/A"),
            "beta":               safe_round(info.get("beta")),
        }
    except Exception as e:
        print(f"Stock info error for {symbol}: {e}")
        return {
            "name": symbol, "sector": "N/A", "industry": "N/A",
            "market_cap": 0, "shares_outstanding": 0,
            "pe_ratio": 0, "pbv": 0, "eps": 0,
            "roe": 0, "roa": 0, "debt_equity": 0,
            "current_ratio": 0, "div_yield": 0,
            "52w_high": "N/A", "52w_low": "N/A", "beta": 0
        }


def search_stocks(query: str) -> list:
    """
    ค้นหาหุ้นไทย SET/MAI จาก yfinance
    รองรับทั้ง symbol (PTT, KBANK) และชื่อบริษัท
    คืนค่า list of dict: {symbol, name, exchange}
    """
    if not query or len(query.strip()) < 1:
        return []
    query = query.strip().upper()
    results = []
    seen = set()

    # 1. ลอง query ตรงๆ ว่าเป็น symbol ถูกต้องไหม
    try:
        ticker = yf.Ticker(f"{query}.BK")
        info = ticker.fast_info
        price = getattr(info, 'last_price', None)
        if price and price > 0:
            name = getattr(ticker, 'info', {}).get('longName', query) or query
            results.append({"symbol": query, "name": name, "exchange": "SET"})
            seen.add(query)
    except Exception:
        pass

    # 2. yfinance search API
    try:
        search_result = yf.Search(query, max_results=20)
        quotes = search_result.quotes if hasattr(search_result, 'quotes') else []
        for q in quotes:
            sym_raw = q.get('symbol', '')
            # กรอง: ต้องลงท้ายด้วย .BK (SET) หรือ .BK เท่านั้น
            if not sym_raw.endswith('.BK'):
                continue
            sym = sym_raw.replace('.BK', '')
            if sym in seen:
                continue
            name = q.get('longname') or q.get('shortname') or sym
            results.append({"symbol": sym, "name": name, "exchange": "SET"})
            seen.add(sym)
            if len(results) >= 15:
                break
    except Exception:
        pass

    # 3. ถ้าไม่มีผลจาก search API — ลองเดา symbols ที่คล้ายกัน
    if len(results) == 0:
        candidates = [query, query + "F", "A" + query, query[:3]]
        for sym in candidates:
            if sym in seen:
                continue
            try:
                t = yf.Ticker(f"{sym}.BK")
                fi = t.fast_info
                price = getattr(fi, 'last_price', None)
                if price and price > 0:
                    name = t.info.get('longName', sym) if hasattr(t, 'info') else sym
                    results.append({"symbol": sym, "name": name or sym, "exchange": "SET"})
                    seen.add(sym)
            except Exception:
                pass

    return results[:15]


def validate_symbol(symbol: str) -> bool:
    """ตรวจสอบว่า symbol นี้มีข้อมูลใน yfinance ไหม"""
    try:
        ticker = yf.Ticker(f"{symbol}.BK")
        fi = ticker.fast_info
        price = getattr(fi, 'last_price', None)
        return bool(price and float(price) > 0)
    except Exception:
        return False


def is_market_open() -> bool:
    """เช็คว่าตลาด SET เปิดอยู่ไหม (UTC+7)"""
    try:
        from datetime import timezone, timedelta
        tz_bkk = timezone(timedelta(hours=7))
        now = datetime.now(tz_bkk)
        if now.weekday() >= 5:
            return False
        h, m = now.hour, now.minute
        morning   = (10, 0) <= (h, m) <= (12, 30)
        afternoon = (14, 30) <= (h, m) <= (17, 0)
        return morning or afternoon
    except:
        return False


def calculate_dividend_cagr(divs: pd.DataFrame) -> float:
    """คำนวณ CAGR ของปันผลรายปีย้อนหลัง 5 ปี"""
    try:
        if divs is None or divs.empty:
            return 0.0
        annual = divs.groupby('year')['amount'].sum().sort_index()
        if len(annual) < 2:
            return 0.0
        first = annual.iloc[0]
        last  = annual.iloc[-1]
        years = annual.index[-1] - annual.index[0]
        if first <= 0 or years <= 0:
            return 0.0
        cagr = ((last / first) ** (1 / years) - 1) * 100
        return round(cagr, 2)
    except:
        return 0.0
