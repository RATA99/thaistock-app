"""
SETTRADE OpenAPI Integration
────────────────────────────
Docs: https://developer.settrade.com/
OAuth2 PKCE flow + REST + WebSocket streaming

Credentials ต้องอยู่ใน .env:
    SETTRADE_APP_ID=...
    SETTRADE_APP_SECRET=...
"""
import os
import time
import base64
import hashlib
import hmac
import json
import threading
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional, Callable
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────
SETTRADE_BASE     = "https://api.settrade.com/api"
SETTRADE_WS_BASE  = "wss://streaming.settrade.com"
SETTRADE_TOKEN_URL = f"{SETTRADE_BASE}/oauth/token"

APP_ID     = os.getenv("SETTRADE_APP_ID",     "")
APP_SECRET = os.getenv("SETTRADE_APP_SECRET", "")


# ── Token Manager ──────────────────────────────────────────────────────
class _TokenManager:
    """
    App Authentication (application-level token)
    ไม่ต้อง login user — ใช้ client_credentials grant
    สำหรับ market data ที่ไม่ต้องการ user consent
    """
    def __init__(self):
        self._token:   Optional[str] = None
        self._expires: float = 0.0
        self._lock = threading.Lock()

    def _fetch(self) -> Optional[str]:
        if not APP_ID or not APP_SECRET:
            return None
        try:
            resp = requests.post(
                SETTRADE_TOKEN_URL,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     APP_ID,
                    "client_secret": APP_SECRET,
                    "scope":         "MarketData",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                self._token   = data.get("access_token")
                expires_in    = int(data.get("expires_in", 3600))
                self._expires = time.time() + expires_in - 60  # refresh 1min early
                return self._token
        except Exception as e:
            print(f"[SETTRADE] Token fetch error: {e}")
        return None

    def get(self) -> Optional[str]:
        with self._lock:
            if self._token and time.time() < self._expires:
                return self._token
            return self._fetch()


_token_mgr = _TokenManager()


def _headers() -> dict:
    token = _token_mgr.get()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def is_configured() -> bool:
    """ตรวจว่า credentials พร้อมใช้งาน"""
    return bool(APP_ID and APP_SECRET)


# ── Market Data REST ───────────────────────────────────────────────────

def get_quote(symbol: str) -> Optional[dict]:
    """
    ดึงราคาล่าสุด (real-time ถ้าตลาดเปิด)
    symbol: "PTT" (ไม่ต้องมี .BK)
    """
    if not is_configured():
        return None
    try:
        url = f"{SETTRADE_BASE}/market/quotes/{symbol}"
        resp = requests.get(url, headers=_headers(), timeout=5)
        if resp.status_code != 200:
            return None
        d = resp.json()
        # Map SETTRADE fields to our standard format
        return {
            "price":      float(d.get("last",       d.get("close", 0))),
            "change":     float(d.get("change",     0)),
            "pct_change": float(d.get("percentChange", d.get("pct_change", 0))),
            "high":       float(d.get("high",        0)),
            "low":        float(d.get("low",         0)),
            "open":       float(d.get("open",        0)),
            "prev_close": float(d.get("previousClose", d.get("prev_close", 0))),
            "volume":     int(d.get("volume",        0)),
            "bid":        float(d.get("bid",         0)),
            "ask":        float(d.get("ask",         0)),
            "bid_vol":    int(d.get("bidVolume",     0)),
            "ask_vol":    int(d.get("askVolume",     0)),
            "source":     "settrade_realtime",
        }
    except Exception as e:
        print(f"[SETTRADE] get_quote error {symbol}: {e}")
        return None


def get_intraday_ohlcv(symbol: str, interval: str = "5") -> Optional[pd.DataFrame]:
    """
    ดึงข้อมูล intraday OHLCV
    interval: "1", "5", "15", "30", "60" (นาที)
    """
    if not is_configured():
        return None
    try:
        url = f"{SETTRADE_BASE}/market/historical/{symbol}/intraday"
        params = {"resolution": interval, "limit": 500}
        resp = requests.get(url, headers=_headers(), params=params, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        bars = data.get("bars", data.get("data", data))
        if not bars:
            return None
        df = pd.DataFrame(bars)
        # Normalize column names
        col_map = {
            "time": "datetime", "t": "datetime",
            "open": "Open", "o": "Open",
            "high": "High", "h": "High",
            "low":  "Low",  "l": "Low",
            "close":"Close","c": "Close",
            "volume":"Volume","v":"Volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"], unit="s", utc=True)
            df["datetime"] = df["datetime"].dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
            df = df.set_index("datetime")
        for col in ["Open","High","Low","Close","Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Close"])
        return df[["Open","High","Low","Close","Volume"]].copy()
    except Exception as e:
        print(f"[SETTRADE] get_intraday_ohlcv error {symbol}: {e}")
        return None


def get_multi_quotes(symbols: list) -> dict:
    """
    ดึงราคาหลายหุ้นพร้อมกัน (batch)
    returns: {symbol: quote_dict, ...}
    """
    if not is_configured():
        return {}
    results = {}
    # SETTRADE batch endpoint (ถ้ามี)
    try:
        url = f"{SETTRADE_BASE}/market/quotes"
        params = {"symbols": ",".join(symbols)}
        resp = requests.get(url, headers=_headers(), params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data if isinstance(data, list) else data.get("data", [])
            for item in items:
                sym = item.get("symbol", "")
                if sym:
                    results[sym] = {
                        "price":      float(item.get("last",  0)),
                        "change":     float(item.get("change", 0)),
                        "pct_change": float(item.get("percentChange", 0)),
                        "volume":     int(item.get("volume", 0)),
                        "source":     "settrade_realtime",
                    }
            return results
    except Exception:
        pass

    # Fallback: individual requests (slower)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=5) as ex:
        fmap = {ex.submit(get_quote, s): s for s in symbols}
        for fut in as_completed(fmap):
            sym = fmap[fut]
            try:
                q = fut.result(timeout=5)
                if q:
                    results[sym] = q
            except Exception:
                pass
    return results


# ── WebSocket Streaming ────────────────────────────────────────────────

class SettradeTicker:
    """
    WebSocket streaming สำหรับราคา realtime
    
    ใช้งาน:
        ticker = SettradeTicker()
        ticker.subscribe(["PTT","KBANK","AOT"], callback=my_func)
        ticker.start()
        ...
        ticker.stop()
    
    callback signature: callback(symbol: str, data: dict)
    """

    def __init__(self):
        self._ws = None
        self._thread:   Optional[threading.Thread] = None
        self._running   = False
        self._symbols:  list = []
        self._callback: Optional[Callable] = None
        self._last:     dict = {}  # {symbol: latest_quote}

    def subscribe(self, symbols: list, callback: Optional[Callable] = None):
        self._symbols  = [s.upper() for s in symbols]
        self._callback = callback

    def get_latest(self, symbol: str) -> Optional[dict]:
        return self._last.get(symbol.upper())

    def get_all_latest(self) -> dict:
        return dict(self._last)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def _run(self):
        import websocket

        token = _token_mgr.get()
        if not token:
            print("[SETTRADE WS] No token available")
            return

        ws_url = f"{SETTRADE_WS_BASE}/streaming?token={token}"

        def on_open(ws):
            # Subscribe to quote stream for each symbol
            for sym in self._symbols:
                sub_msg = json.dumps({
                    "event":   "subscribe",
                    "channel": "quote",
                    "symbol":  sym,
                })
                ws.send(sub_msg)
            print(f"[SETTRADE WS] Subscribed to {len(self._symbols)} symbols")

        def on_message(ws, message):
            try:
                data = json.loads(message)
                sym  = data.get("symbol", data.get("s", ""))
                if not sym:
                    return
                quote = {
                    "price":      float(data.get("last",  data.get("l",  0))),
                    "change":     float(data.get("change",data.get("ch", 0))),
                    "pct_change": float(data.get("pct",   data.get("p",  0))),
                    "bid":        float(data.get("bid",   data.get("b",  0))),
                    "ask":        float(data.get("ask",   data.get("a",  0))),
                    "volume":     int(data.get("volume",  data.get("v",  0))),
                    "timestamp":  data.get("time", datetime.now().isoformat()),
                    "source":     "settrade_ws",
                }
                self._last[sym] = quote
                if self._callback:
                    self._callback(sym, quote)
            except Exception as e:
                print(f"[SETTRADE WS] Parse error: {e}")

        def on_error(ws, error):
            print(f"[SETTRADE WS] Error: {error}")

        def on_close(ws, code, msg):
            print(f"[SETTRADE WS] Closed: {code} {msg}")
            # Auto-reconnect ถ้ายังต้องการ
            if self._running:
                time.sleep(5)
                self._run()

        self._ws = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        self._ws.run_forever(ping_interval=30, ping_timeout=10)


# ── Streamlit helpers ──────────────────────────────────────────────────

def get_status() -> dict:
    """ตรวจสถานะการเชื่อมต่อ"""
    if not is_configured():
        return {"ok": False, "msg": "ยังไม่ได้ตั้งค่า credentials ใน .env"}
    token = _token_mgr.get()
    if not token:
        return {"ok": False, "msg": "ขอ token ไม่สำเร็จ — ตรวจสอบ APP_ID/APP_SECRET"}
    return {"ok": True, "msg": "Connected ✅", "token_preview": token[:12] + "..."}
