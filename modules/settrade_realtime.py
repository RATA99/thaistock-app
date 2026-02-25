"""
SETTRADE OpenAPI — Realtime Market Data
ใช้ REST API สำหรับ quote + MQTT สำหรับ price streaming

Authentication: App ID + App Secret (HMAC-SHA256 signature)
Endpoint: https://open-api.settrade.com  (production)
          https://open-api-test.settrade.com  (sandbox)

docs: https://developer.settrade.com/open-api
"""
import hmac
import hashlib
import base64
import time
import json
import threading
import requests
from typing import Optional, Callable, Dict
from datetime import datetime


# ── Environment ──────────────────────────────────────────────────────────
SANDBOX_BASE  = "https://open-api-test.settrade.com"
PROD_BASE     = "https://open-api.settrade.com"

SANDBOX_MQTT  = "open-api-test.settrade.com"
PROD_MQTT     = "open-api.settrade.com"
MQTT_PORT     = 1883
MQTT_PORT_TLS = 8883


def _sign(app_secret: str, timestamp: str, app_id: str, payload: str = "") -> str:
    """
    SETTRADE signature = HMAC-SHA256( app_secret, timestamp + app_id + payload )
    encoded as base64
    """
    msg = timestamp + app_id + payload
    key = app_secret.encode("utf-8")
    sig = hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(sig).decode("utf-8")


class SettradeClient:
    """
    Client สำหรับ SETTRADE OpenAPI
    - get_quote()        : ดึงราคาปัจจุบัน (REST)
    - get_ohlcv()        : ดึง intraday bars (REST)
    - subscribe_price()  : subscribe MQTT realtime tick
    - unsubscribe()      : หยุด streaming
    """

    def __init__(
        self,
        app_id: str,
        app_secret: str,
        sandbox: bool = True,
    ):
        self.app_id     = app_id
        self.app_secret = app_secret
        self.base_url   = SANDBOX_BASE if sandbox else PROD_BASE
        self.mqtt_host  = SANDBOX_MQTT if sandbox else PROD_MQTT
        self.sandbox    = sandbox
        self._mqtt_client = None
        self._subscriptions: Dict[str, Callable] = {}  # symbol -> callback
        self._connected   = False
        self._last_prices: Dict[str, dict] = {}

    # ── Auth headers ─────────────────────────────────────────────────────
    def _headers(self, payload: str = "") -> dict:
        ts  = str(int(time.time() * 1000))
        sig = _sign(self.app_secret, ts, self.app_id, payload)
        return {
            "Content-Type":   "application/json",
            "X-Api-AppId":    self.app_id,
            "X-Api-Timestamp": ts,
            "X-Api-Signature": sig,
        }

    # ── REST: Market Quote ────────────────────────────────────────────────
    def get_quote(self, symbol: str) -> Optional[dict]:
        """
        ดึง quote ปัจจุบัน สำหรับหุ้น SET
        return: {symbol, last, change, pct_change, bid, ask, volume, high, low, open}
        """
        try:
            url = f"{self.base_url}/api/set/stock/{symbol}/realtime"
            r   = requests.get(url, headers=self._headers(), timeout=10)
            if r.status_code == 200:
                d = r.json()
                return {
                    "symbol":     symbol,
                    "price":      d.get("last", 0),
                    "change":     d.get("change", 0),
                    "pct_change": d.get("percentChange", 0),
                    "bid":        d.get("bid", 0),
                    "ask":        d.get("offer", 0),
                    "volume":     d.get("volume", 0),
                    "high":       d.get("high", 0),
                    "low":        d.get("low", 0),
                    "open":       d.get("open", 0),
                    "prev_close": d.get("prior", 0),
                    "timestamp":  datetime.now().strftime("%H:%M:%S"),
                    "source":     "SETTRADE",
                }
            return None
        except Exception as e:
            return None

    def get_quotes_batch(self, symbols: list) -> dict:
        """ดึง quote หลายตัวพร้อมกัน — return {symbol: quote_dict}"""
        results = {}
        # SETTRADE supports batch endpoint
        try:
            syms_str = ",".join(symbols)
            url = f"{self.base_url}/api/set/stock/list?symbols={syms_str}"
            r   = requests.get(url, headers=self._headers(), timeout=15)
            if r.status_code == 200:
                data = r.json()
                for item in data.get("stocks", []):
                    sym = item.get("symbol", "")
                    results[sym] = {
                        "symbol":     sym,
                        "price":      item.get("last", 0),
                        "change":     item.get("change", 0),
                        "pct_change": item.get("percentChange", 0),
                        "volume":     item.get("volume", 0),
                        "high":       item.get("high", 0),
                        "low":        item.get("low", 0),
                        "open":       item.get("open", 0),
                        "timestamp":  datetime.now().strftime("%H:%M:%S"),
                        "source":     "SETTRADE",
                    }
        except Exception:
            pass
        return results

    def get_intraday_bars(self, symbol: str, interval: str = "5") -> Optional[list]:
        """
        ดึง intraday OHLCV bars
        interval: "1", "5", "15", "30", "60" (นาที)
        return: list of {time, open, high, low, close, volume}
        """
        try:
            url = f"{self.base_url}/api/set/stock/{symbol}/intraday-chart?interval={interval}"
            r   = requests.get(url, headers=self._headers(), timeout=15)
            if r.status_code == 200:
                data = r.json()
                bars = []
                for b in data.get("bars", []):
                    bars.append({
                        "time":   b.get("time", ""),
                        "open":   b.get("open", 0),
                        "high":   b.get("high", 0),
                        "low":    b.get("low", 0),
                        "close":  b.get("close", 0),
                        "volume": b.get("volume", 0),
                    })
                return bars
        except Exception:
            pass
        return None

    def get_market_status(self) -> dict:
        """ตรวจสอบสถานะตลาด"""
        try:
            url = f"{self.base_url}/api/set/market-status"
            r   = requests.get(url, headers=self._headers(), timeout=5)
            if r.status_code == 200:
                d = r.json()
                return {
                    "is_open":  d.get("market", "CLOSE") == "OPEN",
                    "status":   d.get("market", "CLOSE"),
                    "session":  d.get("session", ""),
                }
        except Exception:
            pass
        return {"is_open": False, "status": "UNKNOWN", "session": ""}

    # ── MQTT: Realtime Streaming ──────────────────────────────────────────
    def subscribe_price(self, symbols: list, callback: Callable):
        """
        Subscribe realtime price feed ผ่าน MQTT
        callback(symbol: str, data: dict) จะถูกเรียกทุกครั้งที่ราคาเปลี่ยน

        data = {symbol, last, change, pct_change, bid, ask, volume, timestamp}
        """
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            print("❌ paho-mqtt not installed. Run: pip install paho-mqtt")
            return

        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                self._connected = True
                # Subscribe to each symbol's price topic
                for sym in symbols:
                    topic = f"SET/market/stock/{sym}/price"
                    client.subscribe(topic, qos=1)
                    self._subscriptions[sym] = callback
            else:
                print(f"MQTT connect failed: rc={rc}")

        def on_message(client, userdata, msg):
            try:
                payload = json.loads(msg.payload.decode())
                # Extract symbol from topic: SET/market/stock/PTT/price
                parts  = msg.topic.split("/")
                sym    = parts[3] if len(parts) > 3 else "UNKNOWN"
                data   = {
                    "symbol":     sym,
                    "price":      payload.get("last", 0),
                    "change":     payload.get("change", 0),
                    "pct_change": payload.get("percentChange", 0),
                    "bid":        payload.get("bid", 0),
                    "ask":        payload.get("offer", 0),
                    "volume":     payload.get("volume", 0),
                    "timestamp":  datetime.now().strftime("%H:%M:%S"),
                    "source":     "SETTRADE_MQTT",
                }
                self._last_prices[sym] = data
                cb = self._subscriptions.get(sym) or self._subscriptions.get("*")
                if cb:
                    cb(sym, data)
            except Exception as e:
                pass

        def on_disconnect(client, userdata, rc):
            self._connected = False
            if rc != 0:
                # Auto-reconnect
                threading.Timer(5, lambda: client.reconnect()).start()

        # Build MQTT client with SETTRADE auth
        ts  = str(int(time.time() * 1000))
        sig = _sign(self.app_secret, ts, self.app_id, "")
        username = f"{self.app_id}:{ts}"
        password = sig

        client = mqtt.Client(client_id=f"stt_{self.app_id[:8]}_{ts[-6:]}")
        client.username_pw_set(username, password)
        client.on_connect    = on_connect
        client.on_message    = on_message
        client.on_disconnect = on_disconnect

        try:
            client.connect(self.mqtt_host, MQTT_PORT, keepalive=60)
            client.loop_start()
            self._mqtt_client = client
        except Exception as e:
            print(f"MQTT connection error: {e}")

    def unsubscribe(self):
        """หยุด MQTT streaming"""
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            self._mqtt_client = None
            self._connected   = False
            self._subscriptions.clear()

    def get_last_price(self, symbol: str) -> Optional[dict]:
        """ดึงราคาล่าสุดที่ได้รับจาก MQTT"""
        return self._last_prices.get(symbol)

    def is_connected(self) -> bool:
        return self._connected

    def test_connection(self) -> dict:
        """ทดสอบ connection และ credentials"""
        try:
            status = self.get_market_status()
            quote  = self.get_quote("PTT")
            return {
                "ok":            True,
                "market_status": status,
                "test_quote":    quote,
                "sandbox":       self.sandbox,
                "app_id":        self.app_id,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ── Singleton pattern — ใช้ร่วมกันทั้ง app ───────────────────────────────
_client: Optional[SettradeClient] = None

def get_settrade_client(app_id: str, app_secret: str, sandbox: bool = True) -> SettradeClient:
    """Get or create SETTRADE client (singleton)"""
    global _client
    if _client is None:
        _client = SettradeClient(app_id, app_secret, sandbox)
    return _client

def reset_client():
    """Reset client (เมื่อ credentials เปลี่ยน)"""
    global _client
    if _client:
        _client.unsubscribe()
    _client = None
