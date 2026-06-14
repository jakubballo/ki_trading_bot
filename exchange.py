"""
exchange.py – Kraken Futures API-Wrapper.
Demo: demo-futures.kraken.com | Live: futures.kraken.com
Charts-API (OHLCV) immer von futures.kraken.com (kein Auth nötig).
Paper-Modus: Fill-Simulation lokal (Kraken Demo füllt nicht zuverlässig).
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp

from config import config

_INSTRUMENTS_CACHE_FILE = Path("data/instruments_cache.json")
_INSTRUMENTS_CACHE_TTL  = 86400  # 24h

logger = logging.getLogger(__name__)

BASE_URL_LIVE = "https://futures.kraken.com"
BASE_URL_DEMO = "https://demo-futures.kraken.com"
CHARTS_URL   = "https://futures.kraken.com"   # Charts immer live (Public API)

# Intervall-Mapping: Code → Millisekunden (für Kerzenlängen-Berechnung)
_INTERVAL_MS = {
    "1m": 60_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
    "1d": 86_400_000, "1w": 604_800_000,
}

_symbol_filters: Dict[str, dict] = {}


class KrakenError(Exception):
    pass


def _sign(endpoint: str, post_data: str, nonce: str, api_secret: str) -> str:
    """
    Kraken Futures API Signing:
    SHA256(nonce + postdata) → concat endpoint_bytes + hash_bytes
    → HMAC-SHA512 mit base64-decodiertem Secret → base64 encode
    """
    sha256_hash = hashlib.sha256((nonce + post_data).encode("utf-8")).digest()
    message = endpoint.encode("utf-8") + sha256_hash
    mac = hmac.new(base64.b64decode(api_secret), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode("utf-8")


def _normalize_klines(candles: list, interval: str) -> list:
    """
    Konvertiert Kraken-Kerzenformat in Binance-kompatibles Format:
    [open_time_ms, open, high, low, close, volume, close_time_ms, 0, 0, 0, 0, 0]
    """
    interval_ms = _INTERVAL_MS.get(interval, 900_000)
    result = []
    for c in candles:
        t = int(c.get("time", 0))
        result.append([
            t,
            str(c.get("open", 0)),
            str(c.get("high", 0)),
            str(c.get("low", 0)),
            str(c.get("close", 0)),
            str(c.get("volume", 0)),
            t + interval_ms - 1,
            "0", "0", "0", "0", "0",
        ])
    return result


class ExchangeClient:
    """
    Asynchroner Kraken Futures REST-Client.
    Paper-Modus verwendet demo-futures.kraken.com.
    """

    def __init__(self):
        self.api_key    = os.environ.get("KRAKEN_API_KEY", "")
        self.api_secret = os.environ.get("KRAKEN_API_SECRET", "")
        self.is_paper   = config.is_paper
        self._session: Optional[aiohttp.ClientSession] = None
        self._notifier  = None

        # Paper-Trading: lokale Positions- & Order-Simulation
        self._paper_orders: Dict[str, dict] = {}   # order_id → order
        self._paper_position: Optional[dict] = None
        self._paper_balance: float = float(os.environ.get("PAPER_BALANCE", "10000"))
        self._paper_pnl: float = 0.0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _base_url(self) -> str:
        return BASE_URL_DEMO if self.is_paper else BASE_URL_LIVE

    async def _request(self, method: str, endpoint: str,
                       params: dict = None, signed: bool = False) -> Any:
        """REST-Request mit Retry und Kraken-Signing."""
        session = await self._get_session()
        url = f"{self._base_url()}{endpoint}"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        backoff = [1, 2, 4, 8, 16, 32, 60]

        for attempt, delay in enumerate(backoff + [60] * 5):
            try:
                nonce = str(int(time.time() * 1000))
                req_params = dict(params or {})

                if method.upper() == "GET":
                    query = urllib.parse.urlencode(req_params) if req_params else ""
                    if signed:
                        sig = _sign(endpoint, query, nonce, self.api_secret)
                        headers.update({"APIKey": self.api_key,
                                        "Authent": sig, "Nonce": nonce})
                    full_url = f"{url}?{query}" if query else url
                    async with session.get(
                        full_url, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        return await self._handle_response(resp, endpoint)

                else:  # POST
                    post_data = urllib.parse.urlencode(req_params)
                    if signed:
                        sig = _sign(endpoint, post_data, nonce, self.api_secret)
                        headers.update({"APIKey": self.api_key,
                                        "Authent": sig, "Nonce": nonce})
                    async with session.post(
                        url, data=post_data, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15)
                    ) as resp:
                        return await self._handle_response(resp, endpoint)

            except KrakenError:
                raise
            except asyncio.TimeoutError:
                logger.warning(f"Timeout {endpoint} – Versuch {attempt + 1}")
                if attempt < len(backoff) - 1:
                    await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Request-Fehler {endpoint}: {e} – Versuch {attempt + 1}")
                if attempt < len(backoff) - 1:
                    await asyncio.sleep(delay)
                else:
                    raise

        raise KrakenError(f"Alle Versuche fehlgeschlagen: {endpoint}")

    async def _handle_response(self, resp: aiohttp.ClientResponse, endpoint: str) -> Any:
        if resp.status == 429:
            retry = int(resp.headers.get("Retry-After", 30))
            logger.warning(f"Rate-Limit (429) – Warte {retry}s")
            await asyncio.sleep(retry)
            raise KrakenError("Rate-Limit – Retry")

        if resp.status not in (200, 201):
            text = await resp.text()
            raise KrakenError(f"HTTP {resp.status} bei {endpoint}: {text[:200]}")

        data = await resp.json(content_type=None)
        result = data.get("result", "")
        if result not in ("success", "") and "error" in data:
            raise KrakenError(f"Kraken Fehler: {data['error']}")
        return data

    # ─── Charts / OHLCV ────────────────────────────────────────────────────────

    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> list:
        """
        Lädt OHLCV-Daten von der Kraken Charts-API.
        Gibt normalisiertes Binance-kompatibles Format zurück.
        Endpoint: GET https://futures.kraken.com/api/charts/v1/mark/{symbol}/{interval}
        """
        try:
            session = await self._get_session()
            url = f"{CHARTS_URL}/api/charts/v1/mark/{symbol}/{interval}"
            params = {"count": limit}
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.error(f"Charts-API Fehler {resp.status} für {symbol} {interval}")
                    return []
                data = await resp.json(content_type=None)
                candles = data.get("candles", [])
                normalized = _normalize_klines(candles, interval)
                logger.debug(f"Kerzen geladen: {symbol} {interval} → {len(normalized)} Stück")
                return normalized
        except Exception as e:
            logger.error(f"Fehler beim Laden der Kerzen {symbol}: {e}")
            return []

    # ─── Ticker / Marktdaten ───────────────────────────────────────────────────

    async def get_ticker(self, symbol: str = None) -> dict:
        """Holt Ticker-Daten (Funding, OI, VWAP, Preise) vom Kraken Tickers-Endpoint."""
        try:
            data = await self._request("GET", "/derivatives/api/v3/tickers")
            tickers = {t["symbol"]: t for t in data.get("tickers", [])}
            if symbol:
                return tickers.get(symbol, {})
            return tickers
        except Exception as e:
            logger.error(f"Fehler beim Laden des Tickers: {e}")
            return {}

    async def get_funding_rate(self, symbol: str) -> float:
        """
        Gibt die relative Funding-Rate zurück.
        Kraken liefert fundingRate absolut in USD → teile durch markPrice.
        """
        try:
            ticker = await self.get_ticker(symbol)
            funding_abs = float(ticker.get("fundingRate", 0))
            mark_price  = float(ticker.get("markPrice", ticker.get("last", 1)) or 1)
            return funding_abs / mark_price if mark_price > 0 else 0.0
        except Exception as e:
            logger.error(f"Fehler bei Funding-Rate {symbol}: {e}")
            return 0.0

    async def get_mark_price(self, symbol: str) -> float:
        """Gibt den aktuellen Mark-Preis zurück."""
        try:
            ticker = await self.get_ticker(symbol)
            return float(ticker.get("markPrice", ticker.get("last", 0)) or 0)
        except Exception as e:
            logger.error(f"Fehler beim Mark-Preis {symbol}: {e}")
            return 0.0

    # ─── Konto ─────────────────────────────────────────────────────────────────

    async def get_account_balance(self) -> float:
        """Gibt den verfügbaren USD-Kontostand zurück."""
        if self.is_paper:
            return self._paper_balance

        try:
            data = await self._request("GET", "/derivatives/api/v3/accounts", signed=True)
            accounts = data.get("accounts", {})
            # Multi-Collateral Account
            for acc_name, acc_data in accounts.items():
                balances = acc_data.get("balances", {})
                usd = balances.get("USD", balances.get("USDT", 0))
                if usd:
                    return float(usd)
            return 0.0
        except Exception as e:
            logger.error(f"Fehler beim Kontostand: {e}")
            return 0.0

    async def sync_account_balance(self, state_ref) -> float:
        balance = await self.get_account_balance()
        state_ref.account_balance_usdt = balance
        state_ref.balance_last_synced_utc = datetime.now(timezone.utc).isoformat()
        return balance

    # ─── Positionen ────────────────────────────────────────────────────────────

    async def get_open_positions(self) -> list:
        """Gibt offene Positionen zurück (Kraken-Format → Binance-kompatibel)."""
        if self.is_paper:
            if self._paper_position:
                return [self._paper_position]
            return []
        try:
            data = await self._request("GET", "/derivatives/api/v3/openpositions", signed=True)
            positions = []
            for p in data.get("openPositions", []):
                size = float(p.get("size", 0))
                side = p.get("side", "long")
                positions.append({
                    "symbol":      p.get("symbol"),
                    "positionAmt": size if side == "long" else -size,
                    "entryPrice":  float(p.get("price", 0)),
                    "side":        "BUY" if side == "long" else "SELL",
                })
            return [p for p in positions if abs(float(p["positionAmt"])) > 0]
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Positionen: {e}")
            return []

    async def get_position_risk(self, symbol: str) -> Optional[dict]:
        """Gibt Liquidations-Info für eine Position zurück."""
        if self.is_paper:
            return {"liquidationPrice": 0}
        try:
            positions = await self.get_open_positions()
            for p in positions:
                if p.get("symbol") == symbol:
                    return {"liquidationPrice": p.get("liquidationPrice", 0)}
            return None
        except Exception as e:
            logger.error(f"Fehler bei Position-Risk {symbol}: {e}")
            return None

    # ─── Instrument-Filter ─────────────────────────────────────────────────────

    async def load_symbol_filters(self):
        """
        Lädt Mindestgrößen und Tick-Größen für alle konfigurierten Symbole.
        Liest zuerst aus Cache (data/instruments_cache.json, 24h TTL),
        dann Kraken API, dann hardcodierte Defaults.
        """
        global _symbol_filters

        # Bekannte Defaults – immer verfügbar
        defaults = {
            "PF_XBTUSD":  {"step_size": 0.001,  "min_qty": 0.001,  "tick_size": 0.5,    "min_notional": 1.0},
            "PF_ETHUSD":  {"step_size": 0.01,   "min_qty": 0.01,   "tick_size": 0.05,   "min_notional": 1.0},
            "PF_SOLUSD":  {"step_size": 1.0,    "min_qty": 1.0,    "tick_size": 0.01,   "min_notional": 1.0},
            "PF_XRPUSD":  {"step_size": 50.0,   "min_qty": 50.0,   "tick_size": 0.0001, "min_notional": 1.0},
            "PF_LINKUSD": {"step_size": 1.0,    "min_qty": 1.0,    "tick_size": 0.01,   "min_notional": 1.0},
        }

        # 1. Cache lesen (verhindert 50 simultane API-Calls beim Start)
        try:
            if _INSTRUMENTS_CACHE_FILE.exists():
                age = time.time() - _INSTRUMENTS_CACHE_FILE.stat().st_mtime
                if age < _INSTRUMENTS_CACHE_TTL:
                    cached = json.loads(_INSTRUMENTS_CACHE_FILE.read_text())
                    for symbol in config.symbols:
                        if symbol in cached:
                            _symbol_filters[symbol] = cached[symbol]
                    if all(s in _symbol_filters for s in config.symbols):
                        logger.info(f"Instrument-Filter aus Cache geladen (Alter: {age/3600:.1f}h)")
                        return
        except Exception as e:
            logger.debug(f"Cache-Lesen fehlgeschlagen: {e}")

        # 2. Kraken API
        try:
            data = await self._request("GET", "/derivatives/api/v3/instruments")
            instruments = {i["symbol"]: i for i in data.get("instruments", [])}

            for symbol in config.symbols:
                inst = instruments.get(symbol, {})
                fallback = defaults.get(symbol, {"step_size": 0.01, "min_qty": 0.01,
                                                  "tick_size": 0.01, "min_notional": 1.0})
                _symbol_filters[symbol] = {
                    "step_size":    float(inst.get("contractSize", fallback["step_size"])),
                    "min_qty":      fallback["min_qty"],
                    "max_qty":      float(inst.get("maxPositionSize", 1_000_000)),
                    "tick_size":    float(inst.get("tickSize", fallback["tick_size"])),
                    "min_notional": fallback["min_notional"],
                }
                logger.info(f"Filter geladen: {symbol} → {_symbol_filters[symbol]}")

            # Cache schreiben für alle anderen Bots
            try:
                _INSTRUMENTS_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                _INSTRUMENTS_CACHE_FILE.write_text(json.dumps(_symbol_filters, indent=2))
                logger.info("Instrument-Filter in Cache gespeichert")
            except Exception as e:
                logger.debug(f"Cache-Schreiben fehlgeschlagen: {e}")

        except Exception as e:
            logger.warning(f"Instrument-Filter API nicht erreichbar ({e}) – nutze Hardcode-Defaults")
            # 3. Hardcodierte Defaults
            for symbol in config.symbols:
                if symbol not in _symbol_filters:
                    fb = defaults.get(symbol, {"step_size": 0.01, "min_qty": 0.01,
                                               "tick_size": 0.01, "min_notional": 1.0})
                    _symbol_filters[symbol] = {**fb, "max_qty": 1_000_000}

    def get_symbol_filters(self, symbol: str) -> Optional[dict]:
        return _symbol_filters.get(symbol)

    # ─── Orders ────────────────────────────────────────────────────────────────

    async def place_limit_order(self, symbol: str, side: str, qty: float,
                                price: float) -> Optional[dict]:
        """Platziert eine Limit-Entry-Order."""
        kraken_side = "buy" if side.upper() == "BUY" else "sell"

        if self.is_paper:
            order_id = f"paper_{int(time.time() * 1000)}"
            self._paper_orders[order_id] = {
                "orderId": order_id, "symbol": symbol, "side": side.upper(),
                "type": "LIMIT", "price": price, "qty": qty,
                "status": "NEW", "created_at": time.time(),
            }
            logger.info(f"[PAPER] Limit-Order: {side} {qty} {symbol} @ {price}")
            return {"orderId": order_id, "status": "placed"}

        try:
            data = await self._request("POST", "/derivatives/api/v3/sendorder", {
                "orderType": "lmt",
                "symbol": symbol,
                "side": kraken_side,
                "size": qty,
                "limitPrice": price,
            }, signed=True)
            send_status = data.get("sendStatus", {})
            order_id = send_status.get("orderId", "")
            logger.info(f"Limit-Order: {side} {qty} {symbol} @ {price} (ID: {order_id})")
            return {"orderId": order_id, "status": send_status.get("status", "")}
        except Exception as e:
            logger.error(f"Fehler bei Limit-Order {symbol}: {e}")
            return None

    async def place_stop_market(self, symbol: str, side: str,
                                stop_price: float, close_position: bool = True) -> Optional[dict]:
        """Platziert eine Stop-Market-Order (SL)."""
        kraken_side = "buy" if side.upper() == "BUY" else "sell"
        qty = self._paper_get_position_qty(symbol) if self.is_paper else 0

        if self.is_paper:
            order_id = f"paper_sl_{int(time.time() * 1000)}"
            self._paper_orders[order_id] = {
                "orderId": order_id, "symbol": symbol, "side": side.upper(),
                "type": "STOP_MARKET", "stop_price": stop_price, "qty": qty,
                "status": "NEW",
            }
            logger.info(f"[PAPER] SL-Order: {side} {symbol} @ {stop_price}")
            return {"orderId": order_id}

        try:
            params: dict = {
                "orderType": "stp",
                "symbol": symbol,
                "side": kraken_side,
                "stopPrice": stop_price,
                "triggerSignal": "mark",
            }
            if not close_position:
                params["size"] = qty
            data = await self._request("POST", "/derivatives/api/v3/sendorder", params, signed=True)
            send_status = data.get("sendStatus", {})
            order_id = send_status.get("orderId", "")
            logger.info(f"SL gesetzt: {symbol} @ {stop_price} (ID: {order_id})")
            return {"orderId": order_id}
        except Exception as e:
            logger.error(f"Fehler beim SL-Setzen {symbol}: {e}")
            return None

    async def place_take_profit_market(self, symbol: str, side: str,
                                       stop_price: float,
                                       close_position: bool = True) -> Optional[dict]:
        """Platziert eine Take-Profit-Order."""
        kraken_side = "buy" if side.upper() == "BUY" else "sell"
        qty = self._paper_get_position_qty(symbol) if self.is_paper else 0

        if self.is_paper:
            order_id = f"paper_tp_{int(time.time() * 1000)}"
            self._paper_orders[order_id] = {
                "orderId": order_id, "symbol": symbol, "side": side.upper(),
                "type": "TAKE_PROFIT_MARKET", "stop_price": stop_price, "qty": qty,
                "status": "NEW",
            }
            logger.info(f"[PAPER] TP-Order: {side} {symbol} @ {stop_price}")
            return {"orderId": order_id}

        try:
            params: dict = {
                "orderType": "take_profit",
                "symbol": symbol,
                "side": kraken_side,
                "stopPrice": stop_price,
                "triggerSignal": "mark",
            }
            if not close_position:
                params["size"] = qty
            data = await self._request("POST", "/derivatives/api/v3/sendorder", params, signed=True)
            send_status = data.get("sendStatus", {})
            order_id = send_status.get("orderId", "")
            logger.info(f"TP gesetzt: {symbol} @ {stop_price} (ID: {order_id})")
            return {"orderId": order_id}
        except Exception as e:
            logger.error(f"Fehler beim TP-Setzen {symbol}: {e}")
            return None

    async def place_market_order(self, symbol: str, side: str, qty: float) -> Optional[dict]:
        """Market-Order (nur Emergency-Close)."""
        kraken_side = "buy" if side.upper() == "BUY" else "sell"

        if self.is_paper:
            logger.info(f"[PAPER] Market-Order (Emergency): {side} {qty} {symbol}")
            self._paper_close_position(symbol)
            return {"orderId": f"paper_mkt_{int(time.time() * 1000)}"}

        try:
            data = await self._request("POST", "/derivatives/api/v3/sendorder", {
                "orderType": "mkt",
                "symbol": symbol,
                "side": kraken_side,
                "size": qty,
            }, signed=True)
            return data.get("sendStatus", {})
        except Exception as e:
            logger.error(f"Fehler bei Market-Order {symbol}: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id) -> bool:
        """Cancelt eine einzelne Order."""
        if self.is_paper:
            self._paper_orders.pop(str(order_id), None)
            return True
        try:
            await self._request("POST", "/derivatives/api/v3/cancelorder",
                                {"order_id": str(order_id)}, signed=True)
            logger.info(f"Order gecancelt: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Canceln {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancelt alle offenen Orders für ein Symbol."""
        if self.is_paper:
            to_remove = [oid for oid, o in self._paper_orders.items()
                         if o.get("symbol") == symbol]
            for oid in to_remove:
                del self._paper_orders[oid]
            return True
        try:
            await self._request("POST", "/derivatives/api/v3/cancelallorders",
                                {"symbol": symbol}, signed=True)
            logger.info(f"Alle Orders gecancelt: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Canceln aller Orders {symbol}: {e}")
            return False

    async def get_open_orders(self, symbol: str = None) -> list:
        """Gibt offene Orders zurück."""
        if self.is_paper:
            orders = list(self._paper_orders.values())
            if symbol:
                orders = [o for o in orders if o.get("symbol") == symbol]
            return [{"orderId": o["orderId"], "symbol": o.get("symbol")} for o in orders]

        try:
            data = await self._request("GET", "/derivatives/api/v3/openorders", signed=True)
            orders = []
            for o in data.get("openOrders", []):
                if symbol and o.get("symbol") != symbol:
                    continue
                orders.append({
                    "orderId": o.get("order_id", o.get("orderId")),
                    "symbol": o.get("symbol"),
                    "side": o.get("side", "").upper(),
                    "type": o.get("orderType"),
                })
            return orders
        except Exception as e:
            logger.error(f"Fehler beim Abrufen offener Orders: {e}")
            return []

    async def get_order_status(self, symbol: str, order_id) -> Optional[dict]:
        """Holt den Status einer einzelnen Order."""
        if self.is_paper:
            o = self._paper_orders.get(str(order_id))
            if o:
                return {"status": o["status"], "orderId": order_id}
            return {"status": "FILLED", "orderId": order_id}

        try:
            data = await self._request("GET", "/derivatives/api/v3/orders/status",
                                       {"orderIds": str(order_id)}, signed=True)
            orders = data.get("orders", [])
            if orders:
                o = orders[0]
                return {
                    "status": "FILLED" if o.get("status") == "filled" else o.get("status", ""),
                    "orderId": order_id,
                    "avgPrice": float(o.get("avgPrice", o.get("limitPrice", 0))),
                    "executedQty": float(o.get("filledSize", 0)),
                }
            return None
        except Exception as e:
            logger.error(f"Fehler beim Order-Status {order_id}: {e}")
            return None

    async def check_orphan_orders(self):
        """Cancelt verwaiste Orders ohne zugehörige Position."""
        try:
            from state import state as bot_state
            open_orders = await self.get_open_orders()
            pos = bot_state.open_position

            for order in open_orders:
                order_id = str(order.get("orderId", ""))
                symbol = order.get("symbol")
                is_known = (
                    pos.symbol == symbol and order_id in {
                        str(pos.sl_order_id), str(pos.tp_order_id), str(pos.entry_order_id)
                    }
                )
                if not is_known:
                    logger.warning(f"Verwaiste Order: {order_id} ({symbol}) – Canceln")
                    await self.cancel_order(symbol, order_id)
        except Exception as e:
            logger.error(f"Fehler beim Orphan-Check: {e}")

    # ─── Leverage / Margin ──────────────────────────────────────────────────────

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED"):
        """Kraken Futures verwendet immer Cross-Margin (kein ISOLATED). Ignoriert."""
        logger.debug(f"set_margin_type ignoriert (Kraken: immer Cross): {symbol}")

    async def set_leverage(self, symbol: str, leverage: int):
        """Kraken Futures: Hebel wird pro Order implizit durch Margin gesteuert."""
        logger.debug(f"set_leverage: {symbol} → {leverage}x (wird bei Order-Größe berücksichtigt)")

    # ─── Paper-Trading Hilfsmethoden ────────────────────────────────────────────

    def _paper_get_position_qty(self, symbol: str) -> float:
        if self._paper_position and self._paper_position.get("symbol") == symbol:
            return abs(float(self._paper_position.get("positionAmt", 0)))
        return 0.0

    def _paper_close_position(self, symbol: str):
        if self._paper_position and self._paper_position.get("symbol") == symbol:
            self._paper_position = None
            # Orders für dieses Symbol entfernen
            to_remove = [oid for oid, o in self._paper_orders.items()
                         if o.get("symbol") == symbol]
            for oid in to_remove:
                del self._paper_orders[oid]

    def paper_simulate_fill(self, order_id: str, fill_price: float) -> Optional[dict]:
        """
        Simuliert einen Order-Fill für Paper-Trading.
        Wird vom WebSocket-Manager aufgerufen wenn Mark-Preis die Order-Grenze kreuzt.
        """
        order = self._paper_orders.get(str(order_id))
        if not order:
            return None

        order["status"] = "FILLED"
        logger.info(f"[PAPER-FILL] {order['side']} {order.get('qty', '?')} "
                    f"{order['symbol']} @ {fill_price:.4f}")
        return {
            "symbol": order["symbol"],
            "side": order["side"],
            "qty": order.get("qty", 0),
            "price": fill_price,
            "order_id": order_id,
        }

    async def emergency_close(self, symbol: str, qty: float, side: str) -> bool:
        logger.critical(f"EMERGENCY CLOSE: {side} {qty} {symbol}")
        try:
            await self.cancel_all_orders(symbol)
            result = await self.place_market_order(symbol, side, qty)
            return result is not None
        except Exception as e:
            logger.critical(f"Emergency-Close fehlgeschlagen: {e}")
            return False

    # ─── Listen-Key (Binance-Kompatibilität – für Kraken nicht benötigt) ────────

    async def create_listen_key(self) -> Optional[str]:
        return None  # Kraken: kein Listen-Key nötig

    async def renew_listen_key(self, listen_key: str) -> bool:
        return True


exchange = ExchangeClient()
