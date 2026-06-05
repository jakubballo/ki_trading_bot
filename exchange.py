"""
exchange.py – Binance Futures API-Wrapper.
Kapselt alle REST-Calls an die Binance Futures API mit Rate-Limit-Schutz.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import aiohttp
from config import config

logger = logging.getLogger(__name__)

# Binance Futures Base-URLs
BASE_URL_LIVE = "https://fapi.binance.com"
BASE_URL_DEMO = "https://demo-fapi.binance.com"

# Rate-Limit-Schwelle (von 2400 gesamt)
RATE_LIMIT_PAUSE_THRESHOLD = 2000
RATE_LIMIT_PAUSE_SECONDS = 30

# Symbol-Filter Cache
_symbol_filters: Dict[str, dict] = {}

# IP-Ban Flag
_ip_banned: bool = False


class BinanceError(Exception):
    """Binance API Fehler."""
    pass


class ExchangeClient:
    """
    Asynchroner Binance Futures REST-Client.
    Verwaltet Rate-Limits, Fehlerbehandlung und Paper-Trading-Modus.
    """

    def __init__(self):
        self.api_key = os.environ.get("BINANCE_API_KEY", "")
        self.api_secret = os.environ.get("BINANCE_SECRET", "")
        self.is_paper = config.is_paper
        self._session: Optional[aiohttp.ClientSession] = None
        self._used_weight: int = 0
        self._notifier = None  # Wird in main.py gesetzt

    async def _get_session(self) -> aiohttp.ClientSession:
        """Gibt die aktive Session zurück, erstellt eine neue falls nötig."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "X-MBX-APIKEY": self.api_key,
                    "Content-Type": "application/json",
                }
            )
        return self._session

    async def close(self):
        """Schließt die HTTP-Session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _sign_params(self, params: dict) -> dict:
        """Signiert die Request-Parameter mit HMAC-SHA256."""
        import hashlib
        import hmac
        import urllib.parse

        params["timestamp"] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    async def _request(self, method: str, endpoint: str, params: dict = None,
                       signed: bool = False) -> Any:
        """
        Führt einen REST-Call durch mit:
        - Rate-Limit-Überwachung
        - HTTP 429/418 Behandlung
        - Exponential Backoff bei Fehlern
        """
        global _ip_banned

        if _ip_banned:
            raise BinanceError("IP ist gesperrt (HTTP 418) – Bot gestoppt")

        params = params or {}
        if signed:
            params = self._sign_params(params)

        base_url = BASE_URL_DEMO if self.is_paper else BASE_URL_LIVE
        url = f"{base_url}{endpoint}"
        session = await self._get_session()

        backoff_delays = [1, 2, 4, 8, 16, 32, 60]

        for attempt, delay in enumerate(backoff_delays + [60] * 10):
            try:
                async with session.request(method, url, params=params,
                                           timeout=aiohttp.ClientTimeout(total=15)) as resp:

                    # Rate-Limit-Header auslesen
                    used_weight = resp.headers.get("X-MBX-USED-WEIGHT-1M")
                    if used_weight:
                        self._used_weight = int(used_weight)
                        if self._used_weight > RATE_LIMIT_PAUSE_THRESHOLD:
                            logger.warning(f"Rate-Limit nahe: {self._used_weight}/2400 – Pausiere {RATE_LIMIT_PAUSE_SECONDS}s")
                            if self._notifier:
                                self._notifier.send_warning(
                                    f"⚡ Rate-Limit: {self._used_weight}/2400 – Pausiere 30s"
                                )
                            await asyncio.sleep(RATE_LIMIT_PAUSE_SECONDS)

                    # IP-Ban
                    if resp.status == 418:
                        _ip_banned = True
                        msg = "IP wurde von Binance gesperrt (HTTP 418)! Bot stoppt."
                        logger.critical(msg)
                        if self._notifier:
                            await self._notifier.send_critical(msg)
                        raise BinanceError(msg)

                    # Rate-Limit überschritten
                    if resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", delay))
                        logger.warning(f"HTTP 429 – Warte {retry_after}s (Versuch {attempt + 1})")
                        await asyncio.sleep(retry_after)
                        continue

                    # Erfolg
                    if resp.status == 200:
                        return await resp.json()

                    # API-Fehler
                    error_data = await resp.json()
                    error_code = error_data.get("code", 0)
                    error_msg = error_data.get("msg", "Unbekannter Fehler")

                    # -4046: Margin-Typ bereits gesetzt (ignorieren)
                    if error_code == -4046:
                        return {"code": -4046, "msg": "Already set"}

                    logger.error(f"Binance API Fehler {error_code}: {error_msg} (Endpoint: {endpoint})")
                    raise BinanceError(f"API Fehler {error_code}: {error_msg}")

            except BinanceError:
                raise
            except asyncio.TimeoutError:
                logger.warning(f"Timeout bei {endpoint} – Versuch {attempt + 1}")
                if attempt < len(backoff_delays) - 1:
                    await asyncio.sleep(delay)
            except Exception as e:
                logger.error(f"Request-Fehler bei {endpoint}: {e} – Versuch {attempt + 1}")
                if attempt < len(backoff_delays) - 1:
                    await asyncio.sleep(delay)
                else:
                    raise

        raise BinanceError(f"Alle Versuche fehlgeschlagen für {endpoint}")

    # ─── Konto & Infrastruktur ──────────────────────────────────────────────

    async def get_account_balance(self) -> float:
        """Holt den aktuellen USDT-Kontostand."""
        try:
            data = await self._request("GET", "/fapi/v2/balance", signed=True)
            for asset in data:
                if asset.get("asset") == "USDT":
                    balance = float(asset.get("availableBalance", 0))
                    logger.info(f"Kontostand: {balance:.2f} USDT")
                    return balance
            return 0.0
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Kontostands: {e}")
            return 0.0

    async def sync_account_balance(self, state_ref) -> float:
        """Synchronisiert den Kontostand in den State."""
        balance = await self.get_account_balance()
        state_ref.account_balance_usdt = balance
        state_ref.balance_last_synced_utc = datetime.now(timezone.utc).isoformat()
        return balance

    async def get_open_positions(self) -> list:
        """Gibt alle offenen Positionen zurück."""
        try:
            data = await self._request("GET", "/fapi/v2/positionRisk", signed=True)
            # Nur Positionen mit tatsächlicher Menge zurückgeben
            open_positions = [p for p in data if abs(float(p.get("positionAmt", 0))) > 0]
            return open_positions
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Positionen: {e}")
            return []

    async def get_position_risk(self, symbol: str) -> Optional[dict]:
        """Gibt das Liquidations-Risiko einer Position zurück."""
        try:
            data = await self._request("GET", "/fapi/v2/positionRisk",
                                       params={"symbol": symbol}, signed=True)
            if data:
                return data[0]
            return None
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Position-Risks: {e}")
            return None

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED"):
        """Setzt den Margin-Typ. Fehler -4046 (bereits gesetzt) wird ignoriert."""
        try:
            result = await self._request("POST", "/fapi/v1/marginType",
                                         params={"symbol": symbol, "marginType": margin_type},
                                         signed=True)
            # -4046 bedeutet: bereits gesetzt – kein Fehler
            if isinstance(result, dict) and result.get("code") == -4046:
                logger.debug(f"Margin-Typ für {symbol} bereits auf {margin_type}")
            else:
                logger.info(f"Margin-Typ gesetzt: {symbol} → {margin_type}")
        except BinanceError as e:
            if "-4046" in str(e):
                logger.debug(f"Margin-Typ {margin_type} bereits gesetzt für {symbol}")
            else:
                logger.error(f"Fehler beim Setzen des Margin-Typs: {e}")

    async def set_leverage(self, symbol: str, leverage: int = 3):
        """Setzt den Hebel für ein Symbol."""
        try:
            await self._request("POST", "/fapi/v1/leverage",
                                 params={"symbol": symbol, "leverage": leverage},
                                 signed=True)
            logger.info(f"Hebel gesetzt: {symbol} → {leverage}x")
        except Exception as e:
            logger.error(f"Fehler beim Setzen des Hebels: {e}")

    async def load_symbol_filters(self):
        """Lädt LOT_SIZE stepSize, minQty und tickSize für alle konfigurierten Symbole."""
        global _symbol_filters
        try:
            data = await self._request("GET", "/fapi/v1/exchangeInfo")
            symbols_info = {s["symbol"]: s for s in data.get("symbols", [])}

            for symbol in config.symbols:
                if symbol not in symbols_info:
                    logger.warning(f"Symbol nicht gefunden in Exchange-Info: {symbol}")
                    continue

                filters = symbols_info[symbol].get("filters", [])
                filter_map = {f["filterType"]: f for f in filters}

                lot_size = filter_map.get("LOT_SIZE", {})
                price_filter = filter_map.get("PRICE_FILTER", {})
                min_notional = filter_map.get("MIN_NOTIONAL", {})

                _symbol_filters[symbol] = {
                    "step_size": float(lot_size.get("stepSize", 0.001)),
                    "min_qty": float(lot_size.get("minQty", 0.001)),
                    "max_qty": float(lot_size.get("maxQty", 1000)),
                    "tick_size": float(price_filter.get("tickSize", 0.01)),
                    "min_notional": float(min_notional.get("notional", 5.0)),
                }
                logger.info(f"Filter geladen für {symbol}: {_symbol_filters[symbol]}")

        except Exception as e:
            logger.error(f"Fehler beim Laden der Symbol-Filter: {e}")

    def get_symbol_filters(self, symbol: str) -> Optional[dict]:
        """Gibt die gecachten Symbol-Filter zurück."""
        return _symbol_filters.get(symbol)

    async def get_open_orders(self, symbol: str = None) -> list:
        """Gibt alle offenen Orders zurück."""
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol
            return await self._request("GET", "/fapi/v1/openOrders", params=params, signed=True)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen offener Orders: {e}")
            return []

    async def check_orphan_orders(self):
        """
        Cancelt verwaiste Orders aus dem vorherigen Bot-Lauf.
        Verwaist = Orders ohne zugehörige offene Position im aktuellen State.
        """
        try:
            from state import state as bot_state
            open_orders = await self.get_open_orders()

            for order in open_orders:
                symbol = order.get("symbol")
                order_id = order.get("orderId")

                # Prüfen ob Order zur aktuellen Position gehört
                pos = bot_state.open_position
                is_known = (
                    pos.symbol == symbol and (
                        str(order_id) == str(pos.sl_order_id) or
                        str(order_id) == str(pos.tp_order_id) or
                        str(order_id) == str(pos.entry_order_id)
                    )
                )

                if not is_known:
                    logger.warning(f"Verwaiste Order gefunden: {order_id} ({symbol}) – Canceln")
                    await self.cancel_order(symbol, order_id)

        except Exception as e:
            logger.error(f"Fehler beim Prüfen verwaister Orders: {e}")

    async def cancel_order(self, symbol: str, order_id: int) -> bool:
        """Cancelt eine Order."""
        try:
            await self._request("DELETE", "/fapi/v1/order",
                                 params={"symbol": symbol, "orderId": order_id},
                                 signed=True)
            logger.info(f"Order gecancelt: {order_id} ({symbol})")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Canceln der Order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: str) -> bool:
        """Cancelt alle offenen Orders für ein Symbol."""
        try:
            await self._request("DELETE", "/fapi/v1/allOpenOrders",
                                 params={"symbol": symbol}, signed=True)
            logger.info(f"Alle Orders gecancelt für {symbol}")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Canceln aller Orders für {symbol}: {e}")
            return False

    async def place_limit_order(self, symbol: str, side: str, qty: float,
                                price: float) -> Optional[dict]:
        """
        Platziert eine Limit-Order (GTC).
        """
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": qty,
                "price": price,
                "positionSide": "BOTH",
                "selfTradePreventionMode": "EXPIRE_BOTH",
            }
            result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)
            logger.info(f"Limit-Order platziert: {side} {qty} {symbol} @ {price} (ID: {result.get('orderId')})")
            return result
        except Exception as e:
            logger.error(f"Fehler beim Platzieren der Limit-Order: {e}")
            return None

    async def place_stop_market(self, symbol: str, side: str, stop_price: float,
                                close_position: bool = True) -> Optional[dict]:
        """Platziert eine Stop-Market-Order (SL)."""
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": "STOP_MARKET",
                "workingType": "MARK_PRICE",
                "closePosition": "true" if close_position else "false",
                "stopPrice": stop_price,
                "priceProtect": "true",
                "positionSide": "BOTH",
            }
            result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)
            logger.info(f"SL gesetzt: {symbol} @ {stop_price} (ID: {result.get('orderId')})")
            return result
        except Exception as e:
            logger.error(f"Fehler beim Setzen des SL: {e}")
            return None

    async def place_take_profit_market(self, symbol: str, side: str,
                                       stop_price: float,
                                       close_position: bool = True) -> Optional[dict]:
        """Platziert eine Take-Profit-Market-Order (TP)."""
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": "TAKE_PROFIT_MARKET",
                "workingType": "MARK_PRICE",
                "closePosition": "true" if close_position else "false",
                "stopPrice": stop_price,
                "positionSide": "BOTH",
            }
            result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)
            logger.info(f"TP gesetzt: {symbol} @ {stop_price} (ID: {result.get('orderId')})")
            return result
        except Exception as e:
            logger.error(f"Fehler beim Setzen des TP: {e}")
            return None

    async def place_market_order(self, symbol: str, side: str, qty: float) -> Optional[dict]:
        """
        Platziert eine Market-Order.
        NUR für Emergency-Close erlaubt – niemals als normaler Entry!
        """
        try:
            params = {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": qty,
                "positionSide": "BOTH",
            }
            result = await self._request("POST", "/fapi/v1/order", params=params, signed=True)
            logger.info(f"Market-Order platziert: {side} {qty} {symbol}")
            return result
        except Exception as e:
            logger.error(f"Fehler bei Emergency Market-Order: {e}")
            return None

    async def emergency_close(self, symbol: str, qty: float, side: str) -> bool:
        """
        Notfall-Schließung einer Position via Market-Order.
        side: Gegenrichtung zur offenen Position (LONG → SELL, SHORT → BUY)
        """
        logger.critical(f"EMERGENCY CLOSE: {side} {qty} {symbol}")
        try:
            await self.cancel_all_orders(symbol)
            close_side = "SELL" if side == "BUY" else "BUY"
            result = await self.place_market_order(symbol, close_side, qty)
            return result is not None
        except Exception as e:
            logger.critical(f"Emergency-Close fehlgeschlagen: {e}")
            return False

    async def get_order_status(self, symbol: str, order_id: int) -> Optional[dict]:
        """Holt den Status einer Order."""
        try:
            return await self._request("GET", "/fapi/v1/order",
                                       params={"symbol": symbol, "orderId": order_id},
                                       signed=True)
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Order-Status: {e}")
            return None

    async def get_funding_rate(self, symbol: str) -> float:
        """Holt die aktuelle Funding-Rate für ein Symbol."""
        try:
            data = await self._request("GET", "/fapi/v1/premiumIndex",
                                       params={"symbol": symbol})
            if isinstance(data, list):
                data = data[0]
            return float(data.get("lastFundingRate", 0))
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Funding-Rate: {e}")
            return 0.0

    async def get_mark_price(self, symbol: str) -> float:
        """Holt den aktuellen Mark-Preis."""
        try:
            data = await self._request("GET", "/fapi/v1/premiumIndex",
                                       params={"symbol": symbol})
            if isinstance(data, list):
                data = data[0]
            return float(data.get("markPrice", 0))
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Mark-Preises: {e}")
            return 0.0

    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> list:
        """Holt historische Kerzendaten."""
        try:
            data = await self._request("GET", "/fapi/v1/klines",
                                       params={"symbol": symbol, "interval": interval, "limit": limit})
            return data or []
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Klines: {e}")
            return []

    async def create_listen_key(self) -> Optional[str]:
        """Erstellt einen Listen-Key für User-Data-Stream."""
        try:
            data = await self._request("POST", "/fapi/v1/listenKey", signed=False)
            return data.get("listenKey")
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Listen-Keys: {e}")
            return None

    async def renew_listen_key(self, listen_key: str) -> bool:
        """Erneuert einen Listen-Key (alle 30 Minuten nötig)."""
        try:
            await self._request("PUT", "/fapi/v1/listenKey",
                                 params={"listenKey": listen_key})
            return True
        except Exception as e:
            logger.error(f"Fehler beim Erneuern des Listen-Keys: {e}")
            return False


# Globale Exchange-Instanz
exchange = ExchangeClient()
