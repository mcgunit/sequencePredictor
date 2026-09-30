"""
Market data layer for roadmap item 4 ("Same models for crypto and shares -
a predictor, not a trading bot"): the SQLite store, the daily-bar sources,
the incremental update and the integrity checks. Phase M1 of that item.

The store is one SQLite file, data/markets.sqlite3 (README decision), with
the tables the design names: instruments, bars, predictions, results,
model_performance. It is gitignored like the Optuna db.sqlite3 - bars are
re-fetchable, and a binary file that changes daily does not belong in a
repository whose daily commit is `git add -A`; the tracked record of the
market predictions will be text exports (phase M2), the way the lottery
games keep yearly CSVs.

Sources (probed from the production box on 2026-09-30):
  binance        GET /api/v3/klines, 1d, no key - works; paginated by 1000;
                 the running day's candle is returned open and is skipped
                 until it has closed (close time in the past).
  nasdaq         api.nasdaq.com/api/quote/<symbol>/historical, no key, with
                 browser-like headers - works; ten years of sessions in one
                 answer (2,513 rows for NVDA and ASML), prices as "$227.21",
                 volumes with thousands separators. The shares source in use.
  yahoo          the v8 chart endpoint behind Yahoo's cookie + crumb dance -
                 answered 429 to every request from this address that day;
                 implemented and fixture-tested, an alternate per instrument.
  alphavantage   TIME_SERIES_DAILY with a free key (ALPHAVANTAGE_KEY in the
                 environment; 25 requests a day is plenty for four tickers).
  stooq          the design's first choice for shares now serves a JavaScript
                 browser check instead of the CSV - not usable from a script.
Nasdaq and Yahoo serve the WHOLE history split-adjusted to the day of the
call (NVDA's 2024 bars are stored at a tenth of their traded price, AAPL's
2020 bars at a quarter), so a three-day incremental window would splice two
price scales at the next split. For those sources every update refetches the
full answer (one call, about 2,500 rows) and rewrites the history in place;
a uniform rescale of the overlapping bars is detected, logged and, on an
as-traded source, escalated to a full refetch; after a rescale the bars older
than the source's window are put on the new basis with the same factor, so a
uniform factor never moves a log return, a bin or the store's first bar.
A source is operational, not identity: set_source() changes where an
instrument's bars come from without touching the frozen universe - and
clears the stored bars, because the bases differ (Alpha Vantage is
as-traded, Nasdaq and Yahoo adjusted) and cannot be spliced.

The universe is FROZEN (README: re-selecting "the top 5" every day would
smuggle survivorship and look-ahead into the results): install_universe adds
what is missing and never changes an existing row.

Self-check: python3 -m src.MarketData (part of npm test; no network).
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
import http.client
from http.cookiejar import CookieJar

SCHEMA_VERSION = 1
DEFAULT_DB = os.path.join("data", "markets.sqlite3")
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) sequencePredictor/1.0 (research; daily bars)"
KICKOFF = "2026-09-30"
MARKETS = ("crypto", "shares")

# The frozen universe at kickoff. Crypto: the five largest coins by market
# cap on CoinGecko on 2026-09-30 that are not stablecoins (Tether at #3 and
# USDC at #6 are pegged to the quote currency and have no return to predict),
# quoted in USDT on Binance - BTC and ETH since 2017, BNB 2017, XRP 2018, SOL
# since 2020-08-11, which bounds the joint history. Shares: four liquid
# US-listed names (ASML's Nasdaq listing rather than Euronext, so the four
# share one calendar and one currency). position = the slot in the positional
# game (instrument = position, return bin = digit).
DEFAULT_UNIVERSE = [
    {"market": "crypto", "symbol": "BTC", "name": "Bitcoin", "source": "binance", "source_symbol": "BTCUSDT", "quote": "USDT", "position": 0, "added_on": KICKOFF},
    {"market": "crypto", "symbol": "ETH", "name": "Ethereum", "source": "binance", "source_symbol": "ETHUSDT", "quote": "USDT", "position": 1, "added_on": KICKOFF},
    {"market": "crypto", "symbol": "BNB", "name": "BNB", "source": "binance", "source_symbol": "BNBUSDT", "quote": "USDT", "position": 2, "added_on": KICKOFF},
    {"market": "crypto", "symbol": "XRP", "name": "XRP", "source": "binance", "source_symbol": "XRPUSDT", "quote": "USDT", "position": 3, "added_on": KICKOFF},
    {"market": "crypto", "symbol": "SOL", "name": "Solana", "source": "binance", "source_symbol": "SOLUSDT", "quote": "USDT", "position": 4, "added_on": KICKOFF},
    {"market": "shares", "symbol": "NVDA", "name": "NVIDIA", "source": "nasdaq", "source_symbol": "NVDA", "quote": "USD", "position": 0, "added_on": KICKOFF},
    {"market": "shares", "symbol": "AAPL", "name": "Apple", "source": "nasdaq", "source_symbol": "AAPL", "quote": "USD", "position": 1, "added_on": KICKOFF},
    {"market": "shares", "symbol": "MSFT", "name": "Microsoft", "source": "nasdaq", "source_symbol": "MSFT", "quote": "USD", "position": 2, "added_on": KICKOFF},
    {"market": "shares", "symbol": "ASML", "name": "ASML (Nasdaq listing)", "source": "nasdaq", "source_symbol": "ASML", "quote": "USD", "position": 3, "added_on": KICKOFF},
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS instruments (
    id            INTEGER PRIMARY KEY,
    market        TEXT    NOT NULL,
    symbol        TEXT    NOT NULL,
    name          TEXT,
    source        TEXT    NOT NULL,
    source_symbol TEXT    NOT NULL,
    quote         TEXT,
    position      INTEGER NOT NULL,
    added_on      TEXT    NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1,
    note          TEXT,
    UNIQUE (market, symbol),
    UNIQUE (market, position)
);
CREATE TABLE IF NOT EXISTS bars (
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    date          TEXT    NOT NULL,
    open          REAL    NOT NULL,
    high          REAL    NOT NULL,
    low           REAL    NOT NULL,
    close         REAL    NOT NULL,
    volume        REAL,
    fetched_at    TEXT    NOT NULL,
    PRIMARY KEY (instrument_id, date)
);
CREATE TABLE IF NOT EXISTS predictions (
    model         TEXT    NOT NULL,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    for_date      TEXT    NOT NULL,
    made_at       TEXT    NOT NULL,
    bin           INTEGER,
    direction     INTEGER,
    confidence    REAL,
    probabilities TEXT,
    meta          TEXT,
    PRIMARY KEY (model, instrument_id, for_date)
);
CREATE TABLE IF NOT EXISTS results (
    model         TEXT    NOT NULL,
    instrument_id INTEGER NOT NULL REFERENCES instruments(id),
    for_date      TEXT    NOT NULL,
    settled_at    TEXT    NOT NULL,
    actual_bin    INTEGER,
    actual_return REAL,
    hit_exact     INTEGER,
    hit_adjacent  INTEGER,
    hit_direction INTEGER,
    pnl           REAL,
    PRIMARY KEY (model, instrument_id, for_date)
);
CREATE TABLE IF NOT EXISTS model_performance (
    model       TEXT NOT NULL,
    market      TEXT NOT NULL,
    computed_at TEXT NOT NULL,
    days        INTEGER,
    report      TEXT,
    PRIMARY KEY (model, market)
);
"""


class SourceError(Exception):
    """A data source did not answer usably (network, rate limit, bad payload, no key)."""


def utc_now():
    return datetime.now(timezone.utc)


def iso(dt):
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def connect(path=DEFAULT_DB):
    """Opens (creating if needed) the store and makes sure the schema is there."""
    if path != ":memory:":
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()
    return conn


def install_universe(conn, universe=DEFAULT_UNIVERSE, added_on=None):
    """
    Adds the instruments that are missing and touches none that exist - the
    universe is frozen for an evaluation window. An entry is stamped with its
    own added_on (the kickoff date for the founding rows), else with the
    `added_on` given, else with today: a later addition is never recorded as
    a kickoff member. An entry that cannot be added because its slot is held
    by another symbol, or its symbol sits at another slot, is a conflict and
    is reported, not silently ignored. Returns (added symbols, conflicts).
    """
    added, conflicts = [], []
    fallback = added_on or utc_now().date().isoformat()
    for row in universe:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO instruments (market, symbol, name, source, source_symbol, quote, position, added_on, active) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
            (row["market"], row["symbol"], row.get("name"), row["source"], row["source_symbol"], row.get("quote"),
             int(row["position"]), row.get("added_on") or fallback))
        if cursor.rowcount:
            added.append(row["symbol"])
            continue
        holder = conn.execute("SELECT symbol FROM instruments WHERE market = ? AND position = ?",
                              (row["market"], int(row["position"]))).fetchone()
        placed = conn.execute("SELECT position FROM instruments WHERE market = ? AND symbol = ?",
                              (row["market"], row["symbol"])).fetchone()
        if holder and holder["symbol"] != row["symbol"]:
            conflicts.append(f"{row['market']} position {row['position']} is held by {holder['symbol']}; {row['symbol']} not installed "
                             f"- retire {holder['symbol']} (active = 0) and give {row['symbol']} a new position")
        elif placed and placed["position"] != int(row["position"]):
            conflicts.append(f"{row['market']}:{row['symbol']} is installed at position {placed['position']}, not {row['position']}; left as is")
    conn.commit()
    return added, conflicts


SOURCES = ("binance", "nasdaq", "yahoo", "alphavantage")


def set_source(conn, market, symbol, source, source_symbol=None):
    """
    Points an instrument at another source. Identity (market, symbol,
    position, quote) never changes, and the source symbol must name the same
    listing: an exchange suffix (ASML.AS is the EUR Euronext listing) on an
    instrument quoted in USD or USDT is refused. Sources do not share a price
    basis (Nasdaq and Yahoo adjusted, Alpha Vantage and Binance as-traded),
    so a change clears the stored bars and the next update reloads them.
    Returns the number of bars cleared.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; one of {SOURCES}")
    row = conn.execute("SELECT id, source, source_symbol, quote FROM instruments WHERE market = ? AND symbol = ?",
                       (market, symbol)).fetchone()
    if row is None:
        raise ValueError(f"no instrument {market}:{symbol}")
    if source_symbol and "." in source_symbol and str(row["quote"] or "").upper() in ("USD", "USDT"):
        raise ValueError(f"{source_symbol!r} names another listing (exchange suffix) - {market}:{symbol} is the {row['quote']} listing")
    new_symbol = source_symbol or row["source_symbol"]
    changed = source != row["source"] or new_symbol != row["source_symbol"]
    conn.execute("UPDATE instruments SET source = ?, source_symbol = ? WHERE id = ?", (source, new_symbol, row["id"]))
    cleared = 0
    if changed:
        cleared = conn.execute("DELETE FROM bars WHERE instrument_id = ?", (row["id"],)).rowcount
    conn.commit()
    return cleared


def instruments(conn, market=None, active_only=True):
    sql = "SELECT * FROM instruments"
    clauses, params = [], []
    if market:
        clauses.append("market = ?")
        params.append(market)
    if active_only:
        clauses.append("active = 1")
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY market, position"
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def last_bar_date(conn, instrument_id):
    row = conn.execute("SELECT MAX(date) AS d FROM bars WHERE instrument_id = ?", (instrument_id,)).fetchone()
    return row["d"] if row and row["d"] else None


def upsert_bars(conn, instrument_id, bars, fetched_at=None):
    """
    Writes bars (dicts with date, open, high, low, close, volume). A bar that
    already exists is updated when a value changed (sources revise), left
    alone otherwise. Returns (inserted, updated).
    """
    stamp = fetched_at or iso(utc_now())
    inserted = updated = 0
    for bar in bars:
        values = tuple(float(bar[k]) for k in ("open", "high", "low", "close"))
        volume = None if bar.get("volume") is None else float(bar["volume"])
        existing = conn.execute("SELECT open, high, low, close, volume FROM bars WHERE instrument_id = ? AND date = ?",
                                (instrument_id, bar["date"])).fetchone()
        if existing is None:
            conn.execute("INSERT INTO bars (instrument_id, date, open, high, low, close, volume, fetched_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (instrument_id, bar["date"], *values, volume, stamp))
            inserted += 1
        elif tuple(existing[k] for k in ("open", "high", "low", "close")) != values or existing["volume"] != volume:
            conn.execute("UPDATE bars SET open = ?, high = ?, low = ?, close = ?, volume = ?, fetched_at = ? WHERE instrument_id = ? AND date = ?",
                         (*values, volume, stamp, instrument_id, bar["date"]))
            updated += 1
    conn.commit()
    return inserted, updated


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_get(url, headers=None, timeout=30, retries=3, backoff=2.0, opener=None):
    """GET with a browser-ish agent and retries on 429/5xx and network errors. Returns bytes."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json, text/plain, */*", **(headers or {})})
    open_fn = opener.open if opener else urllib.request.urlopen
    last = None
    for attempt in range(retries):
        try:
            with open_fn(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (429, 500, 502, 503, 504):
                raise SourceError(f"HTTP {exc.code} from {url.split('?')[0]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            # HTTPException covers IncompleteRead and friends, which urllib does not wrap
            last = exc
        if attempt + 1 < retries:
            time.sleep(backoff * (attempt + 1))
    raise SourceError(f"{url.split('?')[0]}: {last}")


def http_json(url, **kwargs):
    raw = http_get(url, **kwargs)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SourceError(f"{url.split('?')[0]}: not JSON ({raw[:60]!r})") from exc


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


def parse_binance_klines(rows, now_ms):
    """
    Binance kline rows -> bars. A kline whose close time has not passed is the
    running day's candle, still open, and is skipped: a stored bar is a closed
    bar. The bar's date is the UTC date of its open (crypto's day is UTC).
    """
    bars = []
    for row in rows:
        open_ms, open_, high, low, close, volume, close_ms = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
        if int(close_ms) >= now_ms:
            continue
        day = datetime.fromtimestamp(int(open_ms) / 1000, tz=timezone.utc).date()
        bars.append({"date": day.isoformat(), "open": float(open_), "high": float(high), "low": float(low),
                     "close": float(close), "volume": float(volume)})
    return bars


def fetch_binance_daily(source_symbol, since=None, now=None, page_size=1000, http=http_json):
    """All closed daily bars from `since` (ISO date, inclusive) or from the listing."""
    now = now or utc_now()
    now_ms = int(now.timestamp() * 1000)
    start_ms = 0
    if since:
        start_ms = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp() * 1000)
    bars = []
    while True:
        query = urllib.parse.urlencode({"symbol": source_symbol, "interval": "1d", "limit": page_size, "startTime": start_ms})
        rows = http(f"{BINANCE_KLINES}?{query}")
        if not isinstance(rows, list):
            raise SourceError(f"binance {source_symbol}: unexpected payload {str(rows)[:80]}")
        if not rows:
            break
        bars.extend(parse_binance_klines(rows, now_ms))
        if len(rows) < page_size:
            break
        start_ms = int(rows[-1][6]) + 1
    return bars


YAHOO_CHART = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"


def parse_yahoo_chart(payload):
    """Yahoo v8 chart payload -> bars; sessions without a close (holidays, None rows) are skipped."""
    try:
        result = payload["chart"]["result"][0]
        stamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError) as exc:
        error = (payload.get("chart") or {}).get("error") if isinstance(payload, dict) else None
        raise SourceError(f"yahoo: unusable chart payload ({error or 'no result'})") from exc
    tz = timezone.utc
    zone = (result.get("meta") or {}).get("exchangeTimezoneName")
    if zone:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(zone)
        except Exception:
            tz = timezone.utc
    bars = []
    for i, stamp in enumerate(stamps):
        values = [quote.get(k, [None] * len(stamps))[i] for k in ("open", "high", "low", "close", "volume")]
        if any(v is None for v in values[:4]):
            continue
        day = datetime.fromtimestamp(int(stamp), tz=tz).date()
        bars.append({"date": day.isoformat(), "open": float(values[0]), "high": float(values[1]), "low": float(values[2]),
                     "close": float(values[3]), "volume": None if values[4] is None else float(values[4])})
    return bars


class YahooClient:
    """
    The cookie-and-crumb dance Yahoo Finance requires: a first request sets a
    cookie, /v1/test/getcrumb returns the crumb that every chart request must
    carry. One client per update run; a 429 anywhere is a SourceError, and
    the update moves on to the next instrument.
    """

    def __init__(self, http=None):
        self._jar = CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._jar))
        self._crumb = None
        self._http = http

    def _get(self, url):
        if self._http:
            return self._http(url)
        return http_get(url, opener=self._opener, retries=2)

    def crumb(self):
        if self._crumb is None:
            self._get("https://fc.yahoo.com")            # sets the cookie (a 404 body is fine)
            crumb = self._get("https://query2.finance.yahoo.com/v1/test/getcrumb").decode("utf-8", "replace").strip()
            if not crumb or " " in crumb or len(crumb) > 40:
                raise SourceError(f"yahoo: no crumb ({crumb[:40]!r})")
            self._crumb = crumb
        return self._crumb

    def fetch(self, source_symbol, since=None, now=None):
        now = now or utc_now()
        params = {"interval": "1d", "events": "history", "crumb": self.crumb()}
        if since:
            params["period1"] = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp())
            params["period2"] = int(now.timestamp())
        else:
            params["range"] = "max"
        url = YAHOO_CHART.format(symbol=urllib.parse.quote(source_symbol)) + "?" + urllib.parse.urlencode(params)
        raw = self._get(url)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SourceError(f"yahoo {source_symbol}: not JSON ({raw[:60]!r})") from exc
        bars = parse_yahoo_chart(payload)
        # the running session, if present, has not closed: keep only dates before today (exchange-local)
        today = now.date().isoformat()
        return [b for b in bars if b["date"] < today]


NASDAQ_HISTORICAL = "https://api.nasdaq.com/api/quote/{symbol}/historical"
NASDAQ_HEADERS = {"Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
NASDAQ_MAX_YEARS = 10   # the endpoint answers at most ten years back in one call


def years_back(day, years):
    """The same calendar day `years` earlier; 29 February falls back to the 28th."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def _nasdaq_number(text):
    text = str(text).replace("$", "").replace(",", "").strip()
    if not text or text.upper() == "N/A":
        return None
    return float(text)


def parse_nasdaq_historical(payload):
    """Nasdaq's historical table -> bars, oldest first; rows without a full price set are skipped."""
    try:
        rows = payload["data"]["tradesTable"]["rows"]
    except (KeyError, TypeError) as exc:
        message = None
        if isinstance(payload, dict):
            message = (payload.get("status") or {}).get("bCodeMessage") or payload.get("message")
        raise SourceError(f"nasdaq: unusable payload ({message or 'no trades table'})") from exc
    bars = []
    for row in rows or []:
        try:
            month, day, year = row["date"].split("/")
            values = [_nasdaq_number(row.get(k)) for k in ("open", "high", "low", "close")]
        except (KeyError, ValueError, AttributeError):
            continue
        if any(v is None for v in values):
            continue
        volume = _nasdaq_number(row.get("volume"))
        bars.append({"date": f"{year}-{month}-{day}", "open": values[0], "high": values[1], "low": values[2],
                     "close": values[3], "volume": volume})
    bars.sort(key=lambda b: b["date"])
    return bars


def fetch_nasdaq_daily(source_symbol, since=None, now=None, http=http_json):
    """
    Sessions from `since` (or the ten years the endpoint offers) to today. A
    row dated today is kept only after the US session has closed (16:00 New
    York plus a margin), so a stored bar is a closed bar.
    """
    now = now or utc_now()
    try:
        from zoneinfo import ZoneInfo
        new_york = now.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        new_york = now
    today = new_york.date()
    start = date.fromisoformat(since) if since else years_back(today, NASDAQ_MAX_YEARS)
    query = urllib.parse.urlencode({"assetclass": "stocks", "fromdate": start.isoformat(), "todate": today.isoformat(), "limit": 9999})
    payload = http(NASDAQ_HISTORICAL.format(symbol=urllib.parse.quote(source_symbol)) + "?" + query, headers=NASDAQ_HEADERS)
    bars = parse_nasdaq_historical(payload)
    session_closed = new_york.hour >= 17
    return [b for b in bars if b["date"] < today.isoformat() or (b["date"] == today.isoformat() and session_closed)]


ALPHAVANTAGE = "https://www.alphavantage.co/query"


def parse_alphavantage_daily(payload):
    if not isinstance(payload, dict):
        raise SourceError("alphavantage: unusable payload")
    for key in ("Note", "Information", "Error Message"):
        if key in payload:
            raise SourceError(f"alphavantage: {str(payload[key])[:120]}")
    series = payload.get("Time Series (Daily)")
    if not isinstance(series, dict):
        raise SourceError("alphavantage: no daily series in payload")
    bars = []
    for day in sorted(series):
        v = series[day]
        bars.append({"date": day, "open": float(v["1. open"]), "high": float(v["2. high"]), "low": float(v["3. low"]),
                     "close": float(v["4. close"]), "volume": float(v.get("5. volume") or 0)})
    return bars


def fetch_alphavantage_daily(source_symbol, api_key, since=None, http=http_json):
    if not api_key:
        raise SourceError("alphavantage: no key (set ALPHAVANTAGE_KEY)")
    # the compact answer is the newest 100 sessions - enough for an incremental
    # update; the first load asks for everything
    outputsize = "compact" if since else "full"
    query = urllib.parse.urlencode({"function": "TIME_SERIES_DAILY", "symbol": source_symbol, "outputsize": outputsize, "apikey": api_key})
    bars = parse_alphavantage_daily(http(f"{ALPHAVANTAGE}?{query}"))
    if since:
        bars = [b for b in bars if b["date"] >= since]
    return bars


def fetcher_for(instrument, env=None, yahoo=None):
    """A callable(since) for the instrument's source, or a SourceError."""
    env = os.environ if env is None else env
    source = instrument["source"]
    symbol = instrument["source_symbol"]
    if source == "binance":
        return lambda since=None: fetch_binance_daily(symbol, since)
    if source == "nasdaq":
        return lambda since=None: fetch_nasdaq_daily(symbol, since)
    if source == "yahoo":
        client = yahoo or YahooClient()
        return lambda since=None: client.fetch(symbol, since)
    if source == "alphavantage":
        key = env.get("ALPHAVANTAGE_KEY")
        return lambda since=None: fetch_alphavantage_daily(symbol, key, since)
    raise SourceError(f"{instrument['symbol']}: unknown source {source!r}")


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

REFETCH_DAYS = 3   # as-traded sources: re-read the newest few stored days so a revision lands
ADJUSTED_SOURCES = ("nasdaq", "yahoo")   # serve the whole history adjusted to the day of the call: always refetched whole
RESCALE_MIN_OVERLAP = 3      # overlapping bars needed to call a rescale
RESCALE_TOLERANCE = 0.01     # the overlap's ratios must agree within 1% ...
RESCALE_THRESHOLD = 0.05     # ... and sit at least 5% away from 1
FULL_HISTORY_MIN_BARS = 250  # a full answer must hold at least a year before older stored bars are dropped after a rescale


def detect_rescale(conn, instrument_id, bars):
    """
    The common ratio stored close / fetched close over the bars that are
    already stored, when they all agree and it is not 1: the source has put
    its history on a new basis (a split, an adjustment). None otherwise.
    """
    ratios = []
    for bar in bars:
        row = conn.execute("SELECT close FROM bars WHERE instrument_id = ? AND date = ?", (instrument_id, bar["date"])).fetchone()
        if row and row["close"] and float(bar["close"]) > 0:
            ratios.append(float(row["close"]) / float(bar["close"]))
    if len(ratios) < RESCALE_MIN_OVERLAP:
        return None
    if max(ratios) / min(ratios) - 1 > RESCALE_TOLERANCE:
        return None
    ratio = sum(ratios) / len(ratios)
    return None if abs(ratio - 1) < RESCALE_THRESHOLD else ratio


def update_instrument(conn, instrument, fetch, refetch_days=REFETCH_DAYS, now=None, full=False):
    """
    Update of one instrument. An as-traded source (Binance, Alpha Vantage)
    is read from a few days before the newest stored bar, so a revision
    lands; an adjusted source (Nasdaq, Yahoo) is read whole every time,
    because a split rescales everything it serves. When the overlap shows
    one common rescale, the history is rewritten in place - an as-traded
    source is then refetched whole first - and, when the full answer covers
    at least a year, the stored bars older than the answer are put on the
    new basis with the same factor, so the store ends on one scale and its
    first bar stays. `full=True` forces the whole history.
    Returns a summary dict; no exception escapes, it is recorded instead.
    """
    started = time.time()
    last = last_bar_date(conn, instrument["id"])
    adjusted = instrument["source"] in ADJUSTED_SOURCES
    since = None
    if last and not full and not adjusted:
        since = (date.fromisoformat(last) - timedelta(days=refetch_days)).isoformat()
    summary = {"symbol": instrument["symbol"], "market": instrument["market"], "source": instrument["source"],
               "since": since, "full": since is None, "fetched": 0, "inserted": 0, "updated": 0,
               "rescaled_by": None, "rescaled_older": 0, "error": None}
    try:
        bars = fetch(since)
        ratio = detect_rescale(conn, instrument["id"], bars)
        if ratio is not None and since is not None:
            # the newest bars moved by one common factor: the source's whole
            # history is on a new basis, so take all of it
            bars = fetch(None)
            since = None
            summary.update({"since": None, "full": True})
        summary["rescaled_by"] = ratio
        summary["fetched"] = len(bars)
        inserted, updated = upsert_bars(conn, instrument["id"], bars, fetched_at=iso(now or utc_now()))
        summary.update({"inserted": inserted, "updated": updated})
        if ratio is not None and since is None and len(bars) >= FULL_HISTORY_MIN_BARS:
            # the stored bars older than the answer are still on the old
            # basis: put them on the new one with the same factor. A uniform
            # factor leaves every log return - and so every bin the game cut
            # from them - unchanged, and the store's first bar stays where it
            # was, which is what keeps the game's history stable.
            first = min(b["date"] for b in bars)
            summary["rescaled_older"] = conn.execute(
                "UPDATE bars SET open = open / ?, high = high / ?, low = low / ?, close = close / ?, fetched_at = ? "
                "WHERE instrument_id = ? AND date < ?",
                (ratio, ratio, ratio, ratio, iso(now or utc_now()), instrument["id"], first)).rowcount
            conn.commit()
    except SourceError as exc:
        summary["error"] = str(exc)
    except Exception as exc:  # a malformed payload, an odd date: recorded, never fatal for the run
        summary["error"] = f"{type(exc).__name__}: {exc}"
    summary["first"] = conn.execute("SELECT MIN(date) FROM bars WHERE instrument_id = ?", (instrument["id"],)).fetchone()[0]
    summary["last"] = last_bar_date(conn, instrument["id"])
    summary["seconds"] = round(time.time() - started, 1)
    return summary


def update_all(conn, market=None, env=None, log=print, full=False):
    """Updates every active instrument (of one market, if given); one Yahoo session for the run."""
    yahoo = YahooClient()
    summaries = []
    for instrument in instruments(conn, market):
        try:
            fetch = fetcher_for(instrument, env=env, yahoo=yahoo)
        except SourceError as exc:
            summaries.append({"symbol": instrument["symbol"], "market": instrument["market"], "source": instrument["source"],
                              "error": str(exc), "fetched": 0, "inserted": 0, "updated": 0,
                              "first": None, "last": last_bar_date(conn, instrument["id"])})
            log(f"{instrument['symbol']}: {exc}")
            continue
        summary = update_instrument(conn, instrument, fetch, full=full)
        summaries.append(summary)
        if summary["error"]:
            log(f"{instrument['symbol']} ({instrument['source']}): FAILED - {summary['error']} (stored through {summary['last']})")
        else:
            scope = "the whole history" if summary["full"] else f"since {summary['since']}"
            note = ""
            if summary["rescaled_by"] is not None:
                note = (f" | HISTORY RESCALED by x{1 / summary['rescaled_by']:.4g} (a split or an adjustment) and rewritten"
                        + (f", {summary['rescaled_older']} older bar(s) put on the new basis" if summary["rescaled_older"] else ""))
            log(f"{instrument['symbol']} ({instrument['source']}): {summary['fetched']} bar(s) fetched, {scope}, "
                f"{summary['inserted']} new, {summary['updated']} revised; stored {summary['first']} .. {summary['last']} ({summary['seconds']}s){note}")
    return summaries


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

JUMP_LOG_RETURN = 0.35   # |log return| beyond which a day counts as a jump (a 2:1 split seam is 0.69)


def check_bars(conn, instrument, today=None):
    """
    Gaps, bad prices and staleness for one instrument. Crypto trades every
    day, so any missing calendar day is a gap; shares skip weekends and
    holidays, so only a hole longer than four calendar days (a long weekend
    is Friday to Tuesday) is reported.
    """
    rows = conn.execute("SELECT date, open, high, low, close FROM bars WHERE instrument_id = ? ORDER BY date",
                        (instrument["id"],)).fetchall()
    today = today or utc_now().date()
    report = {"symbol": instrument["symbol"], "rows": len(rows), "first": None, "last": None,
              "gaps": [], "bad_prices": 0, "jumps": 0, "stale_days": None}
    if not rows:
        return report
    max_gap = 1 if instrument["market"] == "crypto" else 4
    previous = None
    previous_close = None
    for row in rows:
        day = date.fromisoformat(row["date"])
        values = [row["open"], row["high"], row["low"], row["close"]]
        if any(v is None or not math.isfinite(v) or v <= 0 for v in values) or row["high"] < row["low"]:
            report["bad_prices"] += 1
        if previous is not None and (day - previous).days > max_gap:
            report["gaps"].append({"from": previous.isoformat(), "to": day.isoformat(), "days": (day - previous).days - 1})
        # a close-to-close move beyond JUMP_LOG_RETURN: for a share almost
        # always a price-scale seam (a split the store did not follow), for
        # crypto a rare real day - counted, so --check shows it either way
        if previous_close and row["close"] and row["close"] > 0 and previous_close > 0 \
                and abs(math.log(row["close"] / previous_close)) > JUMP_LOG_RETURN:
            report["jumps"] += 1
        previous = day
        previous_close = row["close"]
    report["first"], report["last"] = rows[0]["date"], rows[-1]["date"]
    report["stale_days"] = (today - date.fromisoformat(rows[-1]["date"])).days
    return report


# ---------------------------------------------------------------------------
# Series for the models (phase M2 builds the game on these)
# ---------------------------------------------------------------------------

def closes(conn, instrument_id):
    rows = conn.execute("SELECT date, close FROM bars WHERE instrument_id = ? ORDER BY date", (instrument_id,)).fetchall()
    return [r["date"] for r in rows], [float(r["close"]) for r in rows]


def log_returns(dates, values):
    """Daily log returns; the first date drops out."""
    out = [math.log(values[i] / values[i - 1]) for i in range(1, len(values))]
    return dates[1:], out


def aligned_returns(conn, market):
    """
    The market's instruments as one table: dates on which EVERY active
    instrument has a bar (a share missing a session drops that day for all),
    the matrix of log returns (days x instruments, position order) and the
    symbols. This is the sequence the positional game is cut from.
    """
    members = instruments(conn, market)
    if not members:
        return [], [], []
    series = {}
    common = None
    for member in members:
        dates, values = closes(conn, member["id"])
        series[member["id"]] = dict(zip(dates, values))
        common = set(dates) if common is None else common & set(dates)
    days = sorted(common or [])
    matrix = []
    for i in range(1, len(days)):
        row = []
        for member in members:
            s = series[member["id"]]
            row.append(math.log(s[days[i]] / s[days[i - 1]]))
        matrix.append(row)
    return days[1:], matrix, [m["symbol"] for m in members]


def status(conn, today=None):
    """One line per instrument for the CLI and, later, the page."""
    lines = []
    for instrument in instruments(conn, active_only=False):
        report = check_bars(conn, instrument, today=today)
        gap_text = f"{len(report['gaps'])} gap(s)" if report["gaps"] else "no gaps"
        lines.append(f"{instrument['market']:7s} {instrument['symbol']:5s} pos {instrument['position']} via {instrument['source']:12s} "
                     f"{report['rows']:6d} bars {report['first'] or '-'} .. {report['last'] or '-'} | {gap_text}, "
                     f"{report['bad_prices']} bad price(s), {report['jumps']} jump(s), stale {report['stale_days']} day(s)"
                     + ("" if instrument["active"] else " [inactive]"))
    return lines


# ---------------------------------------------------------------------------
# Self-check (no network)
# ---------------------------------------------------------------------------

def _self_check():
    conn = connect(":memory:")

    # 1. The frozen universe: installed once, never changed.
    added, conflicts = install_universe(conn)
    assert len(added) == len(DEFAULT_UNIVERSE) == 9 and conflicts == [], (added, conflicts)
    assert install_universe(conn) == ([], [])
    conn.execute("UPDATE instruments SET name = 'renamed' WHERE symbol = 'BTC'")
    assert install_universe(conn) == ([], []) and conn.execute("SELECT name FROM instruments WHERE symbol = 'BTC'").fetchone()[0] == "renamed"
    crypto = instruments(conn, "crypto")
    assert [i["symbol"] for i in crypto] == ["BTC", "ETH", "BNB", "XRP", "SOL"] and [i["position"] for i in crypto] == [0, 1, 2, 3, 4]
    assert all(i["added_on"] == KICKOFF for i in crypto)
    # a later window: its own date, never the kickoff's; a taken slot is a conflict, not silence
    later = [{"market": "crypto", "symbol": "ADA", "name": "Cardano", "source": "binance", "source_symbol": "ADAUSDT", "quote": "USDT", "position": 4},
             {"market": "crypto", "symbol": "DOGE", "name": "Dogecoin", "source": "binance", "source_symbol": "DOGEUSDT", "quote": "USDT", "position": 5},
             {"market": "crypto", "symbol": "SOL", "name": "Solana", "source": "binance", "source_symbol": "SOLUSDT", "quote": "USDT", "position": 6}]
    added, conflicts = install_universe(conn, later, added_on="2027-01-04")
    assert added == ["DOGE"] and len(conflicts) == 2 and "held by SOL" in conflicts[0] and "position 4, not 6" in conflicts[1], (added, conflicts)
    assert conn.execute("SELECT added_on FROM instruments WHERE symbol = 'DOGE'").fetchone()[0] == "2027-01-04"
    conn.execute("DELETE FROM instruments WHERE symbol = 'DOGE'")
    print("frozen universe: 9 instruments installed once, positions fixed, re-install changes nothing; a later entry keeps its own date, a taken slot is reported")

    # 2. Binance klines: parsed, the open candle skipped, dates in UTC.
    now = datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)
    day = lambda d: int(datetime(2026, 9, d, tzinfo=timezone.utc).timestamp() * 1000)
    rows = [[day(28), "1", "2", "0.5", "1.5", "10", day(29) - 1, "x", 1, "x", "x", "0"],
            [day(29), "1.5", "2", "1", "1.2", "11", day(30) - 1, "x", 1, "x", "x", "0"],
            [day(30), "1.2", "1.3", "1.1", "1.25", "5", day(30) + 86_400_000 - 1, "x", 1, "x", "x", "0"]]
    bars = parse_binance_klines(rows, int(now.timestamp() * 1000))
    assert [b["date"] for b in bars] == ["2026-09-28", "2026-09-29"], bars
    assert bars[1]["close"] == 1.2 and bars[0]["volume"] == 10.0
    # pagination: a fake endpoint serving 1000 then 2 rows
    served = []
    def fake_http(url):
        start = int(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["startTime"][0])
        served.append(start)
        first = 0 if start == 0 else start
        count = 1000 if start == 0 else 2
        base = day(1) - 1005 * 86_400_000 if start == 0 else first
        return [[base + i * 86_400_000, "1", "1", "1", "1", "1", base + (i + 1) * 86_400_000 - 1, "x", 1, "x", "x", "0"] for i in range(count)]
    fetched = fetch_binance_daily("BTCUSDT", now=now, http=fake_http)
    assert len(served) == 2 and served[1] == fake_http.__defaults__ is None or served[1] > served[0]
    assert len(fetched) == 1002
    print("binance: open candle skipped, UTC dates, pagination follows the last close time")

    # 3. Upsert: idempotent, revisions counted, incremental since = last - 3 days.
    btc = crypto[0]
    inserted, updated = upsert_bars(conn, btc["id"], bars, fetched_at="t0")
    assert (inserted, updated) == (2, 0)
    assert upsert_bars(conn, btc["id"], bars, fetched_at="t1") == (0, 0)
    revised = [dict(bars[1], close=1.21)]
    assert upsert_bars(conn, btc["id"], revised, fetched_at="t2") == (0, 1)
    assert conn.execute("SELECT close, fetched_at FROM bars WHERE instrument_id = ? AND date = '2026-09-29'", (btc["id"],)).fetchone()[:] == (1.21, "t2")
    asked = []
    def fake_fetch(since=None):
        asked.append(since)
        return [{"date": "2026-09-30", "open": 1, "high": 1, "low": 1, "close": 1.3, "volume": 1}]
    summary = update_instrument(conn, btc, fake_fetch, now=now)
    assert asked == ["2026-09-26"] and summary["inserted"] == 1 and summary["last"] == "2026-09-30" and summary["error"] is None
    failing = update_instrument(conn, btc, lambda since=None: (_ for _ in ()).throw(SourceError("429")), now=now)
    assert failing["error"] == "429" and failing["last"] == "2026-09-30"
    odd = update_instrument(conn, btc, lambda since=None: (_ for _ in ()).throw(KeyError("close")), now=now)
    assert odd["error"].startswith("KeyError") and odd["last"] == "2026-09-30", odd
    print("store: upsert idempotent, revisions land, incremental fetch starts 3 days before the newest bar, any failure is recorded, none escapes")

    # 3b. A split: the source rescales its whole history. On an as-traded
    #     source the overlap's common ratio escalates to a full refetch; on an
    #     adjusted source every update is a full refetch anyway. Either way
    #     the stored series ends on one scale, with no fake -ln(k) return, and
    #     after a rescale the bars older than the answer are dropped.
    xrp = crypto[3]
    days300 = [(date(2025, 12, 1) + timedelta(days=i)).isoformat() for i in range(300)]
    old_scale = [{"date": d, "open": 200, "high": 201, "low": 199, "close": 200.0, "volume": 1} for d in days300]
    upsert_bars(conn, xrp["id"], [{"date": "2025-11-01", "open": 200, "high": 201, "low": 199, "close": 200.0, "volume": 1}] + old_scale, fetched_at="t")
    new_scale = [dict(b, open=100, high=100.5, low=99.5, close=100.0) for b in old_scale] + \
                [{"date": "2026-09-27", "open": 100, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}]
    asked = []
    def split_source(since=None):
        asked.append(since)
        return [b for b in new_scale if since is None or b["date"] >= since]
    summary = update_instrument(conn, xrp, split_source, now=now)
    assert asked == ["2026-09-23", None] and summary["full"] and abs(summary["rescaled_by"] - 2.0) < 1e-9, (asked, summary)
    assert summary["rescaled_older"] == 1 and last_bar_date(conn, xrp["id"]) == "2026-09-27"
    dates_x, closes_x = closes(conn, xrp["id"])
    assert dates_x[0] == "2025-11-01" and set(closes_x) == {100.0}, "the older bar is kept, on the new basis"
    _, returns_x = log_returns(dates_x, closes_x)
    assert max(abs(r) for r in returns_x) < 1e-12, "no fake split return may remain"
    assert check_bars(conn, xrp, today=date(2026, 9, 28))["jumps"] == 0
    # the adjusted path: an instrument on nasdaq is always read whole
    msft = instruments(conn, "shares")[2]
    upsert_bars(conn, msft["id"], old_scale[:5], fetched_at="t")
    asked.clear()
    summary = update_instrument(conn, msft, split_source, now=now)
    assert asked == [None] and summary["full"] and summary["rescaled_by"] is not None
    # a small revision is not a rescale, and a seam left behind would show as a jump
    assert detect_rescale(conn, msft["id"], [dict(new_scale[0], close=100.4)]) is None
    upsert_bars(conn, msft["id"], [{"date": "2026-09-28", "open": 50, "high": 50, "low": 50, "close": 50.0, "volume": 1}], fetched_at="t")
    assert check_bars(conn, msft, today=date(2026, 9, 29))["jumps"] == 1
    conn.execute("DELETE FROM bars WHERE instrument_id IN (?, ?)", (xrp["id"], msft["id"]))
    conn.commit()
    print("split: a common rescale of the overlap rewrites the whole history on one scale, puts older bars on the new basis, and a seam shows as a jump")

    # 3c. HTTP failures that urllib does not wrap become source errors too.
    class Broken:
        def open(self, request, timeout=None):
            raise http.client.IncompleteRead(b"partial")
    try:
        http_get("https://example.invalid/x", opener=Broken(), retries=2, backoff=0)
        raise AssertionError("IncompleteRead escaped")
    except SourceError:
        pass
    class Garbage:
        def open(self, request, timeout=None):
            class R:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def read(self): return b"\x80abc"
            return R()
    try:
        http_json("https://example.invalid/x", opener=Garbage(), retries=1, backoff=0)
        raise AssertionError("undecodable body accepted")
    except SourceError:
        pass
    print("http: truncated responses and undecodable bodies are source errors, not crashes")

    # 4. Yahoo chart parsing: exchange-local dates, a None session skipped, the running session dropped.
    ny = [datetime(2026, 9, d, 13, 30, tzinfo=timezone.utc) for d in (25, 28, 29, 30)]
    payload = {"chart": {"result": [{"meta": {"exchangeTimezoneName": "America/New_York"},
                                      "timestamp": [int(t.timestamp()) for t in ny],
                                      "indicators": {"quote": [{"open": [10, None, 11, 12], "high": [11, None, 12, 13], "low": [9, None, 10, 11],
                                                                "close": [10.5, None, 11.5, 12.5], "volume": [100, None, 110, 120]}]}}],
                         "error": None}}
    ybars = parse_yahoo_chart(payload)
    assert [b["date"] for b in ybars] == ["2026-09-25", "2026-09-29", "2026-09-30"], ybars
    client = YahooClient(http=lambda url: b"abc123crumb" if "getcrumb" in url else (b"" if "fc.yahoo" in url else json.dumps(payload).encode()))
    got = client.fetch("NVDA", now=now)
    assert [b["date"] for b in got] == ["2026-09-25", "2026-09-29"] and client.crumb() == "abc123crumb"
    try:
        parse_yahoo_chart({"chart": {"result": None, "error": {"code": "Not Found"}}})
        raise AssertionError("bad payload accepted")
    except SourceError:
        pass
    rate_limited = YahooClient(http=lambda url: b"Too Many Requests")
    try:
        rate_limited.fetch("NVDA", now=now)
        raise AssertionError("429 body accepted as a crumb")
    except SourceError:
        pass
    print("yahoo: exchange-local dates, holidays skipped, running session dropped, crumb dance, bad payloads refused")

    # 4b. Nasdaq: "$" and thousands separators parsed, N/A rows skipped, oldest first,
    #     today's row kept only after the New York close; a source can be switched.
    nasdaq = {"data": {"symbol": "NVDA", "tradesTable": {"rows": [
        {"date": "09/30/2026", "close": "$230.00", "volume": "1,000", "open": "$229.00", "high": "$231.00", "low": "$228.00"},
        {"date": "09/29/2026", "close": "$227.21", "volume": "101,494,900", "open": "$230.97", "high": "$232.82", "low": "$227.025"},
        {"date": "09/28/2026", "close": "N/A", "volume": "N/A", "open": "N/A", "high": "N/A", "low": "N/A"},
        {"date": "09/25/2026", "close": "$225.07", "volume": "89,947,710", "open": "$225.13", "high": "$226.00", "low": "$224.00"}]}},
        "status": {"rCode": 200}}
    parsed = parse_nasdaq_historical(nasdaq)
    assert [b["date"] for b in parsed] == ["2026-09-25", "2026-09-29", "2026-09-30"] and parsed[1]["volume"] == 101494900.0 and parsed[1]["low"] == 227.025
    before_close = datetime(2026, 9, 30, 18, 0, tzinfo=timezone.utc)     # 14:00 New York
    after_close = datetime(2026, 9, 30, 21, 30, tzinfo=timezone.utc)     # 17:30 New York
    asked_urls = []
    def nasdaq_http(url, headers=None):
        asked_urls.append((url, headers))
        return nasdaq
    assert [b["date"] for b in fetch_nasdaq_daily("NVDA", now=before_close, http=nasdaq_http)] == ["2026-09-25", "2026-09-29"]
    assert [b["date"] for b in fetch_nasdaq_daily("NVDA", since="2026-09-27", now=after_close, http=nasdaq_http)][-1] == "2026-09-30"
    assert "fromdate=2016-09-30" in asked_urls[0][0] and "fromdate=2026-09-27" in asked_urls[1][0] and asked_urls[0][1]["Origin"].endswith("nasdaq.com")
    try:
        parse_nasdaq_historical({"status": {"rCode": 400, "bCodeMessage": [{"errorMessage": "no data"}]}})
        raise AssertionError("bad payload accepted")
    except SourceError:
        pass
    leap = fetch_nasdaq_daily("NVDA", now=datetime(2028, 2, 29, 15, 0, tzinfo=timezone.utc), http=nasdaq_http)
    assert "fromdate=2018-02-28" in asked_urls[-1][0] and leap, "the ten-year window must survive a leap day"
    # switching a source: same identity, bars cleared because the bases differ; another listing refused
    nvda_row = instruments(conn, "shares")[0]
    upsert_bars(conn, nvda_row["id"], [{"date": "2026-09-29", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}], fetched_at="t")
    assert set_source(conn, "shares", "NVDA", "nasdaq") == 0, "the same source clears nothing"
    assert set_source(conn, "shares", "NVDA", "yahoo") == 1 and last_bar_date(conn, nvda_row["id"]) is None
    assert instruments(conn, "shares")[0]["source"] == "yahoo" and instruments(conn, "shares")[0]["position"] == 0
    set_source(conn, "shares", "NVDA", "nasdaq")
    for bad in (("stooq", None), ("yahoo", "ASML.AS")):
        try:
            set_source(conn, "shares", "NVDA", *bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    print("nasdaq: prices and volumes parsed, N/A skipped, today's row only after the close, ten-year window survives a leap day; "
          "a source switch clears the bars and another listing is refused")

    # 5. Alpha Vantage parsing and the key requirement.
    av = {"Time Series (Daily)": {"2026-09-29": {"1. open": "1", "2. high": "2", "3. low": "0.5", "4. close": "1.5", "5. volume": "7"},
                                  "2026-09-26": {"1. open": "1", "2. high": "2", "3. low": "0.5", "4. close": "1.4", "5. volume": "6"}}}
    assert [b["date"] for b in parse_alphavantage_daily(av)] == ["2026-09-26", "2026-09-29"]
    try:
        parse_alphavantage_daily({"Note": "rate limit"})
        raise AssertionError("rate-limit note accepted")
    except SourceError:
        pass
    try:
        fetch_alphavantage_daily("NVDA", None)
        raise AssertionError("no key accepted")
    except SourceError:
        pass
    assert fetch_alphavantage_daily("NVDA", "k", since="2026-09-27", http=lambda url: av)[0]["date"] == "2026-09-29"
    print("alphavantage: parsed oldest-first, rate-limit notes and a missing key are source errors")

    # 6. Integrity: crypto gaps are any missing day; a share's weekend is not a gap, a week is.
    sol = crypto[4]
    upsert_bars(conn, sol["id"], [{"date": d, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
                                  for d in ("2026-09-20", "2026-09-21", "2026-09-23", "2026-09-24")], fetched_at="t")
    report = check_bars(conn, sol, today=date(2026, 9, 30))
    assert report["gaps"] == [{"from": "2026-09-21", "to": "2026-09-23", "days": 1}] and report["stale_days"] == 6, report
    nvda = instruments(conn, "shares")[0]
    upsert_bars(conn, nvda["id"], [{"date": d, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
                                   for d in ("2026-09-18", "2026-09-21", "2026-09-22", "2026-09-29")], fetched_at="t")
    report = check_bars(conn, nvda, today=date(2026, 9, 30))
    assert len(report["gaps"]) == 1 and report["gaps"][0]["from"] == "2026-09-22" and report["gaps"][0]["days"] == 6, report
    upsert_bars(conn, nvda["id"], [{"date": "2026-09-30", "open": 1, "high": 0.5, "low": 1, "close": 1, "volume": 1}], fetched_at="t")
    assert check_bars(conn, nvda, today=date(2026, 10, 1))["bad_prices"] == 1
    print("integrity: crypto misses a day = gap, a share weekend is not, a week is; inverted high/low is a bad price")

    # 7. Aligned returns keep only the days every instrument has.
    eth = crypto[1]
    for member, dates in ((btc, ["2026-10-01", "2026-10-02", "2026-10-03"]), (eth, ["2026-10-01", "2026-10-03"])):
        upsert_bars(conn, member["id"], [{"date": d, "open": 1, "high": 1, "low": 1, "close": 2.0 ** i, "volume": 1} for i, d in enumerate(dates)], fetched_at="t")
    conn.execute("UPDATE instruments SET active = 0 WHERE symbol IN ('BNB', 'XRP', 'SOL')")
    days, matrix, symbols = aligned_returns(conn, "crypto")
    assert symbols == ["BTC", "ETH"]
    assert days[-1] == "2026-10-03" and "2026-10-02" not in days, days
    assert abs(matrix[-1][1] - math.log(2.0)) < 1e-12
    d, r = log_returns(["a", "b", "c"], [1.0, 2.0, 1.0])
    assert d == ["b", "c"] and abs(r[0] - math.log(2)) < 1e-12 and abs(r[1] + math.log(2)) < 1e-12
    assert len(status(conn, today=date(2026, 10, 4))) == 9
    print("series: aligned returns drop the days an instrument lacks; log returns and status lines behave")
    print("MarketData self-check OK")


if __name__ == "__main__":
    _self_check()
