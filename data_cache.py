"""Lightweight on-disk cache for yfinance .info dicts (24h TTL).

Used by the fundamental-screen scripts so re-runs within a day don't re-hit
yfinance (faster + avoids rate limits). Cache lives in cache/ (gitignored).
"""
import os
import time
import pickle
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def cached_info(ticker, max_age_hours=24):
    """Return yf.Ticker(ticker).info, cached on disk for max_age_hours."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{ticker.replace('.', '_')}_info.pkl")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        info = {}
    time.sleep(0.15)   # politeness only on a real network fetch (cache miss)
    try:
        with open(path, "wb") as f:
            pickle.dump(info, f)
    except Exception:
        pass
    return info
