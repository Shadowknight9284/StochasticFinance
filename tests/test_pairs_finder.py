"""
Tests for the pair finder, on synthetic prices written straight into a temporary database (no network).
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import MultiStockManager
from src.strategy.pairs_finder import PairsFinder
from src.utils.config import Config

N = 500
START, END = "2022-01-03", pd.bdate_range("2022-01-03", periods=N)[-1].strftime("%Y-%m-%d")


def synthetic_prices(seed=0, n=N):
    """A and B share a random-walk trend with a fast mean-reverting spread (cointegrated); C walks on its own."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, periods=n).strftime('%Y-%m-%d')
    trend = 100 + np.cumsum(rng.normal(0, 1, n))
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.8 * spread[t - 1] + rng.normal(0, 4)
    return dates, {"AAA": trend + spread, "BBB": 0.5 * trend + 20, "CCC": 80 + np.cumsum(rng.normal(0, 1, n))}


class TestPairsFinder(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "test.db")
        manager = MultiStockManager(self.db)
        dates, prices = synthetic_prices()
        for ticker, p in prices.items():
            stock = manager.add_stock(ticker)
            pd.DataFrame({"date": dates, "open": p, "high": p, "low": p, "close": p, "volume": 1_000_000,
                          "adj_close": p}).to_sql(stock.table_name, stock.conn, if_exists="replace", index=False)
        manager.download_data_batch = self.offline  # every test here must run from the cache
        self.finder = PairsFinder(Config())
        self.finder.stock_manager = manager

    def offline(self, tickers, *a, **k):
        self.downloads = getattr(self, "downloads", []) + list(tickers)
        return {t: False for t in tickers}

    def tearDown(self):
        self.tmp.cleanup()

    def test_uses_cointegration_config(self):
        config = Config()
        config.cointegration.min_correlation = 0.3
        self.assertEqual(PairsFinder(config).validator.min_correlation, 0.3)

    def test_stock_data_is_date_indexed_and_filtered(self):
        data = self.finder.stock_manager.get_stock_data("AAA", "2022-02-01", "2022-02-28")
        self.assertEqual(data.index.min(), "2022-02-01")
        self.assertLessEqual(data.index.max(), "2022-02-28")
        self.assertIsNone(self.finder.stock_manager.get_stock_data("NOPE"))

    def test_db_path_is_respected(self):
        self.assertTrue(os.path.getsize(self.db) > 0)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "stocks.db")))

    def test_cache_coverage(self):
        m = self.finder.stock_manager
        self.assertTrue(m.has_data("AAA", START, "2023-06-01"))
        self.assertFalse(m.has_data("AAA", "2020-01-01", "2023-06-01"))

    def test_finds_the_cointegrated_pair(self):
        pairs = self.finder.find_pairs_from_universe(["AAA", "BBB", "CCC"], START, END)
        names = [p["pair"] for p in pairs]
        self.assertIn("AAA-BBB", names)
        self.assertTrue(all(p["is_valid"] for p in pairs))
        self.assertNotIn("AAA-CCC", names)
        self.assertFalse(getattr(self, "downloads", []), "cached tickers were downloaded again")

    def test_no_usable_tickers_returns_empty(self):
        self.assertEqual(self.finder.find_pairs_from_universe(["NOPE1", "NOPE2"], START, END), [])


if __name__ == "__main__":
    unittest.main()
