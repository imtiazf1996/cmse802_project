# tests/test_data_clean.py
import unittest
import pandas as pd
import sys, pathlib

# add project root to path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_clean import coerce_posting_date


class TestDataClean(unittest.TestCase):
    """Unit tests for data_clean.py"""

    def setUp(self):
        # small, in-memory DataFrame simulating messy dates
        self.df = pd.DataFrame({
            "posting_date": ["2024-05-01", "05/02/2024", "bad", None],
            "price": [10000, 12000, 9000, 8000]
        })

    def test_posting_date_coerces_to_datetime(self):
        """posting_date should convert to datetime with 2 valid entries"""
        out = coerce_posting_date(self.df, col="posting_date")
        self.assertIn("posting_date", out.columns)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["posting_date"]))
        self.assertEqual(out["posting_date"].notna().sum(), 2)


if __name__ == "__main__":
    unittest.main()
