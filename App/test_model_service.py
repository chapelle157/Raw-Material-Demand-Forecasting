from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from model_service import compute_inventory_policy


class InventoryPolicyTests(unittest.TestCase):
    def test_historical_variability_is_used_when_forecast_band_is_zero(self):
        forecast = pd.DataFrame(
            {
                "SKU": ["P1 / W1", "P1 / W1"],
                "Date": pd.to_datetime(["2026-01-12", "2026-01-19"]),
                "ForecastDemand": [70.0, 84.0],
                "LowerConfidence": [70.0, 84.0],
                "UpperConfidence": [70.0, 84.0],
            }
        )
        historical = pd.DataFrame(
            {
                "Product_Code": ["P1"] * 4,
                "Warehouse": ["W1"] * 4,
                "Date": pd.to_datetime(["2025-12-15", "2025-12-22", "2025-12-29", "2026-01-05"]),
                "Order_Demand": [42.0, 70.0, 56.0, 98.0],
            }
        )

        policy = compute_inventory_policy(forecast, "P1 / W1", 7, 95, 1.5, 12.5, 150, historical)

        self.assertEqual(policy["uncertainty_source"], "historical_variability")
        self.assertGreater(policy["safety_stock"], 0)
        self.assertAlmostEqual(policy["reorder_point"], 77.0 + policy["safety_stock"])

    def test_missing_forecast_is_reported_as_unavailable(self):
        forecast = pd.DataFrame(
            columns=["SKU", "Date", "ForecastDemand", "LowerConfidence", "UpperConfidence"]
        )

        policy = compute_inventory_policy(forecast, "P1 / W1", 7, 95, 1.5, 12.5, 150)

        self.assertEqual(policy["uncertainty_source"], "unavailable")
        self.assertTrue(np.isnan(policy["safety_stock"]))


if __name__ == "__main__":
    unittest.main()
