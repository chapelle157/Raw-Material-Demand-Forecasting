from __future__ import annotations

import unittest

from data_loader import load_demand_data, validate_uploaded_csv


class UploadedCsvValidationTests(unittest.TestCase):
    def test_builtin_dataset_loads(self):
        data = load_demand_data()
        self.assertFalse(data.full.empty)
        self.assertTrue({"Date", "Order_Demand", "Product_Code"}.issubset(data.full.columns))

    def test_valid_aliases_are_normalized(self):
        content = b"ds,Sales,material,location\n2026-01-01,10,M1,W1\n2026-01-08,12,M1,W1\n2026-01-15,14,M1,W1\n2026-01-22,16,M1,W1\n"
        result = validate_uploaded_csv(content)
        self.assertTrue(result.is_valid)
        self.assertTrue({"Date", "Order_Demand", "Product_Code", "Warehouse", "SKU"}.issubset(result.data.columns))
        self.assertEqual(result.row_count, 4)
        self.assertEqual(result.sku_count, 1)

    def test_missing_warehouse_gets_safe_default(self):
        content = b"Date,Demand,SKU\n2026-01-01,10,M1\n2026-01-08,12,M1\n2026-01-15,14,M1\n2026-01-22,16,M1\n"
        result = validate_uploaded_csv(content)
        self.assertTrue(result.is_valid)
        self.assertEqual(result.data["Warehouse"].unique().tolist(), ["Uploaded"])

    def test_empty_csv_is_rejected(self):
        result = validate_uploaded_csv(b"")
        self.assertFalse(result.is_valid)
        self.assertIn("empty", result.errors[0].lower())

    def test_missing_required_columns_are_rejected(self):
        cases = {
            "date": b"Demand,SKU\n10,M1\n12,M1\n14,M1\n16,M1\n",
            "demand": b"Date,SKU\n2026-01-01,M1\n2026-01-08,M1\n2026-01-15,M1\n2026-01-22,M1\n",
            "sku": b"Date,Demand\n2026-01-01,10\n2026-01-08,12\n2026-01-15,14\n2026-01-22,16\n",
        }
        for missing, content in cases.items():
            with self.subTest(missing=missing):
                result = validate_uploaded_csv(content)
                self.assertFalse(result.is_valid)
                self.assertIn(missing, result.errors[0].lower())

    def test_invalid_date_is_rejected(self):
        content = b"Date,Demand,SKU\ninvalid,10,M1\n2026-01-08,12,M1\n2026-01-15,14,M1\n2026-01-22,16,M1\n"
        result = validate_uploaded_csv(content)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("date" in error.lower() for error in result.errors))

    def test_non_numeric_demand_is_rejected(self):
        content = b"Date,Demand,SKU\n2026-01-01,ten,M1\n2026-01-08,12,M1\n2026-01-15,14,M1\n2026-01-22,16,M1\n"
        result = validate_uploaded_csv(content)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("non-numeric" in error.lower() for error in result.errors))

    def test_insufficient_history_is_rejected(self):
        content = b"Date,Demand,SKU\n2026-01-01,10,M1\n2026-01-08,12,M1\n2026-01-15,14,M1\n"
        result = validate_uploaded_csv(content)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("at least" in error.lower() for error in result.errors))


if __name__ == "__main__":
    unittest.main()
