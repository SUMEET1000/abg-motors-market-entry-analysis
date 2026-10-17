import unittest
import os
import pandas as pd
import numpy as np

from src.data_pipeline import clean_income_series, clean_japanese_data, clean_indian_data
from src.forecast import run_indian_forecast
from src.tableau_prep import prepare_tableau_export

class TestABGMotorsPipeline(unittest.TestCase):
    
    def test_income_cleaning(self):
        sample = pd.Series(["  1,425,390 ", "445,344", " 70,089 "])
        cleaned = clean_income_series(sample)
        self.assertEqual(cleaned.dtype, np.float64)
        self.assertEqual(cleaned.iloc[0], 1425390.0)
        self.assertEqual(cleaned.iloc[1], 445344.0)
        self.assertEqual(cleaned.iloc[2], 70089.0)
        
    def test_indian_age_car_derivation(self):
        sample_in = pd.DataFrame({
            "ID": ["TEST1", "TEST2"],
            "CURR_AGE": [30, 45],
            "GENDER": ["M", "F"],
            "ANN_INCOME": ["500,000", "750,000"],
            "DT_MAINT": ["6/30/2019", "6/29/2019"]
        })
        res = clean_indian_data(sample_in, ref_date="2019-06-30")
        # Same day should be 0 + 1 = 1 day
        self.assertEqual(res.iloc[0]["AGE_CAR"], 1)
        # 1 day earlier should be 1 + 1 = 2 days
        self.assertEqual(res.iloc[1]["AGE_CAR"], 2)
        
    def test_forecast_meets_business_target(self):
        # Assert that the forecast output exists and exceeds 12,000 sales
        forecast_path = "data/processed/indian_forecast_results.csv"
        self.assertTrue(os.path.exists(forecast_path), "Forecast results CSV must exist.")
        df_forecast = pd.read_csv(forecast_path)
        
        rel_buyers_p50 = (df_forecast["PROB_PURCHASE_RELATIVE"] >= 0.50).sum()
        rel_buyers_p70 = (df_forecast["PROB_PURCHASE_RELATIVE"] >= 0.70).sum()
        
        self.assertGreater(rel_buyers_p50, 12000, "Base forecast must exceed 12,000 sales target")
        self.assertGreater(rel_buyers_p70, 12000, "Ultra-conservative forecast must still exceed 12,000 sales target")
        
    def test_tableau_export_structure(self):
        export_path = "data/processed/abg_motors_tableau_export.csv"
        self.assertTrue(os.path.exists(export_path), "Tableau export CSV must exist.")
        df_tab = pd.read_csv(export_path)
        
        expected_cols = [
            "CUSTOMER_ID", "COUNTRY", "CURR_AGE", "AGE_GROUP", "GENDER",
            "ANN_INCOME", "INCOME_QUARTILE", "CAR_AGE_DAYS", "CAR_AGE_BRACKET",
            "PURCHASE_FLAG", "PURCHASE_PROBABILITY", "CONVERSION_STATUS"
        ]
        for col in expected_cols:
            self.assertIn(col, df_tab.columns)
            
        self.assertEqual(len(df_tab), 110000, "Total unified records must equal 40,000 (JP) + 70,000 (IN)")

if __name__ == "__main__":
    unittest.main()
