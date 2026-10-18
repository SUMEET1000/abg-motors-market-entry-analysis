"""
ABG Motors India Market Entry Feasibility Analysis
Master End-to-End Execution Pipeline
"""

import sys
import unittest
from src.data_pipeline import run_pipeline
from src.eda_figures import generate_eda_figures
from src.models import train_and_evaluate_models
from src.forecast import run_indian_forecast
from src.tableau_prep import prepare_tableau_export

def main():
    print("=" * 70)
    print("ABG MOTORS: INDIA MARKET ENTRY ANALYSIS - END-TO-END PIPELINE")
    print("=" * 70)
    
    print("\n[STEP 1/5] Ingesting, sanitizing data & engineering AGE_CAR...")
    run_pipeline()
    
    print("\n[STEP 2/5] Generating publication-quality EDA figures...")
    generate_eda_figures()
    
    print("\n[STEP 3/5] Training models, running 5-fold CV & computing Odds Ratios...")
    train_and_evaluate_models()
    
    print("\n[STEP 4/5] Scoring 70,000 Indian leads vs. 12,000 unit viability hurdle...")
    run_indian_forecast()
    
    print("\n[STEP 5/5] Exporting unified 110,000-row Tableau dataset...")
    prepare_tableau_export()
    
    print("\n" + "=" * 70)
    print("RUNNING AUTOMATED PIPELINE TESTS")
    print("=" * 70)
    suite = unittest.defaultTestLoader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)
    
    if test_result.wasSuccessful():
        print("\nSUCCESS: All pipeline steps and verification tests completed with 0 errors!")
        print("Outputs saved in:")
        print("  - Notebook:  notebooks/capstone_market_entry_analysis.ipynb")
        print("  - Reports:   reports/ABG_Motors_Market_Entry_Executive_Report.md")
        print("  - Tableau:   reports/TABLEAU_DASHBOARD_GUIDE.md")
        print("  - Figures:   reports/figures/")
        print("  - Data:      data/processed/")
    else:
        print("\nWARNING: Some tests failed. Please inspect test logs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
