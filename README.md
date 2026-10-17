# ABG Motors: India Market Entry Feasibility Analysis
### Data Science & Machine Learning Capstone Project

An end-to-end econometric and machine learning study evaluating the feasibility of ABG Motors (a Japanese automotive manufacturer) entering the Indian passenger car market, based on comparative demographic and purchasing patterns.

---

## Key Results at a Glance

| Objective / Metric | Target Hurdle | Model Result (Base) | Stress Test ($p \ge 0.70$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Annual Vehicle Sales** | **$\ge$ 12,000 units** | **42,953 units (3.58x)** | **20,841 units (1.74x)** | **CRITERIA MET (GO)** |
| **Surveyed Market Conversion** | $\ge 17.14\%$ | 61.36% | 29.77% | Surplus: +30,953 units |
| **Model ROC-AUC** | > 0.70 | **0.722** (Logistic) / **0.784** (RF) | - | High Discrimination |
| **Model Recall (Buyers)** | > 70% | **78.03%** | - | Low False Negatives |

---

## Repository Structure

```
Internshala Data Science/
├── CAPSTONE PROJECT.pdf                        # Original project brief
├── README.md                                   # Project overview and reproduction instructions
├── requirements.txt                            # Dependencies
├── data/
│   ├── raw/                                    # Downloaded original datasets (Japan & India)
│   └── processed/
│       ├── japanese_cleaned.csv                # Sanitized training dataset (n=40,000)
│       ├── indian_featured.csv                 # Feature-engineered Indian leads (n=70,000)
│       ├── indian_forecast_results.csv         # Model scoring & predictions (n=70,000)
│       └── abg_motors_tableau_export.csv       # Unified dataset for Tableau (n=110,000)
├── notebooks/
│   └── capstone_market_entry_analysis.ipynb   # Master reproducible Jupyter Notebook
├── src/
│   ├── data_pipeline.py                        # Ingestion, cleaning & feature engineering
│   ├── eda_figures.py                          # Statistical visualization generator
│   ├── models.py                               # Model training, cross-validation & odds ratios
│   ├── forecast.py                             # Prospective scoring & target benchmarking
│   └── tableau_prep.py                         # Tableau schema merger & data export
├── reports/
│   ├── ABG_Motors_Market_Entry_Executive_Report.md # Full technical & business report
│   ├── TABLEAU_DASHBOARD_GUIDE.md              # Tableau workbook calculated fields & layout
│   └── figures/                                # High-resolution publication plots
└── tests/
    └── test_pipeline.py                        # Automated unit tests for pipeline & constraints
```

---

## Reproduction & Execution

### 1. Run Data Preparation & Feature Engineering
```bash
python -m src.data_pipeline
```

### 2. Generate EDA Visualizations
```bash
python -m src.eda_figures
```

### 3. Train Models & Evaluate Coefficients
```bash
python -m src.models
```

### 4. Forecast Indian Market & Compare vs. 12,000 Target
```bash
python -m src.forecast
```

### 5. Export Tableau Dataset
```bash
python -m src.tableau_prep
```

### 6. Run Automated Tests
```bash
python -m unittest tests/test_pipeline.py
```

---

## Deliverables Checklist (from Capstone Brief)
- [x] **Deliverable 1: Classification Model** (Logistic Regression, Random Forest, Gradient Boosting).
- [x] **Deliverable 2: Justification of Decisions** (Documented train-test split, leakage prevention, scaling).
- [x] **Deliverable 3: Business Interpretation of Coefficients** (Odds ratios, p-values, 95% CIs).
- [x] **Deliverable 4: Validation & Evaluation Metrics** (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC).
- [x] **Deliverable 5: Potential Customer Count Forecast** (42,953 base / 20,841 stress test vs. 12,000 target).
- [x] **Deliverable 6: Tableau & Visual Insights** (110k-row export + dashboard build specification).
