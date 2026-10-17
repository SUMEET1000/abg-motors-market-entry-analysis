# Executive Analysis & Strategy Report: ABG Motors India Market Entry Feasibility

**Author**: Sumeet  
**Role**: Data Scientist / Machine Learning Engineer  
**Case Study**: ABG Motors India Market Entry Feasibility Analysis (Internshala Capstone Project)  
**Date**: October 2026  

---

## 1. Executive Summary

ABG Motors, an established Japanese automotive manufacturer, has completed a feasibility evaluation assessing commercial entry into the Indian passenger vehicle market. Executive management instituted a binding viability hurdle: **the prospective expansion must reliably forecast at least 12,000 vehicle sales** from a surveyed metropolitan sample of 70,000 potential customers over a one-year horizon.

Based on rigorous econometric modeling, statistical hypothesis testing, and cross-market behavioral projection, the empirical finding is definitive:

> **STRATEGIC RECOMMENDATION: PROCEED (GO DECISION)**  
> - **Primary Forecast (Base Scenario, $p \ge 0.50$)**: **42,953 vehicle sales** (61.4% conversion rate). This exceeds the minimum requirement by **+30,953 vehicles (3.58x the target)**.  
> - **Ultra-Conservative Stress Test ($p \ge 0.70$)**: Even if ABG Motors qualifies only prospects with $\ge 70\%$ purchase probability, projected sales reach **20,841 vehicles**, exceeding the hurdle by **+8,841 units (+73.7% safety margin)**.  
> - **Mathematical Expectation**: Cumulative probability summation across all 70,000 leads yields **40,227 expected sales**.

---

## 2. Business Context & Data Audit

### 2.1 The Strategic Problem
Entering a new national market requires high capital expenditures (CAPEX) for assembly lines, regulatory compliance, localized supply chains, and dealership distribution networks. To mitigate financial risk, ABG Motors analyzed sample data from two comparable tier-1 metropolitan cities: one in Japan (historical benchmark) and one in India (prospective expansion).

### 2.2 Datasets & Audit Findings
- **Japanese Dataset (`japanese_dataset.csv`)**: 40,000 completed customer records with historical ground truth (`PURCHASE` = 0 or 1). Features: `ID`, `CURR_AGE`, `GENDER`, `ANN_INCOME`, `AGE_CAR`.
- **Indian Dataset (`indian_dataset.csv`)**: 70,000 prospective customer leads without target labels. Features: `ID`, `CURR_AGE`, `GENDER`, `ANN_INCOME`, `DT_MAINT`.

### 2.3 Clarification of Brief Inconsistencies & Data Discoveries
1. **Curriculum Typo in Deliverable 1**: The brief states *"estimates if an individual is likely to buy a new phone"*. This is a known copy-paste typo in Internshala's curriculum. The case study is strictly automotive manufacturing and vehicle purchase behavior.
2. **Derivation of `AGE_CAR` from `DT_MAINT`**:
   - The Japanese dataset records vehicle age directly (`AGE_CAR`, spanning 1 to 1,020 days, mean: 359 days).
   - The Indian dataset provides maintenance dates (`DT_MAINT`, spanning `2016-09-14` to `2019-06-30`).
   - Calculating elapsed days from the survey cutoff date (`2019-06-30`) to `DT_MAINT` yields a distribution spanning exactly 0 to 1,019 days (mean: 366 days). Adding 1 establishes an identical 1 to 1,020 day scale, confirming that `AGE_CAR` represents elapsed days since vehicle acquisition or last major maintenance.
3. **Cross-Country Income Scale Normalization**:
   - Japanese nominal income: Mean ~359,398 (JPY scale).
   - Indian nominal income: Mean ~1,148,679 (INR scale).
   - If raw unadjusted income is used, the large numerical values in India artificially skew the model to project an unrealistic 96.2% purchase rate (67,305 sales).
   - To maintain methodological validity, our primary model uses **Within-Country Relative Socioeconomic Tiering** (Z-score / percentile standardization), mapping an individual's relative economic standing within their domestic market. Both methods exceed the 12,000 hurdle.

---

## 3. Exploratory Data Analysis & Behavioral Drivers

Demographic and behavioral analysis reveals remarkable parity between the two urban populations:

| Demographic / Attribute | Japanese Market ($n=40,000$) | Indian Market ($n=70,000$) | Parity Assessment |
| :--- | :--- | :--- | :--- |
| **Customer Age Range** | 25 – 65 years | 25 – 65 years | Identical uniform distribution |
| **Mean Customer Age** | 45.00 years | 44.99 years | Perfect parity ($\Delta = 0.01$ yr) |
| **Gender Distribution** | 55.7% Male / 44.3% Female | 50.0% Male / 50.0% Female | Highly comparable |
| **Vehicle Age / Days Since Maint** | 1 – 1,020 days (mean: 359) | 1 – 1,020 days (mean: 366) | Identical lifecycle curves |
| **Historical Purchase Rate** | 57.58% ($n=23,031$) | *Target: $\ge 17.14\%$ (12k/70k)* | Target hurdle is conservative |

### Key Behavioral Drivers in Japan:
- **Vehicle Age (Replacement Cycle)**: The single strongest operational driver. When a customer's vehicle is under 6 months old, purchase probability is only 37.9%. When the vehicle exceeds 18 months (>540 days), purchase probability surges to **78.4%**, demonstrating that aging vehicles trigger replacement purchases.
- **Income Quartile**: High earners (Tier 4) convert at 71.3%, compared to 45.1% for Tier 1.
- **Gender**: Male prospects convert at 60.1% vs. 54.4% for female prospects.
- **Age Cohort**: Conversion peaks in the 28–42 age bracket (61.2%) and slightly softens beyond age 55 (54.8%).

---

## 4. Econometric & Machine Learning Modeling

### 4.1 Methodology & Leakage Prevention (Deliverable #2)
- **Train/Test Partition**: 80% Training ($n=32,000$) / 20% Held-Out Testing ($n=8,000$), stratified on `PURCHASE`.
- **Strict Featurization Ordering**: All scalers and transformers were fitted exclusively on the training split to eliminate data snooping and data leakage.
- **Model Selection Rationale**:
  - **Logistic Regression**: Chosen as the primary model to fulfill Deliverable #3 (statistically rigorous odds ratios, p-values, and transparent business interpretation).
  - **Random Forest & Gradient Boosting**: Implemented as competitive benchmarks to evaluate non-linear feature interactions and establish performance upper bounds.

### 4.2 Model Performance Benchmarking (Deliverable #4)

| Model Name | Test Accuracy | Test Precision | Test Recall (Buyers) | Test F1-Score | Test ROC-AUC | Test PR-AUC | 5-Fold CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Primary)** | **67.07%** | **68.90%** | **78.03%** | **0.7318** | **0.7221** | **0.7692** | **67.52%** |
| Random Forest (Benchmark) | 70.39% | 74.91% | 73.04% | 0.7396 | 0.7844 | 0.8184 | 69.43% |
| Gradient Boosting (Benchmark) | 70.54% | 75.25% | 72.75% | 0.7398 | 0.7844 | 0.8177 | 69.46% |

**Evaluation Takeaways**:
- The primary Logistic Regression model achieves **78.03% Recall** on vehicle buyers, ensuring sales teams identify nearly 4 out of every 5 true purchasers.
- The 5-fold cross-validation score (67.52%) aligns with test performance (67.07%), confirming zero overfitting.

---

## 5. Statistical & Business Interpretation of Coefficients (Deliverable #3)

Estimated parameters from the econometric Logistic Regression model ($n=32,000$ training records):

| Variable | Coefficient ($\beta$) | Std Error | z-statistic | p-value | Odds Ratio ($\exp(\beta)$) | 95% Confidence Interval | Practical Business Interpretation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Intercept ($\beta_0$)** | -1.5811 | 0.0587 | -26.94 | $< 0.0001$ | 0.2058 | [0.183, 0.231] | Baseline log-odds for a female customer with zeroed baseline attributes. |
| **`IS_MALE`** | **+0.2271** | 0.0246 | +9.22 | **$< 0.0001$** | **1.2550** | **[1.196, 1.317]** | Holding all else constant, male prospects have **25.5% higher odds** of purchasing a vehicle compared to females. |
| **`CURR_AGE`** | **-0.0121** | 0.0011 | -11.46 | **$< 0.0001$** | **0.9880** | **[0.986, 0.990]** | Each additional year of age reduces purchase odds by **1.2%**. Younger working professionals are more likely to buy. |
| **`AGE_CAR`** | **+0.0042** | 0.00008 | +54.12 | **$< 0.0001$** | **1.0042** | **[1.0041, 1.0044]** | **Primary driver**. For each 100 days of vehicle aging, purchase odds increase by **+52.5%** ($\exp(0.0042 \times 100) \approx 1.525$). Over 1 year (365 days), purchase odds multiply by **4.67x**. |
| **`ANN_INCOME`** | **+2.39e-06** | 7.54e-08 | +32.42 | **$< 0.0001$** | **1.000002** | **[1.000002, 1.000003]** | For every 100,000 unit increase in annual income, purchase odds increase by **+27.0%** ($\exp(2.39 \times 10^{-6} \times 100,000) \approx 1.270$). |

All variables are statistically significant at $p < 0.0001$, confirming that age, gender, economic status, and vehicle depreciation are structural determinants of passenger car sales.

---

## 6. Indian Market Forecast & Sensitivity Analysis (Deliverable #5)

We projected vehicle demand across all 70,000 prospective Indian leads using our validated models.

### 6.1 Multi-Scenario Feasibility Matrix

| Scenario / Strategy | Probability Threshold | Projected Indian Buyers | Market Share (%) | Target Threshold | Target Surplus | Target Multiple | Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Case (Relative Socioeconomic Tier)** | **$p \ge 0.50$** | **42,953** | **61.4%** | **12,000** | **+30,953** | **3.58x** | **GO** |
| Conservative Filtering | $p \ge 0.60$ | 31,910 | 45.6% | 12,000 | +19,910 | 2.66x | GO |
| Ultra-Conservative Stress Test | $p \ge 0.70$ | 20,841 | 29.8% | 12,000 | +8,841 | 1.74x | GO |
| Mathematical Probabilistic Expectation | $\sum P_i$ | 40,227 | 57.5% | 12,000 | +28,227 | 3.35x | GO |
| Direct Unadjusted Income Baseline | $p \ge 0.50$ | 67,305 | 96.2% | 12,000 | +55,305 | 5.61x | GO |

### 6.2 Key Takeaways
1. **Breakeven Threshold Met Under Every Condition**: Even in an ultra-pessimistic stress test ($p \ge 0.70$), projected sales reach 20,841, clearing the hurdle with an 8,841-unit buffer.
2. **Target Customer Concentration**: The top 30% of Indian leads exhibit purchase probabilities exceeding 70%, allowing sales teams to achieve the 12,000-unit target by engaging only ~17,150 high-probability prospects rather than cold-calling all 70,000.

---

## 7. Tableau Visualization Deliverable (Deliverable #6)

A unified, production-ready dataset has been exported:
- **Location**: `data/processed/abg_motors_tableau_export.csv`
- **Volume**: 110,000 rows (40,000 Japan ground truth + 70,000 India forecast).
- **Enriched Dimensions**: `AGE_GROUP`, `INCOME_QUARTILE`, `CAR_AGE_BRACKET`, `PURCHASE_FLAG`, `PURCHASE_PROBABILITY`, `CONVERSION_STATUS`, `DATA_STAGE`.
- **Implementation Guide**: Complete step-by-step workbook build specifications, calculated fields, and dashboard layouts are provided in [`reports/TABLEAU_DASHBOARD_GUIDE.md`](file:///c:/Users/sumee/Documents/Internshala%20Data%20Science/reports/TABLEAU_DASHBOARD_GUIDE.md).

---

## 8. Strategic Go-To-Market Roadmap

1. **Phase 1: Dealership & Distribution Setup (Months 1–6)**:
   - Establish physical footprint in the surveyed metropolitan city, focusing on commercial hubs with high concentrations of Tier 3 and 4 income earners.
2. **Phase 2: Target Lead Scoring & CRM Integration (Months 6–9)**:
   - Deploy the predictive scoring model into ABG Motors' CRM. Sort leads by `PROB_PURCHASE_RELATIVE` desc.
   - Dispatch sales representatives exclusively to prospects with $p \ge 0.65$ to maximize conversion efficiency and reduce customer acquisition costs (CAC).
3. **Phase 3: Vehicle Upgrade & Trade-In Campaign (Months 9–12)**:
   - Given that vehicles older than 18 months exhibit a 4.67x surge in replacement odds, launch an aggressive *"Exchange & Upgrade"* promotion offering competitive trade-in valuations on existing cars.
