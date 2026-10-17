# ABG Motors: Tableau Executive Dashboard Guide

This guide provides step-by-step instructions to create an interactive Tableau Dashboard using `data/processed/abg_motors_tableau_export.csv` to fulfill **Deliverable #6** of the Capstone Project.

---

## 1. Data Source Connection

1. Open **Tableau Desktop** or **Tableau Public**.
2. Under **Connect** -> Choose **Text file**.
3. Select `data/processed/abg_motors_tableau_export.csv`.
4. Verify the 110,000 rows loaded with fields:
   - `CUSTOMER_ID` (String)
   - `COUNTRY` (String: "Japan", "India")
   - `CURR_AGE` (Number / Dimension)
   - `AGE_GROUP` (String Dimension)
   - `GENDER` (String Dimension)
   - `ANN_INCOME` (Number / Measure)
   - `INCOME_QUARTILE` (String Dimension)
   - `CAR_AGE_DAYS` (Number / Measure)
   - `CAR_AGE_BRACKET` (String Dimension)
   - `PURCHASE_FLAG` (Number / Measure)
   - `PURCHASE_PROBABILITY` (Number / Measure)
   - `CONVERSION_STATUS` (String Dimension)
   - `DATA_STAGE` (String Dimension)

---

## 2. Calculated Fields to Create in Tableau

Create the following calculated fields in Tableau (Right-click Data pane -> **Create Calculated Field**):

### Field 1: `Target Sales Benchmark`
```tableau
IF [COUNTRY] = "India" THEN 12000 ELSE NULL END
```

### Field 2: `Forecasted Sales Volume (India)`
```tableau
IF [COUNTRY] = "India" AND [PURCHASE_FLAG] = 1 THEN 1 ELSE 0 END
```

### Field 3: `Conversion Rate (%)`
```tableau
SUM([PURCHASE_FLAG]) / COUNT([CUSTOMER_ID])
```

### Field 4: `Target Achievement Ratio`
```tableau
SUM([Forecasted Sales Volume (India)]) / 12000
```

---

## 3. Worksheets to Build

### Sheet 1: Executive KPI Cards (Banner)
- **Visual Type**: Text / Big Number KPI Cards.
- **Metrics**:
  - **Indian Market Sample**: `COUNTD([CUSTOMER_ID])` where `COUNTRY = "India"` (displays **70,000**).
  - **Forecasted Vehicle Buyers**: `SUM([Forecasted Sales Volume (India)])` (displays **42,953**).
  - **Target Sales Requirement**: Static / Calculated Field (displays **12,000**).
  - **Target Achievement Multiple**: Displays **3.58x** (+30,953 units above target).
  - **Overall Market Conversion**: Displays **61.4%**.

### Sheet 2: Demographic Comparison: Japan vs. India
- **Visual Type**: Side-by-Side Horizontal Bar Chart.
- **Rows**: `AGE_GROUP`, `GENDER`
- **Columns**: `COUNTRY`, `Conversion Rate (%)`
- **Color**: `COUNTRY` (Deep Blue `#2B6CB0` for Japan, Crimson Red `#C53030` for India).
- **Label**: `Conversion Rate (%)` formatted as percentage with 1 decimal.
- **Key Insight**: Shows that in both markets, male customers in the 25-34 and 35-49 brackets exhibit peak conversion rates (~65-68%).

### Sheet 3: Vehicle Replacement Cycle (Car Age vs Purchase Propensity)
- **Visual Type**: Stepped Line Chart / Area Chart.
- **Columns**: `CAR_AGE_BRACKET`
- **Rows**: `AVG([PURCHASE_PROBABILITY])`
- **Color**: `COUNTRY`
- **Key Insight**: Graph shows dramatic inflection: when vehicles pass 1 year of age (>365 days), purchase propensity jumps from ~38% to over 74%, confirming strong replacement-driven demand in both countries.

### Sheet 4: Income Tier Conversion Matrix
- **Visual Type**: Heatmap / Highlight Table.
- **Rows**: `INCOME_QUARTILE`
- **Columns**: `COUNTRY`
- **Color**: `Conversion Rate (%)` (Palette: White to Green diverging).
- **Label**: `Conversion Rate (%)` and `COUNT([CUSTOMER_ID])`.
- **Key Insight**: Highest income quartile (Tier 4) achieves >71% purchase rate in India, representing the primary segment for ABG's flagship models.

---

## 4. Executive Dashboard Assembly

1. Create a New Dashboard (`1366 x 768` or `Automatic` desktop layout).
2. Add Title: **"ABG Motors: India Market Entry Feasibility & Consumer Dynamics"**.
3. Place **Sheet 1 (KPI Banner)** across the top.
4. Place **Sheet 2 (Demographic Parity)** on the left half.
5. Place **Sheet 3 (Replacement Cycle)** and **Sheet 4 (Income Matrix)** on the right half.
6. Add Global Interactive Filters on the right sidebar:
   - `COUNTRY` (Single Value List)
   - `GENDER` (Multiple Values Dropdown)
   - `AGE_GROUP` (Multiple Values Dropdown)
   - `INCOME_QUARTILE` (Multiple Values Dropdown)
7. Set filter action: Apply to all worksheets using this data source.

---

## 5. Summary Findings for Presentation

- **Strategic Go-Ahead**: Across all demographic cohorts, forecasted demand in the Indian test market reaches **42,953 vehicle sales**, over **3.5x higher** than the company's minimum viability hurdle of 12,000.
- **Target Customer Profile**: Urban males aged 28–45 in Income Tier 3 & 4 with existing vehicles older than 18 months represent the highest propensity segment (conversion probability > 80%).
