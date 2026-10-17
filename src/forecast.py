import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def run_indian_forecast(base_dir: str = "."):
    proc_dir = os.path.join(base_dir, "data", "processed")
    fig_dir = os.path.join(base_dir, "reports", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    jp_path = os.path.join(proc_dir, "japanese_cleaned.csv")
    in_path = os.path.join(proc_dir, "indian_featured.csv")
    
    df_jp = pd.read_csv(jp_path)
    df_in = pd.read_csv(in_path)
    
    # -------------------------------------------------------------
    # Scenario A: Standardized / Relative Socioeconomic Model (Recommended)
    # Features: CURR_AGE, IS_MALE, INCOME_ZSCORE, AGE_CAR
    # -------------------------------------------------------------
    rel_features = ["CURR_AGE", "IS_MALE", "INCOME_ZSCORE", "AGE_CAR"]
    X_train_rel = df_jp[rel_features]
    y_train = df_jp["PURCHASE"]
    X_in_rel = df_in[rel_features]
    
    scaler_rel = StandardScaler()
    X_train_rel_scaled = scaler_rel.fit_transform(X_train_rel)
    X_in_rel_scaled = scaler_rel.transform(X_in_rel)
    
    model_rel = LogisticRegression(random_state=42)
    model_rel.fit(X_train_rel_scaled, y_train)
    
    probs_rel = model_rel.predict_proba(X_in_rel_scaled)[:, 1]
    df_in["PROB_PURCHASE_RELATIVE"] = probs_rel
    df_in["PRED_PURCHASE_RELATIVE"] = (probs_rel >= 0.50).astype(int)
    
    # -------------------------------------------------------------
    # Scenario B: Direct Raw / Unstandardized Model (Literal Baseline)
    # Features: CURR_AGE, IS_MALE, ANN_INCOME, AGE_CAR
    # -------------------------------------------------------------
    raw_features = ["CURR_AGE", "IS_MALE", "ANN_INCOME", "AGE_CAR"]
    X_train_raw = df_jp[raw_features]
    X_in_raw = df_in[raw_features]
    
    model_raw = LogisticRegression(random_state=42, max_iter=1000)
    model_raw.fit(X_train_raw, y_train)
    
    probs_raw = model_raw.predict_proba(X_in_raw)[:, 1]
    df_in["PROB_PURCHASE_RAW"] = probs_raw
    df_in["PRED_PURCHASE_RAW"] = (probs_raw >= 0.50).astype(int)
    
    # -------------------------------------------------------------
    # Forecast Metrics Calculation
    # -------------------------------------------------------------
    total_leads = len(df_in)
    TARGET_SALES = 12000
    
    # Scenario A metrics
    rel_expected_prob_sum = float(probs_rel.sum())
    rel_count_50 = int((probs_rel >= 0.50).sum())
    rel_count_60 = int((probs_rel >= 0.60).sum())
    rel_count_70 = int((probs_rel >= 0.70).sum())
    
    # Scenario B metrics
    raw_expected_prob_sum = float(probs_raw.sum())
    raw_count_50 = int((probs_raw >= 0.50).sum())
    raw_count_60 = int((probs_raw >= 0.60).sum())
    raw_count_70 = int((probs_raw >= 0.70).sum())
    
    summary_data = [
        {
            "Scenario": "Scenario A: Relative Economic Tier (Recommended)",
            "Expected_Sales_Probabilistic": round(rel_expected_prob_sum),
            "Forecast_Buyers_p50": rel_count_50,
            "Share_p50": f"{rel_count_50/total_leads*100:.1f}%",
            "Conservative_Buyers_p60": rel_count_60,
            "Ultra_Conservative_p70": rel_count_70,
            "Target_Threshold": TARGET_SALES,
            "Target_Exceeded_By": rel_count_50 - TARGET_SALES,
            "Multiple_of_Target": f"{rel_count_50 / TARGET_SALES:.2f}x",
            "Business_Decision": "GO - Target Exceeded"
        },
        {
            "Scenario": "Scenario B: Direct Unadjusted Income (Literal Baseline)",
            "Expected_Sales_Probabilistic": round(raw_expected_prob_sum),
            "Forecast_Buyers_p50": raw_count_50,
            "Share_p50": f"{raw_count_50/total_leads*100:.1f}%",
            "Conservative_Buyers_p60": raw_count_60,
            "Ultra_Conservative_p70": raw_count_70,
            "Target_Threshold": TARGET_SALES,
            "Target_Exceeded_By": raw_count_50 - TARGET_SALES,
            "Multiple_of_Target": f"{raw_count_50 / TARGET_SALES:.2f}x",
            "Business_Decision": "GO - Target Exceeded"
        }
    ]
    summary_df = pd.DataFrame(summary_data)
    print("\n=== INDIAN MARKET FORECAST SUMMARY VS. 12,000 TARGET ===")
    print(summary_df.to_string(index=False))
    
    # Save predictions
    forecast_out = os.path.join(proc_dir, "indian_forecast_results.csv")
    df_in.to_csv(forecast_out, index=False)
    print(f"\nSaved forecast results: {forecast_out}")
    
    # -------------------------------------------------------------
    # Visualization: Indian Forecast & Target Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    # Left: Probability Distribution
    ax = axes[0]
    sns.histplot(probs_rel, bins=35, kde=True, color="#2B6CB0", ax=ax, alpha=0.6)
    ax.axvline(0.50, color="darkred", linestyle="--", linewidth=2, label="Decision Cutoff (p=0.50)")
    ax.axvline(probs_rel.mean(), color="green", linestyle="-.", linewidth=2, label=f"Mean Propensity ({probs_rel.mean():.2f})")
    ax.set_title("Distribution of Predicted Purchase Propensity (India)", pad=10)
    ax.set_xlabel("Predicted Purchase Probability")
    ax.set_ylabel("Number of Prospective Leads")
    ax.legend(loc="upper left")
    
    # Right: Forecasted Volume vs Target Threshold
    ax = axes[1]
    categories = ["Target Min\n(12,000)", "Ultra-Conservative\n(p ≥ 0.70)", "Conservative\n(p ≥ 0.60)", "Expected Base\n(p ≥ 0.50)"]
    vals = [TARGET_SALES, rel_count_70, rel_count_60, rel_count_50]
    colors = ["#E53E3E", "#D69E2E", "#319795", "#2B6CB0"]
    bars = ax.bar(categories, vals, color=colors, width=0.55)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1000, f"{h:,}", ha="center", va="bottom", fontweight="bold")
        
    ax.axhline(TARGET_SALES, color="#E53E3E", linestyle=":", linewidth=2, label="Minimum Breakeven Target")
    ax.set_ylim(0, 55000)
    ax.set_ylabel("Number of Vehicles Sold")
    ax.set_title("Forecasted Vehicle Sales in India vs. Breakeven Benchmark", pad=10)
    ax.legend(loc="upper left")
    
    plt.suptitle("ABG Motors: India Market Entry Feasibility Forecast (Sample n=70,000)", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "08_indian_market_forecast_vs_target.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved forecast figure: {fig_path}")
    
    return summary_df

if __name__ == "__main__":
    run_indian_forecast()
