import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
})

def generate_eda_figures(output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    df_jp = pd.read_csv("data/processed/japanese_cleaned.csv")
    df_in = pd.read_csv("data/processed/indian_featured.csv")
    
    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df_jp["PURCHASE"].value_counts(normalize=True) * 100
    bars = ax.bar(["Did Not Buy (0)", "Bought Car (1)"], [counts[0], counts[1]], color=["#718096", "#2B6CB0"], width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 75)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Japanese Dataset Target Variable: PURCHASE Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_target_distribution.png"), dpi=200)
    plt.close()
    
    # 2. Purchase Rate by Predictors
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # By Gender
    gender_grp = df_jp.groupby("GENDER")["PURCHASE"].mean() * 100
    ax = axes[0, 0]
    bars = ax.bar(["Female (F)", "Male (M)"], [gender_grp["F"], gender_grp["M"]], color=["#D53F8C", "#3182CE"], width=0.45)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 75)
    ax.set_ylabel("Purchase Rate (%)")
    ax.set_title("Purchase Propensity by Gender")
    
    # By Age Bins
    df_jp["AGE_BIN"] = pd.cut(df_jp["CURR_AGE"], bins=[24, 34, 44, 54, 65], labels=["25-34", "35-44", "45-54", "55-65"])
    age_grp = df_jp.groupby("AGE_BIN", observed=False)["PURCHASE"].mean() * 100
    ax = axes[0, 1]
    ax.plot(age_grp.index.astype(str), age_grp.values, marker="o", color="#DD6B20", linewidth=2.5, markersize=8)
    for x, y in zip(age_grp.index.astype(str), age_grp.values):
        ax.text(x, y + 0.8, f"{y:.1f}%", ha="center", fontweight="bold")
    ax.set_ylim(50, 65)
    ax.set_ylabel("Purchase Rate (%)")
    ax.set_title("Purchase Propensity by Age Cohort")
    
    # By Income Quartiles
    df_jp["INCOME_QUARTILE"] = pd.qcut(df_jp["ANN_INCOME"], q=4, labels=["Q1 (Lowest)", "Q2", "Q3", "Q4 (Highest)"])
    inc_grp = df_jp.groupby("INCOME_QUARTILE", observed=False)["PURCHASE"].mean() * 100
    ax = axes[1, 0]
    bars = ax.bar(inc_grp.index.astype(str), inc_grp.values, color=sns.color_palette("Blues_d", 4), width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 85)
    ax.set_ylabel("Purchase Rate (%)")
    ax.set_title("Purchase Propensity by Income Quartile")
    
    # By Car Age Bins (Replacement Cycle)
    df_jp["CAR_AGE_BIN"] = pd.cut(df_jp["AGE_CAR"], bins=[0, 180, 365, 540, 730, 1050], labels=["<6 mos", "6-12 mos", "1-1.5 yrs", "1.5-2 yrs", ">2 yrs"])
    car_age_grp = df_jp.groupby("CAR_AGE_BIN", observed=False)["PURCHASE"].mean() * 100
    ax = axes[1, 1]
    bars = ax.bar(car_age_grp.index.astype(str), car_age_grp.values, color=sns.color_palette("Purples_d", 5), width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 95)
    ax.set_ylabel("Purchase Rate (%)")
    ax.set_title("Purchase Propensity by Vehicle Age (Replacement Cycle)")
    
    plt.suptitle("ABG Motors: Core Behavioral Drivers of Vehicle Purchase (Japan Market)", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_purchase_drivers.png"), dpi=200)
    plt.close()
    
    # 3. Cross-Market Demographic Comparison (Japan vs India)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Age comparison
    ax = axes[0]
    sns.kdeplot(df_jp["CURR_AGE"], ax=ax, label="Japan (n=40,000)", color="#2B6CB0", fill=True, alpha=0.3)
    sns.kdeplot(df_in["CURR_AGE"], ax=ax, label="India (n=70,000)", color="#C53030", fill=True, alpha=0.3)
    ax.set_title("Age Distribution: Japan vs. India")
    ax.set_xlabel("Customer Age")
    ax.legend()
    
    # Car Age / Days Since Maint comparison
    ax = axes[1]
    sns.kdeplot(df_jp["AGE_CAR"], ax=ax, label="Japan (AGE_CAR)", color="#2B6CB0", fill=True, alpha=0.3)
    sns.kdeplot(df_in["AGE_CAR"], ax=ax, label="India (Derived AGE_CAR)", color="#C53030", fill=True, alpha=0.3)
    ax.set_title("Vehicle Age Distribution: Japan vs. India")
    ax.set_xlabel("Car Age / Days Elapsed")
    ax.legend()
    
    # Gender split comparison
    ax = axes[2]
    gender_comp = pd.DataFrame({
        "Japan": df_jp["GENDER"].value_counts(normalize=True) * 100,
        "India": df_in["GENDER"].value_counts(normalize=True) * 100
    })
    gender_comp.plot(kind="bar", ax=ax, color=["#2B6CB0", "#C53030"], width=0.6, rot=0)
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Gender Breakdown: Japan vs. India")
    ax.set_ylim(0, 70)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height() + 1), ha='center', fontweight='bold')
    
    plt.suptitle("Cross-Market Demographic Parity Analysis: Japan vs. India", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_cross_market_comparison.png"), dpi=200)
    plt.close()
    
    # 4. Correlation Matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    corr_cols = ["CURR_AGE", "IS_MALE", "ANN_INCOME", "AGE_CAR", "PURCHASE"]
    corr = df_jp[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-0.5, vmax=0.5, cbar=True, ax=ax, square=True)
    ax.set_title("Correlation Matrix (Japanese Training Set)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_correlation_matrix.png"), dpi=200)
    plt.close()
    
    print(f"Generated all EDA figures in: {output_dir}")

if __name__ == "__main__":
    generate_eda_figures()
