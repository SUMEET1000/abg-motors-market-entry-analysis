import os
import json
import pandas as pd
# pyrefly: ignore[missing-import]
import numpy as np
# pyrefly: ignore[missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore[missing-import]
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    confusion_matrix, roc_curve, precision_recall_curve, classification_report
)
# pyrefly: ignore[missing-import]
import statsmodels.api as sm

def train_and_evaluate_models(data_path: str = "data/processed/japanese_cleaned.csv",
                              output_dir: str = "reports/figures"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    
    # Feature matrix and target
    feature_cols = ["CURR_AGE", "IS_MALE", "ANN_INCOME", "AGE_CAR"]
    X = df[feature_cols].copy()
    y = df["PURCHASE"].copy()
    
    # 80/20 Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Dataset split: Train={X_train.shape[0]} rows, Test={X_test.shape[0]} rows")
    
    # 1. Statistical Logit Model via Statsmodels (for exact p-values, z-stats, CIs)
    # Fit on unstandardized training data for natural currency/year/day interpretations
    X_train_sm = sm.add_constant(X_train)
    logit_sm = sm.Logit(y_train, X_train_sm).fit(disp=False)
    
    coef_df = pd.DataFrame({
        "Feature": logit_sm.params.index,
        "Coefficient": logit_sm.params.values,
        "Std_Error": logit_sm.bse.values,
        "z_stat": logit_sm.tvalues.values,
        "p_value": logit_sm.pvalues.values,
        "Odds_Ratio": np.exp(logit_sm.params.values),
        "CI_Lower_95": np.exp(logit_sm.conf_int()[0].values),
        "CI_Upper_95": np.exp(logit_sm.conf_int()[1].values),
    })
    
    print("\n=== LOGISTIC REGRESSION COEFFICIENTS & ODDS RATIOS ===")
    print(coef_df.to_string(index=False))
    
    # Also fit standardized Statsmodels Logit to compare relative feature impact magnitudes
    scaler_sm = StandardScaler()
    X_train_std = pd.DataFrame(
        scaler_sm.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_train_std_sm = sm.add_constant(X_train_std)
    logit_std = sm.Logit(y_train, X_train_std_sm).fit(disp=False)
    std_coef_df = pd.DataFrame({
        "Feature": logit_std.params.index,
        "Standardized_Coef": logit_std.params.values,
        "Std_Odds_Ratio": np.exp(logit_std.params.values),
        "p_value": logit_std.pvalues.values
    })
    
    # 2. Sklearn Modeling & Benchmarking (Logistic Regression, Random Forest, Gradient Boosting)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }
    
    results = {}
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining and cross-validating {name}...")
        # 5-Fold Cross Validation on Train Set
        cv_scores = cross_validate(
            model, X_train_scaled, y_train,
            cv=cv_stratified,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1
        )
        
        # Fit on full training set
        model.fit(X_train_scaled, y_train)
        
        # Predict on held-out test set
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            "CV_Accuracy_Mean": float(np.mean(cv_scores["test_accuracy"])),
            "CV_F1_Mean": float(np.mean(cv_scores["test_f1"])),
            "CV_ROC_AUC_Mean": float(np.mean(cv_scores["test_roc_auc"])),
            "Test_Accuracy": float(accuracy_score(y_test, y_pred)),
            "Test_Precision": float(precision_score(y_test, y_pred)),
            "Test_Recall": float(recall_score(y_test, y_pred)),
            "Test_F1": float(f1_score(y_test, y_pred)),
            "Test_ROC_AUC": float(roc_auc_score(y_test, y_proba)),
            "Test_PR_AUC": float(average_precision_score(y_test, y_proba)),
            "Test_Log_Loss": float(log_loss(y_test, y_proba))
        }
        results[name] = metrics
        print(f"{name} Results on Held-out Test Set:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
            
    # Save metrics table
    metrics_df = pd.DataFrame(results).T
    metrics_path = os.path.join(output_dir, "model_comparison_metrics.csv")
    metrics_df.to_csv(metrics_path)
    
    # 3. Visualizations
    
    # Plot 1: Forest Plot of Odds Ratios
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot_coefs = coef_df[coef_df["Feature"] != "const"].copy()
    y_pos = np.arange(len(plot_coefs))
    
    ax.errorbar(
        plot_coefs["Odds_Ratio"], y_pos,
        xerr=[plot_coefs["Odds_Ratio"] - plot_coefs["CI_Lower_95"],
              plot_coefs["CI_Upper_95"] - plot_coefs["Odds_Ratio"]],
        fmt='o', color="#2B6CB0", ecolor="#2B6CB0", elinewidth=2, capsize=5, markersize=8
    )
    ax.axvline(1.0, color="gray", linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        f"{row.Feature}\n(p < 0.001)" if row.p_value < 0.001 else f"{row.Feature}\n(p={row.p_value:.3f})"
        for _, row in plot_coefs.iterrows()
    ])
    ax.set_xlabel("Odds Ratio (OR) with 95% Confidence Interval")
    ax.set_title("Logistic Regression: Predictor Odds Ratios (ABG Motors)", pad=15)
    for idx, row in plot_coefs.reset_index().iterrows():
        ax.text(row["Odds_Ratio"], idx + 0.15, f"OR = {row['Odds_Ratio']:.4f}", ha="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_odds_ratios_forest_plot.png"), dpi=200)
    plt.close()
    
    # Plot 2: ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # ROC Curves
    ax = axes[0]
    for name, model in models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Baseline (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)")
    ax.set_title("ROC Curves Comparison")
    ax.legend(loc="lower right")
    
    # PR Curves
    ax = axes[1]
    baseline_pr = y_test.mean()
    for name, model in models.items():
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {pr_auc:.3f})", linewidth=2)
    ax.axhline(baseline_pr, color="k", linestyle="--", alpha=0.6, label=f"No-Skill Baseline ({baseline_pr:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves Comparison")
    ax.legend(loc="lower left")
    
    plt.suptitle("Model Evaluation on 20% Held-Out Test Set (n=8,000)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_roc_pr_curves.png"), dpi=200)
    plt.close()
    
    # Plot 3: Confusion Matrix for Primary Model (Logistic Regression)
    primary_model = models["Logistic Regression"]
    y_pred_lr = primary_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_lr)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"])
    axes[0].set_title("Confusion Matrix (Counts)")
    
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", ax=axes[1], cbar=False,
                xticklabels=["Pred: No", "Pred: Yes"], yticklabels=["Actual: No", "Actual: Yes"])
    axes[1].set_title("Confusion Matrix (Normalized Rates)")
    
    plt.suptitle("Logistic Regression Performance on Test Set (Threshold = 0.50)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "07_confusion_matrix.png"), dpi=200)
    plt.close()
    
    return {
        "coef_df": coef_df,
        "std_coef_df": std_coef_df,
        "metrics_df": metrics_df,
        "primary_model": primary_model,
        "scaler": scaler
    }

if __name__ == "__main__":
    train_and_evaluate_models()
