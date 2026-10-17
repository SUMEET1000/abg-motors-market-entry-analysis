import os
import pandas as pd
import numpy as np

DEFAULT_REF_DATE = "2019-06-30"

def clean_income_series(series: pd.Series) -> pd.Series:
    """Strip formatting artifacts (commas, whitespace) and convert to float."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )

def clean_japanese_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize Japanese dataset attributes."""
    cleaned = df.copy()
    cleaned["ANN_INCOME"] = clean_income_series(cleaned["ANN_INCOME"])
    cleaned["CURR_AGE"] = cleaned["CURR_AGE"].astype(int)
    cleaned["AGE_CAR"] = cleaned["AGE_CAR"].astype(int)
    cleaned["GENDER"] = cleaned["GENDER"].astype(str).str.strip().str.upper()
    cleaned["IS_MALE"] = (cleaned["GENDER"] == "M").astype(int)
    cleaned["PURCHASE"] = cleaned["PURCHASE"].astype(int)
    
    # Normalized economic tier within country (z-score)
    mean_inc = cleaned["ANN_INCOME"].mean()
    std_inc = cleaned["ANN_INCOME"].std()
    cleaned["INCOME_ZSCORE"] = (cleaned["ANN_INCOME"] - mean_inc) / std_inc
    
    return cleaned

def clean_indian_data(df: pd.DataFrame, ref_date: str = DEFAULT_REF_DATE) -> pd.DataFrame:
    """Sanitize Indian dataset and engineer AGE_CAR from DT_MAINT."""
    cleaned = df.copy()
    cleaned["ANN_INCOME"] = clean_income_series(cleaned["ANN_INCOME"])
    cleaned["CURR_AGE"] = cleaned["CURR_AGE"].astype(int)
    cleaned["GENDER"] = cleaned["GENDER"].astype(str).str.strip().str.upper()
    cleaned["IS_MALE"] = (cleaned["GENDER"] == "M").astype(int)
    
    # Engineer AGE_CAR from DT_MAINT
    dt_maint = pd.to_datetime(cleaned["DT_MAINT"], errors="coerce")
    reference = pd.to_datetime(ref_date)
    # Days elapsed from maintenance date to survey reference cutoff
    # +1 ensures minimum of 1 day to match Japanese dataset range (1-1020)
    cleaned["AGE_CAR"] = (reference - dt_maint).dt.days + 1
    
    # Normalized economic tier within country (z-score)
    mean_inc = cleaned["ANN_INCOME"].mean()
    std_inc = cleaned["ANN_INCOME"].std()
    cleaned["INCOME_ZSCORE"] = (cleaned["ANN_INCOME"] - mean_inc) / std_inc
    
    return cleaned

def run_pipeline(base_dir: str = "."):
    raw_dir = os.path.join(base_dir, "data", "raw")
    proc_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    
    # Source paths (check root or raw)
    jp_src = os.path.join(raw_dir, "japanese_dataset.csv")
    if not os.path.exists(jp_src):
        jp_src = os.path.join(base_dir, "japanese_dataset.csv")
        
    in_src = os.path.join(raw_dir, "indian_dataset.csv")
    if not os.path.exists(in_src):
        in_src = os.path.join(base_dir, "indian_dataset.csv")
        
    print(f"Loading Japanese data from: {jp_src}")
    df_jp = pd.read_csv(jp_src)
    jp_clean = clean_japanese_data(df_jp)
    jp_out = os.path.join(proc_dir, "japanese_cleaned.csv")
    jp_clean.to_csv(jp_out, index=False)
    print(f"Saved cleaned Japanese data: {jp_out} (shape: {jp_clean.shape})")
    
    print(f"Loading Indian data from: {in_src}")
    df_in = pd.read_csv(in_src)
    in_clean = clean_indian_data(df_in)
    in_out = os.path.join(proc_dir, "indian_featured.csv")
    in_clean.to_csv(in_out, index=False)
    print(f"Saved featured Indian data: {in_out} (shape: {in_clean.shape})")
    
    # Also save raw copies to data/raw if not there
    raw_jp_target = os.path.join(raw_dir, "japanese_dataset.csv")
    raw_in_target = os.path.join(raw_dir, "indian_dataset.csv")
    if not os.path.exists(raw_jp_target) and os.path.exists(os.path.join(base_dir, "japanese_dataset.csv")):
        df_jp.to_csv(raw_jp_target, index=False)
    if not os.path.exists(raw_in_target) and os.path.exists(os.path.join(base_dir, "indian_dataset.csv")):
        df_in.to_csv(raw_in_target, index=False)

    return jp_clean, in_clean

if __name__ == "__main__":
    run_pipeline()
