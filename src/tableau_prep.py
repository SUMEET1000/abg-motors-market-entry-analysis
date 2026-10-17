import os
import pandas as pd
import numpy as np

def prepare_tableau_export(base_dir: str = "."):
    proc_dir = os.path.join(base_dir, "data", "processed")
    jp_path = os.path.join(proc_dir, "japanese_cleaned.csv")
    in_path = os.path.join(proc_dir, "indian_forecast_results.csv")
    
    df_jp = pd.read_csv(jp_path)
    df_in = pd.read_csv(in_path)
    
    # Process Japan records
    jp_export = pd.DataFrame()
    jp_export["CUSTOMER_ID"] = df_jp["ID"]
    jp_export["COUNTRY"] = "Japan"
    jp_export["CURR_AGE"] = df_jp["CURR_AGE"]
    jp_export["AGE_GROUP"] = pd.cut(
        df_jp["CURR_AGE"],
        bins=[24, 34, 49, 66],
        labels=["25-34 (Young)", "35-49 (Middle)", "50-65 (Mature)"]
    )
    jp_export["GENDER"] = df_jp["GENDER"].map({"M": "Male", "F": "Female"})
    jp_export["ANN_INCOME"] = df_jp["ANN_INCOME"]
    jp_export["INCOME_QUARTILE"] = pd.qcut(df_jp["ANN_INCOME"], q=4, labels=["Tier 1 (Lowest)", "Tier 2", "Tier 3", "Tier 4 (Highest)"])
    jp_export["CAR_AGE_DAYS"] = df_jp["AGE_CAR"]
    jp_export["CAR_AGE_BRACKET"] = pd.cut(
        df_jp["AGE_CAR"],
        bins=[0, 180, 365, 730, 1100],
        labels=["< 6 Months", "6-12 Months", "1-2 Years", "> 2 Years"]
    )
    jp_export["PURCHASE_FLAG"] = df_jp["PURCHASE"]
    jp_export["PURCHASE_PROBABILITY"] = df_jp["PURCHASE"].astype(float)
    jp_export["CONVERSION_STATUS"] = df_jp["PURCHASE"].map({1: "Actual Buyer", 0: "Actual Non-Buyer"})
    jp_export["DATA_STAGE"] = "Historical Ground Truth (Train/Eval)"
    
    # Process India records
    in_export = pd.DataFrame()
    in_export["CUSTOMER_ID"] = df_in["ID"]
    in_export["COUNTRY"] = "India"
    in_export["CURR_AGE"] = df_in["CURR_AGE"]
    in_export["AGE_GROUP"] = pd.cut(
        df_in["CURR_AGE"],
        bins=[24, 34, 49, 66],
        labels=["25-34 (Young)", "35-49 (Middle)", "50-65 (Mature)"]
    )
    in_export["GENDER"] = df_in["GENDER"].map({"M": "Male", "F": "Female"})
    in_export["ANN_INCOME"] = df_in["ANN_INCOME"]
    in_export["INCOME_QUARTILE"] = pd.qcut(df_in["ANN_INCOME"], q=4, labels=["Tier 1 (Lowest)", "Tier 2", "Tier 3", "Tier 4 (Highest)"])
    in_export["CAR_AGE_DAYS"] = df_in["AGE_CAR"]
    in_export["CAR_AGE_BRACKET"] = pd.cut(
        df_in["AGE_CAR"],
        bins=[0, 180, 365, 730, 1100],
        labels=["< 6 Months", "6-12 Months", "1-2 Years", "> 2 Years"]
    )
    in_export["PURCHASE_FLAG"] = df_in["PRED_PURCHASE_RELATIVE"]
    in_export["PURCHASE_PROBABILITY"] = df_in["PROB_PURCHASE_RELATIVE"].round(4)
    in_export["CONVERSION_STATUS"] = df_in["PRED_PURCHASE_RELATIVE"].map({1: "Projected Buyer", 0: "Projected Non-Buyer"})
    in_export["DATA_STAGE"] = "Prospective Lead Forecast"
    
    combined = pd.concat([jp_export, in_export], ignore_index=True)
    out_file = os.path.join(proc_dir, "abg_motors_tableau_export.csv")
    combined.to_csv(out_file, index=False)
    print(f"Generated unified Tableau export: {out_file} (Total records: {len(combined):,})")
    print(combined.groupby(["COUNTRY", "CONVERSION_STATUS"]).size())
    return combined

if __name__ == "__main__":
    prepare_tableau_export()
