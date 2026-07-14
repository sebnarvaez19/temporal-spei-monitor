import os
import sys
import requests
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Directories
WORKSPACE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(WORKSPACE, "data")
CACHED_TC_PATH = os.path.join(DATA_DIR, "cached_terraclimate.csv")
COMPLETED_PET_PATH = os.path.join(DATA_DIR, "completed_pet.csv")

SOI_URL = "https://psl.noaa.gov/data/correlation/soi.data"

def download_and_parse_soi():
    """Downloads monthly SOI data from NOAA PSL and parses it into a clean DataFrame."""
    print("Downloading monthly SOI data from NOAA PSL...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    res = requests.get(SOI_URL, headers=headers, timeout=20)
    if res.status_code != 200:
        raise Exception(f"Failed to fetch SOI: {res.status_code}")
        
    lines = res.text.split("\n")
    records = []
    
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        year_str = parts[0]
        if not year_str.isdigit() or len(year_str) != 4:
            continue
            
        year = int(year_str)
        if year < 1900 or year > 2100:
            continue
            
        if len(parts) < 13:
            continue
            
        for month in range(1, 13):
            val_str = parts[month]
            try:
                val = float(val_str)
                if val <= -99.0:
                    val = np.nan
                records.append({
                    "month": pd.to_datetime(f"{year}-{month:02d}-01"),
                    "soi": val
                })
            except ValueError:
                continue
                
    df = pd.DataFrame(records).sort_values("month").reset_index(drop=True)
    print(f"Parsed SOI data from {df['month'].min().strftime('%Y-%m')} to {df['month'].max().strftime('%Y-%m')}")
    return df

def forecast_pet_for_basins(cached_tc_path, completed_pet_path):
    """Fits seasonal regression with SOI to historical PET, and forecasts PET for 2026."""
    print("\n=== STARTING PET FORECASTING PIPELINE (SEASONAL OLS REGRESSION) ===")
    
    # 1. Load historical cached PET
    if not os.path.exists(cached_tc_path):
        raise FileNotFoundError(f"Cached TerraClimate PET not found at {cached_tc_path}. Run data_pipeline.py first.")
    df_tc = pd.read_csv(cached_tc_path, parse_dates=["month"])
    
    # Find the historical end month (usually Dec 2025)
    hist_end = df_tc["month"].max()
    print(f"Historical TerraClimate data ends in {hist_end.strftime('%Y-%m')}")
    
    # 2. Get SOI
    soi_df = download_and_parse_soi()
    
    # Define the forecast period: from hist_end + 1 month to the latest SOI available in 2026
    forecast_start = hist_end + pd.DateOffset(months=1)
    
    # Find latest available SOI in 2026 that is not NaN
    soi_2026 = soi_df[(soi_df["month"] >= forecast_start) & (soi_df["month"].dt.year <= 2026)].dropna()
    if soi_2026.empty:
        forecast_end = pd.to_datetime("2026-06-01")
        print("No SOI available for 2026. Forecasting using standard seasonal means.")
        use_soi = False
    else:
        forecast_end = soi_2026["month"].max()
        print(f"Forecast period: {forecast_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')} using seasonal OLS with SOI.")
        use_soi = True
        
    forecast_months = pd.date_range(start=forecast_start, end=forecast_end, freq="MS")
    
    all_completed_pet = []
    
    for basin_name, group in df_tc.groupby("basin"):
        print(f"\nForecasting for Basin: {basin_name}")
        
        group = group.sort_values("month").reset_index(drop=True)
        
        # Build modeling dataframe
        model_df = group.copy()
        
        # Add seasonal dummy variables (month of year: 1 to 12)
        # We will use one-hot encoding for the month. To avoid dummy variable trap with intercept, 
        # we can either include 11 dummies + intercept, or 12 dummies and no intercept.
        # 12 dummies and no intercept is very clean.
        for m in range(1, 13):
            model_df[f"month_{m}"] = (model_df["month"].dt.month == m).astype(int)
            
        # Align SOI
        model_df = pd.merge(model_df, soi_df, on="month", how="left")
        model_df["soi"] = model_df["soi"].ffill().bfill()
        
        # Fit OLS
        feature_cols = [f"month_{m}" for m in range(1, 13)]
        if use_soi:
            feature_cols.append("soi")
            
        X_train = model_df[feature_cols]
        y_train = model_df["pet"]
        
        # Fit OLS model (no constant since we have all 12 month dummies)
        model = sm.OLS(y_train, X_train).fit()
        print(f"OLS R-squared: {model.rsquared:.4f}")
        
        # Forecast features
        forecast_features = []
        for m_date in forecast_months:
            m_num = m_date.month
            feat = {f"month_{m}": int(m_num == m) for m in range(1, 13)}
            if use_soi:
                soi_val = soi_df[soi_df["month"] == m_date]["soi"].values
                feat["soi"] = soi_val[0] if len(soi_val) > 0 else 0.0
            forecast_features.append(feat)
            
        X_forecast = pd.DataFrame(forecast_features)
        
        # Predict
        predicted_pet = model.predict(X_forecast)
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            "month": forecast_months,
            "basin": basin_name,
            "pet": predicted_pet
        })
        
        # Combine historical and forecasted
        hist_df = group[["month", "basin", "pet"]]
        completed_basin_df = pd.concat([hist_df, forecast_df], ignore_index=True)
        all_completed_pet.append(completed_basin_df)
        
    combined_pet_df = pd.concat(all_completed_pet, ignore_index=True)
    combined_pet_df.to_csv(completed_pet_path, index=False)
    print(f"\nPET forecasting complete! Saved to {completed_pet_path}")
    return combined_pet_df

if __name__ == "__main__":
    forecast_pet_for_basins(CACHED_TC_PATH, COMPLETED_PET_PATH)
