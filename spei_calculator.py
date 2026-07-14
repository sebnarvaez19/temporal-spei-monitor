import os
import sys
import pandas as pd
import numpy as np
import spei as si
import scipy.stats as sps

# Directories
WORKSPACE = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(WORKSPACE, "data")
CACHED_CHIRPS_PATH = os.path.join(DATA_DIR, "cached_chirps.csv")
COMPLETED_PET_PATH = os.path.join(DATA_DIR, "completed_pet.csv")
SPEI_RESULTS_PATH = os.path.join(DATA_DIR, "spei_results.csv")

def calculate_spei_all_basins(chirps_path, pet_path, output_path):
    """Calculates water balance and SPEI for 1, 3, 6, and 12 month scales for all basins."""
    print("\n=== STARTING SPEI CALCULATION PIPELINE ===")
    
    # 1. Load data
    if not os.path.exists(chirps_path):
        raise FileNotFoundError(f"Cached CHIRPS precip not found at {chirps_path}. Run data_pipeline.py first.")
    if not os.path.exists(pet_path):
        raise FileNotFoundError(f"Completed PET data not found at {pet_path}. Run forecaster.py first.")
        
    df_precip = pd.read_csv(chirps_path, parse_dates=["month"])
    df_pet = pd.read_csv(pet_path, parse_dates=["month"])
    
    # Normalize dates to the 1st of each month to align datasets (CHIRPS might be at month-end, PET at month-start)
    df_precip["month"] = df_precip["month"].dt.to_period("M").dt.to_timestamp()
    df_pet["month"] = df_pet["month"].dt.to_period("M").dt.to_timestamp()
    
    # 2. Merge datasets
    # Make sure we merge on both month and basin
    df_merged = pd.merge(df_precip, df_pet, on=["month", "basin"], how="inner")
    print(f"Merged dataset shape: {df_merged.shape}")
    print(f"Date range: {df_merged['month'].min().strftime('%Y-%m')} to {df_merged['month'].max().strftime('%Y-%m')}")
    
    # 3. Calculate Hydrological Balance (HB = Prec - PET)
    df_merged["hb"] = df_merged["precip"] - df_merged["pet"]
    
    # 4. Calculate SPEI for various timescales
    scales = [1, 3, 6, 12]
    all_results = []
    
    for basin_name, group in df_merged.groupby("basin"):
        print(f"\nCalculating SPEI for Basin: {basin_name}")
        group = group.sort_values("month")
        
        # Prepare Series with DatetimeIndex for the spei package
        # Note: The spei package requires a pandas.Series with a DatetimeIndex
        hb_series = pd.Series(group["hb"].values, index=group["month"])
        
        # Calculate for each scale
        for scale in scales:
            col_name = f"spei_{scale}"
            try:
                # Use Fisk (Log-logistic) distribution, which is the standard for SPEI
                # We specify the timescale.
                # If timescale is 0, it fits on the raw series. If timescale > 0, it rolls it automatically.
                spei_vals = si.spei(hb_series, dist=sps.fisk, timescale=scale)
                
                # Reindex to original series index to ensure correct length and align NaNs for the rolling window
                spei_vals_aligned = spei_vals.reindex(hb_series.index)
                
                # Assign values back to the group
                group[col_name] = spei_vals_aligned.values
                print(f"  Successfully calculated SPEI-{scale}")
            except Exception as e:
                print(f"  Error calculating SPEI-{scale} for {basin_name}: {e}")
                # Fallback: manually calculate rolling sum and fit
                print("  Attempting fallback manual fitting using Fisk distribution...")
                try:
                    rolling_hb = hb_series.rolling(scale, min_periods=scale).sum()
                    # Fit Fisk distribution to non-NaN values
                    valid_rolling = rolling_hb.dropna()
                    params = sps.fisk.fit(valid_rolling)
                    # Convert to CDF and then to normal standard (Z-scores)
                    cdf = sps.fisk.cdf(rolling_hb, *params)
                    # Handle boundaries to avoid infinite values
                    cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
                    spei_manual = sps.norm.ppf(cdf)
                    group[col_name] = spei_manual
                    print(f"  Successfully calculated SPEI-{scale} using manual fallback")
                except Exception as e_inner:
                    print(f"  Fallback failed: {e_inner}")
                    group[col_name] = np.nan
                    
        all_results.append(group)
        
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(output_path, index=False)
    print(f"\nSPEI calculation complete! Saved results to {output_path}")
    return combined_results

if __name__ == "__main__":
    calculate_spei_all_basins(CACHED_CHIRPS_PATH, COMPLETED_PET_PATH, SPEI_RESULTS_PATH)
