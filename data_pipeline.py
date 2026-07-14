import os
import sys
import json
import time
import requests
import geopandas as gpd
import pandas as pd
import xarray as xr
import numpy as np
from shapely.geometry import Point, shape

# Directories
WORKSPACE = r"c:\Users\snarvaez\OneDrive - Corporación Autónoma Regional del Atlántico CRA\Escritorio\desarrollos\cra risk management\temporal-spei-monitor"
DATA_DIR = os.path.join(WORKSPACE, "data")
BASINS_DIR = os.path.join(DATA_DIR, "basins")
GEOJSON_PATH = os.path.join(BASINS_DIR, "basins.geojson")
SIMPLIFIED_GEOJSON_PATH = os.path.join(BASINS_DIR, "basins_simplified.geojson")

CACHED_CHIRPS_PATH = os.path.join(DATA_DIR, "cached_chirps.csv")
CACHED_TC_PATH = os.path.join(DATA_DIR, "cached_terraclimate.csv")

def simplify_geojson(geojson_path, output_path, tolerance=0.002):
    """Loads GeoJSON, simplifies geometry to speed up API requests, and saves it."""
    print(f"Simplifying GeoJSON geometry with tolerance {tolerance}...")
    gdf = gpd.read_file(geojson_path)
    
    # Ensure it's in EPSG:4326 for coordinates used in API
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
        
    # Simplify geometry
    gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"Simplified GeoJSON saved to {output_path}")
    return gdf

def submit_climateserv_request(geom_geojson, start_date, end_date):
    """Submits POST request to ClimateServ and polls until completion, returning daily values."""
    url = "https://climateserv.servirglobal.net/api/submitDataRequest/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    params = {
        "datatype": 0,          # CHIRPS daily precipitation
        "begintime": start_date,
        "endtime": end_date,
        "intervaltype": 0,      # Daily
        "operationtype": 5,     # Average
        "geometry": json.dumps(geom_geojson)
    }
    
    print(f"Submitting ClimateServ request for period {start_date} to {end_date}...")
    res = requests.post(url, data=params, headers=headers, timeout=60)
    if res.status_code != 200:
        raise Exception(f"ClimateServ submission failed: {res.text}")
        
    job_ids = res.json()
    if not job_ids:
        raise Exception("ClimateServ returned empty job ID list")
    job_id = job_ids[0]
    print(f"Job submitted successfully. Job ID: {job_id}")
    
    # Poll for progress
    while True:
        time.sleep(3)
        progress_url = f"https://climateserv.servirglobal.net/api/getDataRequestProgress/?id={job_id}"
        prog_res = requests.get(progress_url, headers=headers, timeout=30)
        if prog_res.status_code != 200:
            print("Failed to get progress, retrying...")
            continue
            
        progress = prog_res.json()[0]
        print(f"  Job {job_id} progress: {progress}%")
        
        if progress == 100.0:
            break
        elif progress == -1:
            raise Exception("ClimateServ reported an error during processing")
            
    # Retrieve data
    data_url = f"https://climateserv.servirglobal.net/api/getDataFromRequest/?id={job_id}"
    data_res = requests.get(data_url, headers=headers, timeout=30)
    if data_res.status_code != 200:
        raise Exception(f"Failed to retrieve data from ClimateServ: {data_res.text}")
        
    raw_data = data_res.json()['data']
    
    # Parse results
    records = []
    for item in raw_data:
        # Some values might be invalid or negative (fill values)
        val = item['value']['avg']
        if val < 0:
            val = np.nan
        records.append({
            "date": pd.to_datetime(item['isodate'], format="%m/%d/%Y"),
            "precip": val
        })
        
    df = pd.DataFrame(records).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df

def fetch_all_chirps(gdf_simplified):
    """Fetches CHIRPS daily precipitation for all basins from 1981 to latest available date."""
    print("\n=== STARTING CHIRPS PRECIPITATION FETCHING ===")
    
    # Define chronological chunks to avoid server timeouts
    chunks = [
        ("01/01/1981", "12/31/1995"),
        ("01/01/1996", "12/31/2009"),
        ("01/01/2010", "12/31/2020"),
        ("01/01/2021", pd.Timestamp.now().strftime("%m/%d/%Y"))
    ]
    
    all_data = []
    
    for idx, row in gdf_simplified.iterrows():
        basin_name = row['basin']
        print(f"\nProcessing Basin: {basin_name}")
        geom_geojson = row['geometry'].__geo_interface__
        
        basin_dfs = []
        for start_d, end_d in chunks:
            try:
                chunk_df = submit_climateserv_request(geom_geojson, start_d, end_d)
                basin_dfs.append(chunk_df)
            except Exception as e:
                print(f"Error fetching chunk {start_d} - {end_d} for {basin_name}: {e}")
                print("Retrying after 10 seconds...")
                time.sleep(10)
                chunk_df = submit_climateserv_request(geom_geojson, start_d, end_d)
                basin_dfs.append(chunk_df)
                
        if basin_dfs:
            basin_df = pd.concat(basin_dfs, ignore_index=True)
            basin_df['basin'] = basin_name
            all_data.append(basin_df)
            
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Resample daily data to monthly sums
    # Group by basin and resample by Month End
    combined_df = combined_df.set_index("date")
    monthly_dfs = []
    for name, group in combined_df.groupby("basin"):
        monthly_group = group['precip'].resample("ME").sum(min_count=1) # min_count=1 prevents sum of NaNs being 0
        monthly_df = monthly_group.reset_index()
        monthly_df['basin'] = name
        monthly_dfs.append(monthly_df)
        
    monthly_combined = pd.concat(monthly_dfs, ignore_index=True)
    monthly_combined.rename(columns={"date": "month"}, inplace=True)
    
    # Save cache
    monthly_combined.to_csv(CACHED_CHIRPS_PATH, index=False)
    print(f"CHIRPS monthly cached to {CACHED_CHIRPS_PATH}")
    return monthly_combined

def fetch_all_terraclimate(gdf_simplified, start_year=1981, end_year=2025):
    """Fetches monthly PET from TerraClimate OPeNDAP from start_year to end_year."""
    print("\n=== STARTING TERRACLIMATE PET FETCHING ===")
    
    # 1. Get bounding box to crop the remote NetCDF
    bbox = gdf_simplified.total_bounds
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Add buffer
    min_lon -= 0.05
    max_lon += 0.05
    min_lat -= 0.05
    max_lat += 0.05
    
    years = list(range(start_year, end_year + 1))
    
    # Create mask for each basin on the grid coordinate system
    # We will query one dataset first to extract the grid coordinates
    first_url = f"http://thredds.northwestknowledge.net:8080/thredds/dodsC/TERRACLIMATE_ALL/data/TerraClimate_pet_{start_year}.nc"
    print(f"Connecting to TerraClimate catalog to establish grid coordinates...")
    try:
        ds = xr.open_dataset(first_url)
        # Crop to bbox
        ds_cropped = ds.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
        lats = ds_cropped.lat.values
        lons = ds_cropped.lon.values
        print(f"Grid dimensions: lats={len(lats)}, lons={len(lons)}")
    except Exception as e:
        raise Exception(f"Failed to connect to TerraClimate OPeNDAP: {e}")
        
    # Generate basin masks
    basin_masks = {}
    for idx, row in gdf_simplified.iterrows():
        basin_name = row['basin']
        geom = row['geometry']
        mask = np.zeros((len(lats), len(lons)), dtype=bool)
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                point = Point(lon, lat)
                if point.within(geom):
                    mask[i, j] = True
        basin_masks[basin_name] = mask
        print(f"  Generated spatial mask for {basin_name}: {mask.sum()} grid cells inside basin.")
        
    # Retrieve yearly datasets
    records = []
    for year in years:
        url = f"http://thredds.northwestknowledge.net:8080/thredds/dodsC/TERRACLIMATE_ALL/data/TerraClimate_pet_{year}.nc"
        print(f"Fetching TerraClimate PET for year {year}...")
        try:
            ds_year = xr.open_dataset(url, chunks={'time': 12})
            pet_year = ds_year['pet'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)).load()
            
            # For each month and basin, calculate the mean PET
            for t_idx in range(len(pet_year.time)):
                dt = pd.to_datetime(pet_year.time.values[t_idx])
                pet_month = pet_year.isel(time=t_idx).values
                
                for basin_name, mask in basin_masks.items():
                    # Mask grid
                    masked_vals = pet_month[mask]
                    # Filter out nans
                    valid_vals = masked_vals[~np.isnan(masked_vals)]
                    mean_val = np.mean(valid_vals) if len(valid_vals) > 0 else np.nan
                    
                    records.append({
                        "month": dt,
                        "basin": basin_name,
                        "pet": mean_val
                    })
        except Exception as e:
            print(f"Error fetching TerraClimate PET for year {year}: {e}")
            print("Retrying after 5 seconds...")
            time.sleep(5)
            # Retrying once
            ds_year = xr.open_dataset(url, chunks={'time': 12})
            pet_year = ds_year['pet'].sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon)).load()
            for t_idx in range(len(pet_year.time)):
                dt = pd.to_datetime(pet_year.time.values[t_idx])
                pet_month = pet_year.isel(time=t_idx).values
                for basin_name, mask in basin_masks.items():
                    masked_vals = pet_month[mask]
                    valid_vals = masked_vals[~np.isnan(masked_vals)]
                    mean_val = np.mean(valid_vals) if len(valid_vals) > 0 else np.nan
                    records.append({
                        "month": dt,
                        "basin": basin_name,
                        "pet": mean_val
                    })
                    
    df = pd.DataFrame(records).sort_values(["basin", "month"]).reset_index(drop=True)
    df.to_csv(CACHED_TC_PATH, index=False)
    print(f"TerraClimate monthly PET cached to {CACHED_TC_PATH}")
    return df

def run_pipeline(force_redownload=False):
    """Runs the full data downloading and caching pipeline."""
    # 1. Simplify and verify GeoJSON
    if not os.path.exists(SIMPLIFIED_GEOJSON_PATH) or force_redownload:
        gdf_simplified = simplify_geojson(GEOJSON_PATH, SIMPLIFIED_GEOJSON_PATH)
    else:
        gdf_simplified = gpd.read_file(SIMPLIFIED_GEOJSON_PATH)
        
    # 2. Fetch CHIRPS
    if not os.path.exists(CACHED_CHIRPS_PATH) or force_redownload:
        chirps_df = fetch_all_chirps(gdf_simplified)
    else:
        print("Using cached CHIRPS monthly precipitation data.")
        chirps_df = pd.read_csv(CACHED_CHIRPS_PATH, parse_dates=["month"])
        
    # 3. Fetch TerraClimate
    if not os.path.exists(CACHED_TC_PATH) or force_redownload:
        tc_df = fetch_all_terraclimate(gdf_simplified)
    else:
        print("Using cached TerraClimate monthly PET data.")
        tc_df = pd.read_csv(CACHED_TC_PATH, parse_dates=["month"])
        
    print("\nPipeline execution complete! Data cached and ready.")
    return chirps_df, tc_df

if __name__ == "__main__":
    run_pipeline()
