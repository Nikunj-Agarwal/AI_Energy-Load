import requests
import pandas as pd
import zipfile
import io
from datetime import datetime, timedelta
import os

# ===== CONFIGURATION PARAMETERS =====
# Date range to fetch data for
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2021, 1, 3)  # End date is inclusive

# Location filter
LOCATION_FILTER = "Delhi"

# Output settings
OUTPUT_DIR = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets"
OUTPUT_FILENAME = "delhi_gkg_data_2021_jan1_3.csv"

# Request timeout (seconds)
TIMEOUT = 30
# ===================================

def get_gdelt_gkg_urls(start_date, end_date):
    """Generate URLs for GDELT GKG files for a date range."""
    urls = []
    current_date = start_date
    
    while current_date <= end_date:
        # GDELT GKG files are released in 15-minute intervals
        for hour in range(24):
            for minute in [0, 15, 30, 45]:
                # Format: YYYYMMDDHHMMSS
                timestamp = f"{current_date.strftime('%Y%m%d')}{hour:02d}{minute:02d}00"
                url = f"http://data.gdeltproject.org/gdeltv2/{timestamp}.gkg.csv.zip"
                urls.append((url, timestamp))
        
        current_date += timedelta(days=1)
    
    return urls

def download_and_filter_gkg(url, timestamp, location_filter):
    """Download GKG file and filter for entries containing the specified location."""
    try:
        response = requests.get(url, timeout=TIMEOUT)
        
        if response.status_code != 200:
            print(f"Failed to download {url}: Status code {response.status_code}")
            return None
        
        # Extract ZIP file content
        z = zipfile.ZipFile(io.BytesIO(response.content))
        filename = f"{timestamp}.gkg.csv"
        
        # GKG columns (V2.1 format)
        cols = ['GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName', 
                'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes', 
                'Locations', 'V2Locations', 'Persons', 'V2Persons', 'Organizations', 
                'V2Organizations', 'V2Tone', 'Dates', 'GCAM', 'SharingImage', 'RelatedImages', 
                'SocialImageEmbeds', 'SocialVideoEmbeds', 'Quotations', 'AllNames', 'Amounts', 
                'TranslationInfo', 'Extras']
        
        # Read the CSV file
        with z.open(filename) as f:
            df = pd.read_csv(f, sep='\t', header=None, names=cols, dtype=str)
        
        # Filter for entries mentioning the location in Locations or V2Locations columns
        location_mask = (df['Locations'].str.contains(location_filter, na=False, case=False) | 
                      df['V2Locations'].str.contains(location_filter, na=False, case=False))
        filtered_df = df[location_mask]
        
        print(f"Found {len(filtered_df)} entries related to {location_filter} in {timestamp}")
        return filtered_df
    
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def main():
    # Create directory to store the data if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Get URLs for the date range
    urls = get_gdelt_gkg_urls(START_DATE, END_DATE)
    print(f"Found {len(urls)} GDELT GKG files to process")
    
    # Download and filter data
    filtered_data = []
    
    for url, timestamp in urls:
        print(f"Processing {timestamp}...")
        filtered_df = download_and_filter_gkg(url, timestamp, LOCATION_FILTER)
        if filtered_df is not None and not filtered_df.empty:
            filtered_data.append(filtered_df)
    
    # Combine all filtered data
    if filtered_data:
        combined_df = pd.concat(filtered_data, ignore_index=True)
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        combined_df.to_csv(output_path, index=False)
        
        print(f"Saved {len(combined_df)} entries to {output_path}")
    else:
        print(f"No data found for {LOCATION_FILTER} in the specified date range.")

if __name__ == "__main__":
    main()