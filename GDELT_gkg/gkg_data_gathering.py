import requests
import pandas as pd
import zipfile
import io
from datetime import datetime, timedelta
import os
import concurrent.futures
from tqdm import tqdm  # pip install tqdm if not installed
import time
import hashlib

"""
GDELT GKG Data Gathering Script

This script downloads and processes GDELT Global Knowledge Graph (GKG) data,
filtering for entries relevant to a specific location. It supports:

1. Full processing: Download and process all data in one go
2. Batch processing: Process data in 3-month chunks
3. Resumable batch processing: Continue from where processing was interrupted

Usage:
- To run all data in one go, uncomment the 'main()' call at the end
- To run in batches, uncomment the 'process_in_batches()' call
- To resume interrupted batch processing, uncomment the 'process_in_batches_with_resume()' call

Configuration:
- Adjust the START_DATE, END_DATE, and LOCATION_FILTER as needed
- Set OUTPUT_DIR and OUTPUT_FILENAME to control where data is saved
- Adjust max_workers in the ThreadPoolExecutor for more/less parallelism

Notes:
- Downloaded data is cached to avoid re-downloading
- Cache directory should be added to .gitignore if using version control
"""

# ===== CONFIGURATION PARAMETERS =====
# Date range to fetch data for
START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2024, 12, 12)  # End date is inclusive

# Location filter
LOCATION_FILTER = "Delhi"

# Output settings
OUTPUT_DIR = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR"
OUTPUT_FILENAME = "delhi_gkg_data_2021_jan1_3.csv"

# Add this line with your other output settings
BATCH_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "raw_data", "batch_outputs")

# Request timeout (seconds)
TIMEOUT = 30
# ===================================

# Update this line
INPUT_FILE = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR\raw_data\delhi_gkg_data_2021_jan1_3.csv"

CACHE_DIR = os.path.join(OUTPUT_DIR, "raw_data", "cache")

def get_cache_path(url):
    """Generate a cache file path for a URL"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{url_hash}.pkl")

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

def download_with_retry(url, max_retries=3):
    """Download a URL with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                return response
            elif response.status_code == 404:
                # File doesn't exist, no need to retry
                print(f"File not found: {url}")
                return None
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Attempt {attempt+1}/{max_retries} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    print(f"Failed to download {url} after {max_retries} attempts")
    return None

def download_and_filter_gkg(url, timestamp, location_filter):
    """Download GKG file and filter for entries containing the specified location."""
    # Check cache first
    cache_path = get_cache_path(url)
    if os.path.exists(cache_path):
        try:
            return pd.read_pickle(cache_path)
        except Exception as e:
            print(f"Error loading cached file: {e}. Proceeding with download.")
    
    try:
        response = download_with_retry(url)
        if response is None:
            return None
        
        # Extract ZIP file content
        z = zipfile.ZipFile(io.BytesIO(response.content))
        filename = f"{timestamp}.gkg.csv"
        
        # Ensure the file exists in the ZIP archive
        if filename not in z.namelist():
            print(f"Error: {filename} not found in the ZIP archive.")
            return None
        
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
        
        # Cache the filtered results
        if filtered_df is not None and not filtered_df.empty:
            filtered_df.to_pickle(cache_path)
        
        return filtered_df
    
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None

def main():
    """Process GDELT GKG data for the configured date range and location filter"""
    # Create directory to store the data if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Get URLs for the date range
    urls = get_gdelt_gkg_urls(START_DATE, END_DATE)
    total_urls = len(urls)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Found {total_urls} GDELT GKG files to process")
    
    # Download and filter data
    filtered_data = []
    start_time = datetime.now()
    
    # Use ThreadPoolExecutor for parallel downloads
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Create a list to track futures
        future_to_url = {executor.submit(download_and_filter_gkg, url, timestamp, LOCATION_FILTER): 
                         (url, timestamp) for url, timestamp in urls}
        
        # Process results as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_url), 
                          total=len(future_to_url), 
                          desc="Downloading files"):
            url, timestamp = future_to_url[future]
            try:
                filtered_df = future.result()
                if filtered_df is not None and not filtered_df.empty:
                    filtered_data.append(filtered_df)
            except Exception as e:
                print(f"Error processing {url}: {e}")
    
    # Calculate elapsed time
    elapsed_time = datetime.now() - start_time
    
    # Combine all filtered data
    if filtered_data:
        combined_df = pd.concat(filtered_data, ignore_index=True)
        
        # Save to CSV
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
        combined_df.to_csv(output_path, index=False)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved {len(combined_df)} entries to {output_path}")
        print(f"Processing completed in {elapsed_time}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] No data found for {LOCATION_FILTER} in the specified date range.")
        print(f"Processing completed in {elapsed_time}")

# Process data in 3-month chunks
def process_in_batches():
    """Process GDELT GKG data in 3-month batches"""
    # Create batch output directory if it doesn't exist
    if not os.path.exists(BATCH_OUTPUT_DIR):
        os.makedirs(BATCH_OUTPUT_DIR)
        
    start = datetime(2021, 1, 1)
    end_date = datetime(2024, 12, 12)
    
    while start < end_date:
        # Calculate batch end (3 months later)
        batch_end = start + timedelta(days=90)
        # Ensure we don't go past the final end date
        if batch_end > end_date:
            batch_end = end_date
            
        print(f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing batch: {start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}")
        
        # Update the output filename for this batch - save in the batch directory
        global OUTPUT_FILENAME
        batch_filename = f"delhi_gkg_data_{start.strftime('%Y%m%d')}_to_{batch_end.strftime('%Y%m%d')}.csv"
        OUTPUT_FILENAME = os.path.join("batch_outputs", batch_filename)
        
        # Set the date range for this batch
        global START_DATE, END_DATE
        START_DATE = start
        END_DATE = batch_end
        
        # Run the main processing
        main()
        
        # Move to next batch
        start = batch_end + timedelta(days=1)

def find_last_completed_batch():
    """Find the latest date that has been processed based on existing files.
    
    Used by the resumable batch processing to determine where to start from.
    Returns None if no completed batches are found.
    """
    if not os.path.exists(BATCH_OUTPUT_DIR):
        return None
        
    batch_files = [f for f in os.listdir(BATCH_OUTPUT_DIR) if f.endswith('.csv')]
    if not batch_files:
        return None
        
    # Extract end dates from filenames using regex
    import re
    end_dates = []
    for filename in batch_files:
        match = re.search(r'_to_(\d{8})\.csv', filename)
        if match:
            date_str = match.group(1)
            try:
                end_date = datetime.strptime(date_str, '%Y%m%d')
                end_dates.append(end_date)
            except ValueError:
                continue
    
    if not end_dates:
        print("Warning: Found batch files but could not extract dates from filenames.")
        return None
        
    # Return the latest end date found
    return max(end_dates)

def process_in_batches_with_resume():
    """Process GDELT GKG data in 3-month batches, resuming from the last processed batch"""
    # Create batch output directory if it doesn't exist
    if not os.path.exists(BATCH_OUTPUT_DIR):
        os.makedirs(BATCH_OUTPUT_DIR)
    
    # Find the last completed batch
    last_completed_date = find_last_completed_batch()
    
    # Set the start date based on whether we're resuming or starting fresh
    if last_completed_date:
        # Resume from the day after the last completed batch
        start = last_completed_date + timedelta(days=1)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Resuming from {start.strftime('%Y-%m-%d')} (previous batch ended at {last_completed_date.strftime('%Y-%m-%d')})")
    else:
        # Start from the beginning
        start = datetime(2021, 1, 1)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting new batch processing from {start.strftime('%Y-%m-%d')}")
        
    end_date = datetime(2024, 12, 12)
    
    # Process in batches from the determined start date
    while start < end_date:
        # Calculate batch end (3 months later)
        batch_end = start + timedelta(days=90)
        # Ensure we don't go past the final end date
        if batch_end > end_date:
            batch_end = end_date
            
        print(f"\n\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing batch: {start.strftime('%Y-%m-%d')} to {batch_end.strftime('%Y-%m-%d')}")
        
        # Update the output filename for this batch - save in the batch directory
        global OUTPUT_FILENAME
        batch_filename = f"delhi_gkg_data_{start.strftime('%Y%m%d')}_to_{batch_end.strftime('%Y%m%d')}.csv"
        OUTPUT_FILENAME = os.path.join("batch_outputs", batch_filename)
        
        # Set the date range for this batch
        global START_DATE, END_DATE
        START_DATE = start
        END_DATE = batch_end
        
        # Run the main processing
        main()
        
        # Move to next batch
        start = batch_end + timedelta(days=1)

def cleanup_cache(delete_all=False):
    """Clean up cache files
    
    Args:
        delete_all (bool): If True, removes all cache files. 
                         If False, only removes empty files.
    """
    if not os.path.exists(CACHE_DIR):
        return
        
    if delete_all:
        print(f"Removing all cache files from {CACHE_DIR}")
        for file in os.listdir(CACHE_DIR):
            os.remove(os.path.join(CACHE_DIR, file))
        print(f"Cache cleanup complete")
    else:
        # Remove only empty cache files
        removed = 0
        for file in os.listdir(CACHE_DIR):
            path = os.path.join(CACHE_DIR, file)
            if os.path.getsize(path) == 0:
                os.remove(path)
                removed += 1
        print(f"Removed {removed} empty cache files")

if __name__ == "__main__":
    process_in_batches_with_resume()
    # process_in_batches()  # Comment this out when using the resume version
    # main()  # Comment this out when using batch processing
    # cleanup_cache()  # Uncomment to clean up empty cache files
    # cleanup_cache(True)  # Uncomment to delete all cache files