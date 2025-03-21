import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from tqdm import tqdm

# Import from config file
from config import (
    PATHS, THEME_CATEGORIES, setup_and_verify, test_directory_writing as config_test_directory_writing
)

#==============================
# CONFIGURATION VARIABLES
#==============================

# File paths from config
INPUT_FILE_PATH = os.path.join(PATHS["RAW_DATA_DIR"], "delhi_gkg_data_2021_jan1_3.csv")
OUTPUT_FILE_PATH = os.path.join(PATHS["PROCESSED_DIR"], "processed_gkg_parsed_data.csv")

# Batch processing settings
BATCH_DIR = PATHS["BATCH_DIR"]
MERGED_OUTPUT_PATH = os.path.join(PATHS["PROCESSED_DIR"], "merged_delhi_gkg_data.csv")
PROCESS_BATCHES = True  # Set to True to process batch files instead of single file

# Processing options
SAMPLE_SIZE = None  # Set to None for full dataset processing

# Theme categories and keywords
theme_categories = THEME_CATEGORIES

# Tone metrics to extract
TONE_METRICS = ['tone', 'positive', 'negative', 'polarity', 'activity', 'self_ref']

#==============================
# DATA VALIDATION FUNCTIONS - NEW!
#==============================

def validate_input_data(df, required_columns=None):
    """Validate that input data has the required columns and structure"""
    if required_columns is None:
        required_columns = ['DATE', 'V2Themes', 'V2Tone', 'Counts', 'Amounts']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"ERROR: Input data missing required columns: {missing_columns}")
        return False
    
    return True

def validate_output_data(df):
    """Verify the processed data has all fields required by aggregation script"""
    expected_theme_columns = [f'theme_{category}' for category in theme_categories]
    expected_tone_columns = [f'tone_{metric}' for metric in TONE_METRICS]
    
    required_columns = ['datetime', 'GKGRECORDID'] + expected_theme_columns + expected_tone_columns
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"WARNING: Processed data missing expected columns: {missing_columns}")
        print("This may cause issues in downstream aggregation.")
        return False
    
    return True

def ensure_theme_category_consistency():
    """Ensure theme categories are properly capitalized and consistent"""
    global theme_categories
    
    # Make a copy with properly capitalized keys
    updated_categories = {}
    for category, keywords in theme_categories.items():
        # Capitalize first letter of category name for consistency
        updated_key = category.capitalize()
        updated_categories[updated_key] = keywords
    
    theme_categories = updated_categories
    return theme_categories

#==============================
# HELPER FUNCTIONS
#==============================

def convert_date(date_str):
    """Convert GDELT date format to datetime"""
    try:
        date_str = str(date_str).zfill(14)  # Ensure 14 digits
        year = int(date_str[0:4])
        month = int(date_str[4:6])
        day = int(date_str[6:8])
        hour = int(date_str[8:10])
        minute = int(date_str[10:12])
        second = int(date_str[12:14])
        return datetime(year, month, day, hour, minute, second)
    except:
        return None

def parse_themes(theme_str):
    """Extract themes from V2Themes column"""
    if pd.isna(theme_str):
        return {}
    
    themes_dict = {}
    themes = theme_str.split(';')
    
    for theme in themes:
        if not theme:
            continue
        
        parts = theme.split(',')
        if len(parts) >= 1:
            theme_name = parts[0].lower()
            themes_dict[theme_name] = 1
    
    return themes_dict

def categorize_themes(themes_dict):
    """Map individual themes to theme categories"""
    categorized = {category: 0 for category in theme_categories}
    
    for theme in themes_dict:
        for category, keywords in theme_categories.items():
            if any(keyword.lower() in theme.lower() for keyword in keywords):
                categorized[category] = 1
                break
    
    return categorized

def parse_tone(tone_str):
    """Extract tone metrics from V2Tone column"""
    if pd.isna(tone_str):
        return {metric: 0 for metric in TONE_METRICS}
    
    values = tone_str.split(',')
    
    if len(values) < 6:
        return {metric: 0 for metric in TONE_METRICS}
    
    try:
        return {
            'tone': float(values[0]),
            'positive': float(values[1]),
            'negative': float(values[2]),
            'polarity': float(values[3]),
            'activity': float(values[4]),
            'self_ref': float(values[5])
        }
    except ValueError:
        # Handle conversion errors
        return {metric: 0 for metric in TONE_METRICS}

def parse_counts(counts_str):
    """Extract entity counts"""
    if pd.isna(counts_str):
        return {}
    
    counts_dict = {}
    counts = counts_str.split(';')
    
    for count in counts:
        if not count:
            continue
        
        parts = count.split('#')
        if len(parts) >= 2:
            entity = parts[0].lower()
            try:
                count_value = int(parts[1])
                counts_dict[entity] = count_value
            except:
                pass
    
    return counts_dict

def parse_amounts(amounts_str):
    """Extract numerical amounts mentioned in text"""
    if pd.isna(amounts_str):
        return []
    
    amounts = []
    amount_entries = amounts_str.split(';')
    
    for entry in amount_entries:
        if not entry:
            continue
        
        try:
            amount_value = float(entry.split(',')[0])
            amounts.append(amount_value)
        except:
            pass
    
    return amounts

#==============================
# BATCH PROCESSING FUNCTIONS
#==============================

def merge_batch_files(batch_dir=BATCH_DIR, output_path=MERGED_OUTPUT_PATH):
    """Merge all batch CSV files into a single file"""
    print(f"Looking for batch files in {batch_dir}...")
    
    # Find all CSV files in the batch directory
    batch_files = glob.glob(os.path.join(batch_dir, "*.csv"))
    
    if not batch_files:
        print("Error: No batch files found!")
        return None
    
    print(f"Found {len(batch_files)} batch files. Merging...")
    
    # Read and concatenate all batch files
    dfs = []
    total_rows = 0
    
    for file_path in tqdm(batch_files, desc="Reading batch files"):
        try:
            # Add encoding parameter
            df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
            dfs.append(df)
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            try:
                # Fallback to latin-1 encoding
                print(f"Retrying with latin-1 encoding...")
                df = pd.read_csv(file_path, encoding='latin-1')
                dfs.append(df)
                total_rows += len(df)
            except Exception as e2:
                print(f"Failed with latin-1 encoding too: {e2}")
    
    if not dfs:
        print("No valid data found in batch files!")
        return None
    
    # Concatenate all dataframes
    try:
        merged_df = pd.concat(dfs, ignore_index=True)
    except ValueError as e:
        print(f"Error merging batch files: {e}")
        return None
    
    print(f"Merged {len(batch_files)} files with a total of {total_rows} rows.")
    
    # Add encoding parameter to the save operation
    merged_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Merged data saved to {output_path}")
    
    return merged_df

#==============================
# ENHANCED PROCESS DATAFRAME FUNCTION
#==============================

from functools import lru_cache

# Add caching to expensive functions
@lru_cache(maxsize=10000)
def parse_themes_cached(theme_str):
    """Cached version of parse_themes for better performance"""
    return parse_themes(theme_str)

def process_dataframe(df):
    """Process a dataframe with all the parsing functions"""
    print(f"Processing dataframe with {len(df)} rows...")
    
    # Validate input data first - NEW!
    if not validate_input_data(df):
        print("ERROR: Input data validation failed. Processing may produce incomplete results.")
    
    # Sample a subset for testing if requested
    if SAMPLE_SIZE:
        sample_size = min(SAMPLE_SIZE, len(df))
        df = df.head(sample_size)
        print(f"Using sample of {sample_size} rows")
    
    # Convert DATE to datetime
    print("Converting dates...")
    df['datetime'] = df['DATE'].apply(convert_date)
    
    # Handle invalid dates - NEW!
    invalid_dates = df['datetime'].isna().sum()
    if (invalid_dates > 0):
        print(f"WARNING: {invalid_dates} rows with invalid dates. These will be excluded.")
        df = df.dropna(subset=['datetime'])
    
    # Ensure consistent theme categories - NEW!
    ensure_theme_category_consistency()
    
    # Process themes with caching
    print("Processing themes with caching...")
    df['parsed_themes'] = df['V2Themes'].apply(lambda x: parse_themes_cached(str(x)) if pd.notna(x) else {})
    df['theme_categories'] = df['parsed_themes'].apply(categorize_themes)
    
    # Create columns for each theme category
    for category in theme_categories:
        df[f'theme_{category}'] = df['theme_categories'].apply(
            lambda x: x.get(category, 0) if isinstance(x, dict) else 0
        )
    
    # Process tone metrics
    print("Processing tone metrics...")
    df['parsed_tone'] = df['V2Tone'].apply(parse_tone)
    for metric in TONE_METRICS:
        df[f'tone_{metric}'] = df['parsed_tone'].apply(lambda x: x.get(metric, 0))
    
    # Process counts and amounts
    print("Processing counts and amounts...")
    df['parsed_counts'] = df['Counts'].apply(parse_counts)
    df['entity_count'] = df['parsed_counts'].apply(lambda x: sum(x.values()) if isinstance(x, dict) else 0)
    df['entity_variety'] = df['parsed_counts'].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    
    df['parsed_amounts'] = df['Amounts'].apply(parse_amounts)
    df['avg_amount'] = df['parsed_amounts'].apply(lambda x: np.mean(x) if x else 0)
    df['max_amount'] = df['parsed_amounts'].apply(lambda x: max(x) if x else 0)
    df['amount_count'] = df['parsed_amounts'].apply(lambda x: len(x) if x else 0)
    
    # Validate the output data - NEW!
    validate_output_data(df)
    
    # Print some statistics about the parsed data
    print("\nTheme distribution in data:")
    for category in theme_categories:
        count = df[f'theme_{category}'].sum()
        print(f"{category}: {count}")
    
    print("\nTone statistics:")
    print(f"Average tone: {df['tone_tone'].mean()}")
    print(f"Most positive: {df['tone_positive'].max()}")
    print(f"Most negative: {df['tone_negative'].max()}")
    
    return df

#==============================
# IMPROVED MAIN PROCESSING FUNCTIONS
#==============================

def test_directory_access():
    """Test if we can write to all required directories"""
    print("Testing directory access and file writing permissions...")
    
    test_dirs = [
        os.path.dirname(INPUT_FILE_PATH),  # ADDED input directory
        os.path.dirname(OUTPUT_FILE_PATH),
        os.path.dirname(MERGED_OUTPUT_PATH),
        BATCH_DIR
    ]
    
    all_passed = True
    for directory in test_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"ERROR: Could not create directory {directory}: {e}")
                all_passed = False
                continue
        
        test_file = os.path.join(directory, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write(f"Test write at {datetime.now()}")
            os.remove(test_file)
            print(f"✓ Successfully wrote to {directory}")
        except Exception as e:
            print(f"✗ ERROR: Could not write to {directory}: {e}")
            all_passed = False
    
    # ADDED - Test DataFrame writing to output directories
    test_df = pd.DataFrame({'test': [1, 2, 3]})
    
    # Test output directory
    try:
        test_csv = os.path.join(os.path.dirname(OUTPUT_FILE_PATH), "test_dataframe.csv")
        test_df.to_csv(test_csv, index=False)
        os.remove(test_csv)
        print(f"✓ Successfully wrote DataFrame to {os.path.dirname(OUTPUT_FILE_PATH)}")
    except Exception as e:
        print(f"✗ ERROR: Could not write DataFrame to {os.path.dirname(OUTPUT_FILE_PATH)}: {e}")
        all_passed = False
    
    # Test merged output directory
    try:
        test_csv = os.path.join(os.path.dirname(MERGED_OUTPUT_PATH), "test_dataframe.csv")
        test_df.to_csv(test_csv, index=False)
        os.remove(test_csv)
        print(f"✓ Successfully wrote DataFrame to {os.path.dirname(MERGED_OUTPUT_PATH)}")
    except Exception as e:
        print(f"✗ ERROR: Could not write DataFrame to {os.path.dirname(MERGED_OUTPUT_PATH)}: {e}")
        all_passed = False
    
    return all_passed

def process_single_file(input_path=INPUT_FILE_PATH, output_path=OUTPUT_FILE_PATH):
    """Process a single GKG data file"""
    # Verify directory structure using config function
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        return
    
    # Test directory access
    if not test_directory_access():
        print("ERROR: Directory access test failed. Aborting processing.")
        return
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data with encoding handling
    print(f"Loading data from {input_path}...")
    try:
        # First try with utf-8 and error replacement
        df = pd.read_csv(input_path, encoding='utf-8', encoding_errors='replace')
        print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error with utf-8 encoding, trying latin-1: {e}")
        try:
            # Fall back to latin-1 which can handle any byte sequence
            df = pd.read_csv(input_path, encoding='latin-1')
            print(f"Data loaded with latin-1 encoding: {df.shape[0]} rows")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    # Process the dataframe
    processed_df = process_dataframe(df)
    
    # Verify we have data before saving - NEW!
    if processed_df is None or len(processed_df) == 0:
        print("ERROR: Processing resulted in empty dataframe. No output will be saved.")
        return
    
    # Save the processed data with improved error handling - NEW!
    try:
        processed_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\nProcessed data saved to {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save output to {output_path}: {e}")

def process_batch_files():
    """Merge and process all batch files"""
    # Verify directory structure using config function
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        return
    
    # Test directory access
    if not test_directory_access():
        print("ERROR: Directory access test failed. Aborting batch processing.")
        return
    
    # First merge all batch files
    print(f"Merging batch files from {BATCH_DIR}...")
    merged_df = merge_batch_files()
    
    if merged_df is None:
        print("ERROR: Batch file merging failed. No data to process.")
        return
        
    if len(merged_df) == 0:
        print("ERROR: Merged data is empty. No data to process.")
        return
    
    print(f"Successfully merged batch files. Processing {len(merged_df)} rows...")
    
    # Process the merged dataframe
    processed_df = process_dataframe(merged_df)
    
    # Verify we have data before saving
    if processed_df is None or len(processed_df) == 0:
        print("ERROR: Processing resulted in empty dataframe. No output will be saved.")
        return
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the processed data with improved error handling
    try:
        processed_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8')
        print(f"\nProcessed merged data saved to {OUTPUT_FILE_PATH}")
        
        # Output some statistics about the processed data
        print(f"Processed {len(processed_df)} rows with {processed_df.shape[1]} columns")
        theme_columns = [col for col in processed_df.columns if col.startswith('theme_')]
        print(f"Extracted {len(theme_columns)} theme categories")
        tone_columns = [col for col in processed_df.columns if col.startswith('tone_')]
        print(f"Extracted {len(tone_columns)} tone metrics")
    except Exception as e:
        print(f"ERROR: Failed to save output to {OUTPUT_FILE_PATH}: {e}")

def process_batch_files_in_chunks():
    """Process batch files with memory-efficient chunking"""
    # Verify directories first
    if not setup_and_verify() or not test_directory_access():
        return
    
    # Find all batch files
    batch_files = glob.glob(os.path.join(BATCH_DIR, "*.csv"))
    if not batch_files:
        print("Error: No batch files found!")
        return
    
    print(f"Found {len(batch_files)} batch files")
    
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    
    # First chunk needs to create the file, others will append
    first_chunk = True
    
    for file_idx, file_path in enumerate(batch_files):
        print(f"\nProcessing batch file {file_idx+1}/{len(batch_files)}: {os.path.basename(file_path)}")
        
        # Process in chunks to save memory
        chunk_size = 50000  # Adjust based on your system's memory
        
        try:
            for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, encoding='utf-8')):
                print(f"  Processing chunk {chunk_idx+1} ({len(chunk)} rows)")
                
                # Process this chunk
                processed_chunk = process_dataframe(chunk)
                
                if processed_chunk is not None and len(processed_chunk) > 0:
                    # Write directly to file with UTF-8 encoding
                    if first_chunk:
                        # First chunk creates the file with headers
                        processed_chunk.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8')
                        first_chunk = False
                    else:
                        # Subsequent chunks append without headers
                        processed_chunk.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, 
                                              index=False, encoding='utf-8')
                    
                    print(f"    Wrote {len(processed_chunk)} rows to output file")
                        
                # Clear memory
                del processed_chunk
                del chunk
        except Exception as e:
            try:
                print(f"  Error with UTF-8 encoding, trying latin-1: {e}")
                for chunk_idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, encoding='latin-1')):
                    print(f"  Processing chunk {chunk_idx+1} ({len(chunk)} rows) with latin-1 encoding")
                    
                    # Process this chunk
                    processed_chunk = process_dataframe(chunk)
                    
                    if processed_chunk is not None and len(processed_chunk) > 0:
                        # Write directly to file with UTF-8 encoding
                        if first_chunk:
                            # First chunk creates the file with headers
                            processed_chunk.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8')
                            first_chunk = False
                        else:
                            # Subsequent chunks append without headers
                            processed_chunk.to_csv(OUTPUT_FILE_PATH, mode='a', header=False, 
                                                  index=False, encoding='utf-8')
                        
                        print(f"    Wrote {len(processed_chunk)} rows to output file")
                            
                    # Clear memory
                    del processed_chunk
                    del chunk
            except Exception as e2:
                print(f"  Error processing file {os.path.basename(file_path)}: {e2}")
                print("  Continuing with next file...")
    
    print(f"Processing complete. Output saved to {OUTPUT_FILE_PATH}")

# Add to gkg_data_sparsing.py
def log_data_quality_metrics(df, output_dir):
    """Log data quality metrics to help identify issues"""
    log_path = os.path.join(output_dir, "data_quality_log.txt")
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"GDELT Data Quality Report - {datetime.now()}\n")
        f.write("="*80 + "\n\n")
        
        # Date range
        f.write(f"Date Range: {df['datetime'].min()} to {df['datetime'].max()}\n\n")
        
        # Record count
        f.write(f"Total Records: {len(df)}\n")
        
        # Missing values
        null_counts = df.isnull().sum()
        f.write("\nMissing Values:\n")
        for col, count in null_counts.items():
            if count > 0:
                f.write(f"  {col}: {count} ({count/len(df)*100:.2f}%)\n")
        
        # Theme distribution
        f.write("\nTheme Distribution:\n")
        for category in theme_categories:
            count = df[f'theme_{category}'].sum()
            f.write(f"  {category}: {count} ({count/len(df)*100:.2f}%)\n")
        
        # Tone statistics
        f.write("\nTone Statistics:\n")
        f.write(f"  Average Tone: {df['tone_tone'].mean():.4f}\n")
        f.write(f"  Tone Range: {df['tone_tone'].min():.4f} to {df['tone_tone'].max():.4f}\n")
        f.write(f"  Most Positive: {df['tone_positive'].max():.4f}\n")
        f.write(f"  Most Negative: {df['tone_negative'].max():.4f}\n")
        
        # Potential anomalies
        extreme_tone = len(df[df['tone_tone'].abs() > 10])
        if extreme_tone > 0:
            f.write(f"\nPotential Anomalies:\n")
            f.write(f"  Records with extreme tone (>10): {extreme_tone}\n")
        
        no_themes = len(df[(df[[f'theme_{c}' for c in theme_categories]].sum(axis=1) == 0)])
        if no_themes > 0:
            f.write(f"  Records with no themes detected: {no_themes}\n")
    
    print(f"Data quality report saved to {log_path}")
    return log_path

# Add to gkg_data_sparsing.py
def process_incremental():
    """Process only new batch files that haven't been processed yet"""
    if not setup_and_verify() or not test_directory_access():
        return
    
    # Get list of batch files
    batch_files = glob.glob(os.path.join(BATCH_DIR, "*.csv"))
    if not batch_files:
        print("No batch files found.")
        return
    
    # Check if output file exists
    if os.path.exists(OUTPUT_FILE_PATH):
        # Load existing processed data with encoding
        try:
            existing_df = pd.read_csv(OUTPUT_FILE_PATH, encoding='utf-8', encoding_errors='replace')
            existing_ids = set(existing_df['GKGRECORDID']) if 'GKGRECORDID' in existing_df.columns else set()
            print(f"Found existing output with {len(existing_ids)} records")
        except Exception as e:
            print(f"Error reading existing output file: {e}")
            try:
                # Try with latin-1 encoding
                existing_df = pd.read_csv(OUTPUT_FILE_PATH, encoding='latin-1')
                existing_ids = set(existing_df['GKGRECORDID']) if 'GKGRECORDID' in existing_df.columns else set()
                print(f"Found existing output with {len(existing_ids)} records (using latin-1 encoding)")
            except:
                print("Error reading existing output file. Will process all files.")
                existing_ids = set()
    else:
        existing_ids = set()
    
    # Process new files incrementally
    for file_idx, file_path in enumerate(batch_files):
        filename = os.path.basename(file_path)
        print(f"Checking file {file_idx+1}/{len(batch_files)}: {filename}")
        
        # Read file with encoding
        try:
            batch_df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
            # Filter to only new records
            if 'GKGRECORDID' in batch_df.columns:
                new_records = batch_df[~batch_df['GKGRECORDID'].isin(existing_ids)]
                if len(new_records) == 0:
                    print(f"  No new records in {filename}, skipping")
                    continue
                print(f"  Found {len(new_records)} new records in {filename}")
                
                # Process new records
                processed_new = process_dataframe(new_records)
                
                # Append to existing output with encoding
                if processed_new is not None and len(processed_new) > 0:
                    processed_new.to_csv(OUTPUT_FILE_PATH, mode='a', 
                                         header=not os.path.exists(OUTPUT_FILE_PATH), 
                                         index=False, encoding='utf-8')
                    print(f"  Appended {len(processed_new)} processed records to output")
                    
                    # Update existing IDs
                    existing_ids.update(processed_new['GKGRECORDID'])
            else:
                print(f"  Warning: File {filename} doesn't have GKGRECORDID column")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
    
    print("Incremental processing complete")

# Main execution
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDELT GKG Data Sparsing")
    parser.add_argument("--mode", choices=["single", "batch", "incremental", "chunks"], 
                        default="chunks", help="Processing mode")
    parser.add_argument("--input", help="Input file path (for single mode)")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--sample", type=int, help="Sample size for testing")
    
    args = parser.parse_args()
    
    print(f"GDELT GKG Data Sparsing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Override global settings with command line args
    if args.input:
        INPUT_FILE_PATH = args.input
    if args.output:
        OUTPUT_FILE_PATH = args.output
    if args.sample:
        SAMPLE_SIZE = args.sample
    
    # Choose processing mode
    if args.mode == "single":
        print(f"Mode: Processing single file: {INPUT_FILE_PATH}")
        process_single_file()
    elif args.mode == "batch":
        print("Mode: Processing all batch files")
        process_batch_files()
    elif args.mode == "incremental":
        print("Mode: Incremental processing of new batch files")
        process_incremental()
    elif args.mode == "chunks":
        print("Mode: Processing batch files in memory-efficient chunks")
        process_batch_files_in_chunks()
    
    print(f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")