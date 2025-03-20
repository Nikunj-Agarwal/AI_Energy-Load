import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
from tqdm import tqdm

# ADDED - Import from config file
from config import (
    PATHS, THEME_CATEGORIES, setup_and_verify, test_directory_writing as config_test_directory_writing
)

#==============================
# CONFIGURATION VARIABLES
#==============================

# File paths - UPDATED TO USE CONFIG
INPUT_FILE_PATH = os.path.join(PATHS["RAW_DATA_DIR"], "delhi_gkg_data_2021_jan1_3.csv")
OUTPUT_FILE_PATH = os.path.join(PATHS["PROCESSED_DIR"], "processed_gkg_parsed_data.csv")

# Batch processing settings - UPDATED TO USE CONFIG
BATCH_DIR = PATHS["BATCH_DIR"]
MERGED_OUTPUT_PATH = os.path.join(PATHS["PROCESSED_DIR"], "merged_delhi_gkg_data.csv")
PROCESS_BATCHES = True  # Set to True to process batch files instead of single file

# Processing options
SAMPLE_SIZE = None  # Set to None for full dataset processing

# Theme categories and keywords - UPDATED TO USE CONFIG
theme_categories = THEME_CATEGORIES

# Tone metrics to extract
TONE_METRICS = ['tone', 'positive', 'negative', 'polarity', 'activity', 'self_ref']

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
            df = pd.read_csv(file_path)
            dfs.append(df)
            total_rows += len(df)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
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
    
    # Save the merged data
    merged_df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")
    
    return merged_df

def process_dataframe(df):
    """Process a dataframe with all the parsing functions"""
    print(f"Processing dataframe with {len(df)} rows...")
    
    # Sample a subset for testing if requested
    if SAMPLE_SIZE:
        sample_size = min(SAMPLE_SIZE, len(df))
        df = df.head(sample_size)
        print(f"Using sample of {sample_size} rows")
    
    # Convert DATE to datetime
    print("Converting dates...")
    df['datetime'] = df['DATE'].apply(convert_date)
    
    # Process themes
    print("Processing themes...")
    df['parsed_themes'] = df['V2Themes'].apply(parse_themes)
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
# MAIN PROCESSING
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
    # ADDED - First verify directory structure using config function
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        return
    
    # Test directory access
    if not test_directory_access():
        print("ERROR: Directory access test failed. Aborting processing.")
        return
        
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    # Load the data
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
        print("\nSample data:")
        print(df.head(2))
        print("\nColumn names:", df.columns.tolist())
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Process the dataframe
    processed_df = process_dataframe(df)
    
    # Save the processed data
    processed_df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to {output_path}")

def process_batch_files():
    """Merge and process all batch files"""
    # ADDED - First verify directory structure using config function
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        return
    
    # Test directory access
    if not test_directory_access():
        print("ERROR: Directory access test failed. Aborting batch processing.")
        return
        
    # First merge all batch files
    merged_df = merge_batch_files()
    
    if merged_df is None:
        print("Batch processing failed - no data to process.")
        return
    
    # Process the merged dataframe
    processed_df = process_dataframe(merged_df)
    
    # Save the processed data
    processed_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f"\nProcessed merged data saved to {OUTPUT_FILE_PATH}")

# Main execution
if __name__ == "__main__":
    print(f"GDELT GKG Data Sparsing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ADDED - First verify directory structure using config function
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        exit(1)
    
    if PROCESS_BATCHES:
        print("Mode: Processing all batch files")
        process_batch_files()
    else:
        print(f"Mode: Processing single file: {INPUT_FILE_PATH}")
        process_single_file()
    
    print(f"Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")