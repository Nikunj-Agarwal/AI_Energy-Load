import pandas as pd
import numpy as np
from datetime import datetime

#==============================
# CONFIGURATION VARIABLES
#==============================

# File paths
INPUT_FILE_PATH = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets\delhi_gkg_data_2021_jan1_3.csv"
OUTPUT_FILE_PATH = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets\processed_gkg_parsed_data.csv"

# Processing options
SAMPLE_SIZE = 10000  # Set to None for full dataset processing

# Theme categories and keywords
theme_categories = {
    'Health': ['health', 'medic', 'disease', 'hospital', 'covid', 'vaccine', 'pandemic'],
    'Political': ['government', 'election', 'politic', 'policy', 'minister', 'president', 'parliament'],
    'Economic': ['econom', 'market', 'trade', 'business', 'financ', 'tax', 'invest'],
    'Education': ['education', 'school', 'university', 'student', 'learning', 'college'],
    'Infrastructure': ['infrastructure', 'construction', 'building', 'transport', 'road', 'highway'],
    'Social': ['social', 'community', 'society', 'people', 'public', 'citizen'],
    'Religious': ['religion', 'religious', 'temple', 'church', 'mosque', 'faith', 'god'],
    'Environment': ['environment', 'climate', 'pollution', 'water', 'ecology', 'green'],
    'Energy': ['energy', 'power', 'electricity', 'fuel', 'oil', 'gas', 'coal']
}

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
    
    return {
        'tone': float(values[0]),
        'positive': float(values[1]),
        'negative': float(values[2]),
        'polarity': float(values[3]),
        'activity': float(values[4]),
        'self_ref': float(values[5])
    }

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
# MAIN PROCESSING
#==============================

# Load the data
print("Loading data...")
df = pd.read_csv(INPUT_FILE_PATH)
print(f"Data loaded with {df.shape[0]} rows and {df.shape[1]} columns")
print("\nSample data:")
print(df.head(2))
print("\nColumn names:", df.columns.tolist())

# Process the data
print("\nProcessing the data...")

# Sample a subset for testing (comment out for full processing)
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

# Clean up intermediate columns if desired
columns_to_drop = ['parsed_themes', 'theme_categories', 'parsed_tone', 
                   'parsed_counts', 'parsed_amounts']

# Uncomment to remove intermediate processing columns
# df = df.drop(columns=columns_to_drop)

# Save the processed data
df.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\nProcessed data saved to {OUTPUT_FILE_PATH}")