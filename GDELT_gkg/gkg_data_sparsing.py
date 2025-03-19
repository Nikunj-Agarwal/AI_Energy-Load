import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

#==============================
# CONFIGURATION VARIABLES
#==============================

# File paths
INPUT_FILE_PATH = r"C:\Users\nikun\Desktop\MLPR\Project\ML_trial\GDELT_gkg\gkg_datasets\delhi_gkg_data_2021_jan1_3.csv"
OUTPUT_FILE_PATH = r"C:\Users\nikun\Desktop\MLPR\Project\ML_trial\GDELT_gkg\gkg_datasets\processed_gkg_15min_intervals.csv"
VISUALIZATION_PATH = r"C:\Users\nikun\Desktop\MLPR\Project\ML_trial\GDELT_gkg\gkg_graphs\theme_distribution.png"

# Processing options
SAMPLE_SIZE = 10000  # Set to None for full dataset processing
TIME_WINDOW_MINUTES = 15  # Interval size in minutes

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

# Aggregation metrics to calculate for each column type
# You can add/remove metrics here (e.g., 'median', 'std', etc.)
THEME_AGGREGATIONS = ['sum', 'mean', 'max']
TONE_AGGREGATIONS = ['mean', 'min', 'max']
COUNT_AGGREGATIONS = ['sum', 'mean', 'max']
AMOUNT_AGGREGATIONS = ['mean', 'max']

# Visualization settings
VIZ_FIGSIZE = (12, 6)
VIZ_ROTATION = 45

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

def bin_to_time_window(dt):
    """Convert datetime to specified time interval"""
    if dt is None:
        return None
    minute = (dt.minute // TIME_WINDOW_MINUTES) * TIME_WINDOW_MINUTES
    return datetime(dt.year, dt.month, dt.day, dt.hour, minute, 0)

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

# Convert DATE to datetime and create time windows
print("Converting dates and creating time windows...")
df['datetime'] = df['DATE'].apply(convert_date)
df['time_window'] = df['datetime'].apply(bin_to_time_window)

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

# Print some statistics before aggregation
print("\nTheme distribution in raw data:")
for category in theme_categories:
    count = df[f'theme_{category}'].sum()
    print(f"{category}: {count}")

print("\nTone statistics:")
print(f"Average tone: {df['tone_tone'].mean()}")
print(f"Most positive: {df['tone_positive'].max()}")
print(f"Most negative: {df['tone_negative'].max()}")

# Group by time windows
print(f"\nAggregating by {TIME_WINDOW_MINUTES}-minute intervals...")
time_window_groups = df.groupby('time_window')

# Set up aggregation functions
agg_functions = {}

# Event count
agg_functions['GKGRECORDID'] = ['count']

# Theme aggregations
for category in theme_categories:
    col = f'theme_{category}'
    agg_functions[col] = THEME_AGGREGATIONS

# Tone aggregations
for metric in TONE_METRICS:
    col = f'tone_{metric}'
    agg_functions[col] = TONE_AGGREGATIONS

# Count and amount aggregations
agg_functions['entity_count'] = COUNT_AGGREGATIONS
agg_functions['entity_variety'] = COUNT_AGGREGATIONS
agg_functions['avg_amount'] = AMOUNT_AGGREGATIONS
agg_functions['max_amount'] = AMOUNT_AGGREGATIONS
agg_functions['amount_count'] = ['sum', 'mean']

# Create aggregated dataframe
df_agg = time_window_groups.agg(agg_functions)

# Flatten the multi-level column names
df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
df_agg = df_agg.reset_index()

# Print information about the aggregated dataset
print(f"\nAggregated data shape: {df_agg.shape}")
print("\nSample of aggregated data:")
print(df_agg.head(2))

# Print some statistics
print("\nTime windows with most events:")
most_events = df_agg.nlargest(5, 'GKGRECORDID_count')
print(most_events[['time_window', 'GKGRECORDID_count']])

# Save the processed data
df_agg.to_csv(OUTPUT_FILE_PATH, index=False)
print(f"\nProcessed data saved to {OUTPUT_FILE_PATH}")

# Optional: Create a quick visualization
plt.figure(figsize=VIZ_FIGSIZE)
plt.bar(range(len(theme_categories)), [df_agg[f'theme_{cat}_sum'].mean() for cat in theme_categories])
plt.xticks(range(len(theme_categories)), theme_categories, rotation=VIZ_ROTATION)
plt.title(f'Average Theme Presence in {TIME_WINDOW_MINUTES}-minute Windows')
plt.tight_layout()
plt.savefig(VISUALIZATION_PATH)
print("Theme distribution visualization saved")