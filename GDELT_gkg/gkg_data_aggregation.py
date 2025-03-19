import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Input/output paths
INPUT_PATH = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets\processed_gkg_parsed_data.csv"
OUTPUT_PATH = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets\aggregated_gkg_15min.csv"

# Define key theme categories most likely to affect energy load
ENERGY_THEMES = ['Energy', 'Environment', 'Infrastructure', 'Social', 'Health']

def aggregate_gkg_to_15min(df):
    """
    Aggregate GKG data into 15-minute intervals, focusing on energy-relevant features
    while minimizing feature count.
    """
    # Ensure datetime is in the correct format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create 15-minute time buckets
    df['time_bucket'] = df['datetime'].dt.floor('15min')
    
    # 1. BASIC AGGREGATION
    # -------------------
    # Define aggregation dictionary focusing on the most important features
    agg_dict = {
        # Count of articles in each interval
        'GKGRECORDID': 'count',
        
        # Key theme categories (sum and max to capture prevalence and intensity)
        **{f'theme_{cat}': ['sum', 'max'] for cat in ENERGY_THEMES},
        
        # Tone metrics (focus on average and extremes)
        'tone_tone': ['mean', 'min', 'max'],
        'tone_negative': 'max',  # Capture most negative sentiment
        'tone_positive': 'max',  # Capture most positive sentiment
        
        # Entity metrics (simplified)
        'entity_count': 'sum',   # Total entities mentioned
    }
    
    # Group by time bucket and aggregate
    agg_df = df.groupby('time_bucket').agg(agg_dict)
    
    # Flatten column multi-index
    agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns]
    
    # Rename count column
    agg_df.rename(columns={'GKGRECORDID_count': 'article_count'}, inplace=True)
    
    # 2. FEATURE ENGINEERING
    # ---------------------
    # Create composite features that combine themes with sentiment
    agg_df['energy_crisis_indicator'] = agg_df['theme_Energy_sum'] * agg_df['tone_negative_max'] 
    agg_df['weather_alert_indicator'] = agg_df['theme_Environment_sum'] * np.abs(agg_df['tone_tone_min'])
    agg_df['social_event_indicator'] = agg_df['theme_Social_sum'] * agg_df['article_count'] / 100
    agg_df['infrastructure_stress'] = agg_df['theme_Infrastructure_sum'] * agg_df['tone_negative_max']
    
    # Calculate tone volatility (max - min) as a single metric instead of std
    agg_df['tone_volatility'] = agg_df['tone_tone_max'] - agg_df['tone_tone_min']
    
    # 3. TIME FEATURES
    # --------------
    # Add temporal features
    agg_df = agg_df.reset_index()
    agg_df['hour'] = agg_df['time_bucket'].dt.hour
    agg_df['day_of_week'] = agg_df['time_bucket'].dt.dayofweek
    agg_df['is_weekend'] = agg_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 4. FEATURE CLEANUP AND SELECTION
    # ------------------------------
    # List of important features to keep (customize as needed)
    keep_features = [
        'time_bucket', 'article_count', 
        'tone_tone_mean', 'tone_negative_max', 'tone_positive_max', 'tone_volatility',
        'theme_Energy_sum', 'theme_Environment_sum', 'theme_Infrastructure_sum', 
        'theme_Social_sum', 'theme_Health_sum',
        'energy_crisis_indicator', 'weather_alert_indicator', 
        'social_event_indicator', 'infrastructure_stress',
        'hour', 'day_of_week', 'is_weekend'
    ]
    
    # Keep only the most relevant features
    final_df = agg_df[keep_features]
    
    return final_df

def handle_missing_intervals(df, start_date=None, end_date=None):
    """
    Ensure all 15-minute intervals exist in the time range, filling gaps with zeros.
    This is important for time series analysis.
    """
    # If no dates provided, use min/max from the data
    if start_date is None:
        start_date = df['time_bucket'].min()
    if end_date is None:
        end_date = df['time_bucket'].max()
    
    # Create complete 15-minute range
    complete_range = pd.date_range(start=start_date, end=end_date, freq='15min')
    
    # Create a reference dataframe with all intervals
    ref_df = pd.DataFrame({'time_bucket': complete_range})
    
    # Merge with actual data
    merged_df = pd.merge(ref_df, df, on='time_bucket', how='left')
    
    # Fill missing values with 0 for numeric columns
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].fillna(0)
    
    return merged_df

def analyze_features(df):
    """
    Analyze feature importance and correlations to help with feature selection.
    """
    # Calculate correlations
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    
    # Print highly correlated features
    print("\nHighly correlated features (r > 0.8):")
    high_corr = (corr_matrix.abs() > 0.8) & (corr_matrix != 1.0)
    correlated_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if high_corr.iloc[i, j]:
                correlated_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    for col1, col2, corr in sorted(correlated_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"{col1} & {col2}: {corr:.3f}")
    
    return corr_matrix

def main():
    # Load the parsed data
    print(f"Loading data from {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH, parse_dates=['datetime'])
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Perform aggregation
    print("Aggregating data to 15-minute intervals...")
    agg_df = aggregate_gkg_to_15min(df)
    
    # Handle missing intervals
    print("Ensuring all 15-minute intervals are present...")
    complete_df = handle_missing_intervals(agg_df)
    
    # Analyze features
    print("Analyzing feature relationships...")
    analyze_features(complete_df)
    
    # Save the aggregated data
    print(f"Saving aggregated data to {OUTPUT_PATH}...")
    complete_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Data aggregation complete. Created {len(complete_df)} 15-minute intervals.")
    
    # Print summary statistics
    print("\nSummary of aggregated data:")
    print(complete_df.describe().T)
    
    # Create a time series plot of key indicators
    plt.figure(figsize=(15, 10))
    
    # Select a subset of data points for clarity
    plot_df = complete_df.iloc[::4]  # Every 4th row (1-hour intervals for cleaner plot)
    
    # Plot article count
    plt.subplot(3, 1, 1)
    plt.plot(plot_df['time_bucket'], plot_df['article_count'], label='Article Count')
    plt.title('Article Volume')
    plt.legend()
    
    # Plot energy theme and energy crisis indicator
    plt.subplot(3, 1, 2)
    plt.plot(plot_df['time_bucket'], plot_df['theme_Energy_sum'], label='Energy Theme Mentions')
    plt.plot(plot_df['time_bucket'], plot_df['energy_crisis_indicator'], label='Energy Crisis Indicator')
    plt.title('Energy-Related Signals')
    plt.legend()
    
    # Plot tone metrics
    plt.subplot(3, 1, 3)
    plt.plot(plot_df['time_bucket'], plot_df['tone_tone_mean'], label='Average Tone')
    plt.plot(plot_df['time_bucket'], plot_df['tone_volatility'], label='Tone Volatility')
    plt.title('Sentiment Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('time_series_indicators.png')
    
    return complete_df

if __name__ == "__main__":
    main()