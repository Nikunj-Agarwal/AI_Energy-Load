import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Import configuration - ADDED IMPORT FROM CONFIG FILE
from config import (
    PATHS, setup_and_verify, test_directory_writing as config_test_directory_writing
)

# Use paths from config instead of hardcoded paths - IMPROVED PATH USAGE
INPUT_PATH = os.path.join(PATHS["PROCESSED_DIR"], "processed_gkg_parsed_data.csv")
OUTPUT_PATH = os.path.join(PATHS["AGGREGATED_DIR"], "aggregated_gkg_15min.csv")
FIGURES_DIR = PATHS["FIGURES_DIR"]

# Define key theme categories most likely to affect energy load
# Updated to match categories used in sparsing.py
ENERGY_THEMES = ['Energy', 'Environment', 'Infrastructure', 'Social', 'Health', 
                'Political', 'Economic']  # Added more relevant themes

def aggregate_gkg_to_15min(df):
    """
    Aggregate GKG data into 15-minute intervals, focusing on energy-relevant features
    while minimizing feature count.
    """
    print(f"Starting aggregation of {len(df)} records...")
    
    # Check if the datetime column exists and is in the correct format
    if 'datetime' not in df.columns:
        print("Error: 'datetime' column not found. Aggregation cannot proceed.")
        return None
    
    # Ensure datetime is in the correct format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create 15-minute time buckets
    df['time_bucket'] = df['datetime'].dt.floor('15min')
    
    # 1. BASIC AGGREGATION
    # -------------------
    print("Performing basic aggregation...")
    
    # Define aggregation dictionary focusing on the most important features
    agg_dict = {
        # Count of articles in each interval
        'GKGRECORDID': 'count',
        
        # All theme categories (sum and max to capture prevalence and intensity)
        **{f'theme_{cat}': ['sum', 'max', 'mean'] for cat in ENERGY_THEMES},
        
        # Tone metrics (focus on average and extremes)
        'tone_tone': ['mean', 'min', 'max', 'std'],
        'tone_negative': ['max', 'mean'],  # Capture negative sentiment
        'tone_positive': ['max', 'mean'],  # Capture positive sentiment
        'tone_polarity': ['mean', 'max'],  # Added polarity metrics
        'tone_activity': ['mean', 'max'],  # Added activity metrics
        
        # Entity metrics (more comprehensive)
        'entity_count': ['sum', 'mean'],   # Total entities mentioned
        'entity_variety': ['sum', 'max'],  # Added entity variety
        
        # Amount metrics (from sparsing)
        'avg_amount': ['mean', 'max'],     # Average amounts in articles
        'max_amount': ['max', 'mean'],     # Maximum amounts
        'amount_count': ['sum']            # Count of amounts mentioned
    }
    
    # Group by time bucket and aggregate
    agg_df = df.groupby('time_bucket').agg(agg_dict)
    
    # Flatten column multi-index
    agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns]
    
    # Rename count column
    agg_df.rename(columns={'GKGRECORDID_count': 'article_count'}, inplace=True)
    
    # 2. FEATURE ENGINEERING
    # ---------------------
    print("Engineering features...")
    
    # Create composite features that combine themes with sentiment
    agg_df['energy_crisis_indicator'] = agg_df['theme_Energy_sum'] * agg_df['tone_negative_max'] 
    agg_df['weather_alert_indicator'] = agg_df['theme_Environment_sum'] * np.abs(agg_df['tone_tone_min'])
    agg_df['social_event_indicator'] = agg_df['theme_Social_sum'] * agg_df['article_count'] / 100
    agg_df['infrastructure_stress'] = agg_df['theme_Infrastructure_sum'] * agg_df['tone_negative_max']
    agg_df['political_crisis_indicator'] = agg_df['theme_Political_sum'] * agg_df['tone_negative_max']
    agg_df['economic_impact_indicator'] = agg_df['theme_Economic_sum'] * agg_df['tone_volatility']
    
    # Calculate tone volatility (max - min) as a single metric instead of std
    agg_df['tone_volatility'] = agg_df['tone_tone_max'] - agg_df['tone_tone_min']
    
    # Article volume change detection
    agg_df = agg_df.reset_index()
    agg_df['prev_article_count'] = agg_df['article_count'].shift(1).fillna(0)
    agg_df['article_count_change'] = agg_df['article_count'] - agg_df['prev_article_count']
    agg_df['article_volume_spike'] = (agg_df['article_count'] > 
                                    agg_df['article_count'].rolling(window=12).mean() * 1.5).astype(int)
    
    # 3. TIME FEATURES
    # --------------
    print("Adding temporal features...")
    
    # Add temporal features
    agg_df['hour'] = agg_df['time_bucket'].dt.hour
    agg_df['day_of_week'] = agg_df['time_bucket'].dt.dayofweek
    agg_df['is_weekend'] = agg_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    agg_df['is_business_hours'] = ((agg_df['hour'] >= 9) & (agg_df['hour'] <= 17) & 
                                  (agg_df['is_weekend'] == 0)).astype(int)
    agg_df['month'] = agg_df['time_bucket'].dt.month
    agg_df['day'] = agg_df['time_bucket'].dt.day
    
    # Add seasonality features using sine and cosine transforms
    agg_df['hour_sin'] = np.sin(2 * np.pi * agg_df['hour'] / 24)
    agg_df['hour_cos'] = np.cos(2 * np.pi * agg_df['hour'] / 24)
    agg_df['day_of_week_sin'] = np.sin(2 * np.pi * agg_df['day_of_week'] / 7)
    agg_df['day_of_week_cos'] = np.cos(2 * np.pi * agg_df['day_of_week'] / 7)
    
    # 4. FEATURE CLEANUP AND SELECTION
    # ------------------------------
    print("Finalizing feature set...")
    
    # List of important features to keep (customize as needed)
    keep_features = [
        'time_bucket', 'article_count', 'article_count_change', 'article_volume_spike',
        
        # Core tone metrics
        'tone_tone_mean', 'tone_negative_max', 'tone_positive_max', 'tone_volatility',
        'tone_polarity_mean', 'tone_activity_mean',
        
        # Theme sums
        'theme_Energy_sum', 'theme_Environment_sum', 'theme_Infrastructure_sum', 
        'theme_Social_sum', 'theme_Health_sum', 'theme_Political_sum', 'theme_Economic_sum',
        
        # Theme intensity (max values)
        'theme_Energy_max', 'theme_Environment_max', 'theme_Infrastructure_max',
        
        # Entity and amount metrics
        'entity_count_sum', 'entity_variety_max', 'max_amount_max',
        
        # Composite indicators
        'energy_crisis_indicator', 'weather_alert_indicator', 
        'social_event_indicator', 'infrastructure_stress',
        'political_crisis_indicator', 'economic_impact_indicator',
        
        # Time features
        'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
        'month', 'day', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'
    ]
    
    # Ensure all requested columns exist
    available_columns = agg_df.columns.tolist()
    final_columns = [col for col in keep_features if col in available_columns]
    
    if len(final_columns) < len(keep_features):
        missing = set(keep_features) - set(final_columns)
        print(f"Warning: Some columns were not found in the data: {missing}")
    
    # Keep only the most relevant features
    final_df = agg_df[final_columns]
    
    print(f"Aggregation complete. Created {len(final_df)} time buckets with {len(final_columns)} features.")
    return final_df

def handle_missing_intervals(df, start_date=None, end_date=None, freq='15min'):
    """
    Ensure all intervals exist in the time range, filling gaps with appropriate values.
    This is important for time series analysis.
    """
    print("Checking for missing time intervals...")
    
    # If no dates provided, use min/max from the data
    if start_date is None:
        start_date = df['time_bucket'].min()
    if end_date is None:
        end_date = df['time_bucket'].max()
    
    print(f"Time range: {start_date} to {end_date}")
    
    # Create complete interval range
    complete_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    if len(complete_range) == len(df):
        print("No missing intervals found.")
    else:
        print(f"Found {len(complete_range) - len(df)} missing intervals out of {len(complete_range)} total.")
    
    # Create a reference dataframe with all intervals
    ref_df = pd.DataFrame({'time_bucket': complete_range})
    
    # Merge with actual data
    merged_df = pd.merge(ref_df, df, on='time_bucket', how='left')
    
    # Fill missing values with appropriate values
    # For article count and most sum metrics, use 0
    sum_cols = [col for col in merged_df.columns if col.endswith('_sum') or col == 'article_count']
    merged_df[sum_cols] = merged_df[sum_cols].fillna(0)
    
    # For means and other metrics, use forward fill then backward fill
    # to maintain reasonable values
    other_numeric = merged_df.select_dtypes(include=[np.number]).columns.difference(sum_cols)
    merged_df[other_numeric] = merged_df[other_numeric].fillna(method='ffill')
    merged_df[other_numeric] = merged_df[other_numeric].fillna(method='bfill')
    
    # Any remaining missing values fill with 0
    merged_df = merged_df.fillna(0)
    
    return merged_df

def analyze_features(df):
    """
    Analyze feature importance and correlations to help with feature selection.
    """
    print("Analyzing feature relationships...")
    
    # Skip non-numeric columns for correlation analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlations
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Create mask for upper triangle
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_correlations.png'), dpi=300)
    plt.close()
    
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
    
    # Feature distribution analysis
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(ENERGY_THEMES[:5]):  # First 5 themes
        plt.subplot(2, 3, i+1)
        sns.histplot(df[f'theme_{col}_sum'], kde=True)
        plt.title(f'{col} Theme Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'theme_distributions.png'), dpi=300)
    plt.close()
    
    return corr_matrix

def generate_time_series_plots(df):
    """Generate exploratory time series plots of key features"""
    print("Generating time series visualizations...")
    
    # Select a subset of data points for clarity if dataset is large
    plot_df = df
    if len(df) > 1000:
        # Sample at 1-hour intervals for cleaner plots
        plot_df = df.iloc[::4]  
    
    # 1. Article volume and spikes
    plt.figure(figsize=(15, 8))
    plt.plot(plot_df['time_bucket'], plot_df['article_count'], label='Article Count')
    plt.scatter(plot_df['time_bucket'][plot_df['article_volume_spike'] == 1], 
                plot_df['article_count'][plot_df['article_volume_spike'] == 1],
                color='red', alpha=0.6, label='Volume Spikes')
    plt.title('Article Volume Over Time with Detected Spikes', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Article Count', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'article_volume_spikes.png'), dpi=300)
    plt.close()
    
    # 2. Theme presence over time
    plt.figure(figsize=(15, 10))
    for i, theme in enumerate(ENERGY_THEMES[:5]):  # First 5 themes
        plt.plot(plot_df['time_bucket'], plot_df[f'theme_{theme}_sum'], 
                label=f'{theme} Theme', alpha=0.7)
    plt.title('Theme Presence Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Theme Mentions', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'theme_presence.png'), dpi=300)
    plt.close()
    
    # 3. Crisis indicators
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.plot(plot_df['time_bucket'], plot_df['energy_crisis_indicator'], 
             label='Energy Crisis', color='red', alpha=0.7)
    plt.plot(plot_df['time_bucket'], plot_df['infrastructure_stress'], 
             label='Infrastructure Stress', color='orange', alpha=0.7)
    plt.title('Energy & Infrastructure Crisis Indicators', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(plot_df['time_bucket'], plot_df['weather_alert_indicator'], 
             label='Weather Alert', color='blue', alpha=0.7)
    plt.plot(plot_df['time_bucket'], plot_df['social_event_indicator'], 
             label='Social Event', color='green', alpha=0.7)
    plt.title('Weather & Social Event Indicators', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'crisis_indicators.png'), dpi=300)
    plt.close()
    
    # 4. Tone metrics
    plt.figure(figsize=(15, 8))
    plt.plot(plot_df['time_bucket'], plot_df['tone_tone_mean'], label='Average Tone', color='purple')
    plt.fill_between(plot_df['time_bucket'], 
                     plot_df['tone_tone_mean'] - plot_df['tone_volatility']/2,
                     plot_df['tone_tone_mean'] + plot_df['tone_volatility']/2, 
                     alpha=0.3, color='purple', label='Tone Volatility')
    plt.title('Sentiment Tone and Volatility Over Time', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'tone_metrics.png'), dpi=300)
    plt.close()
    
    # 5. Weekly patterns
    daily_pattern = df.groupby('hour')['article_count'].mean().reset_index()
    weekly_pattern = df.groupby('day_of_week')['article_count'].mean().reset_index()
    
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    sns.barplot(x='hour', y='article_count', data=daily_pattern)
    plt.title('Average Article Volume by Hour of Day', fontsize=14)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Average Article Count', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.barplot(x='day_of_week', y='article_count', data=weekly_pattern)
    plt.title('Average Article Volume by Day of Week', fontsize=14)
    plt.xlabel('Day of Week (0=Monday, 6=Sunday)', fontsize=12)
    plt.ylabel('Average Article Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'temporal_patterns.png'), dpi=300)
    plt.close()

def test_directory_writing():
    """Test writing to all required directories before processing"""
    print("Testing directory writing permissions...")
    
    # IMPROVED DIRECTORY TESTING - More comprehensive tests
    directories = [
        os.path.dirname(INPUT_PATH),  # Added input directory
        os.path.dirname(OUTPUT_PATH),
        FIGURES_DIR
    ]
    
    all_passed = True
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"ERROR: Could not create directory {directory}: {e}")
                all_passed = False
                continue
        
        # Test writing a small file
        test_file = os.path.join(directory, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write(f"Test write at {datetime.now()}")
            os.remove(test_file)
            print(f"✓ Successfully wrote to {directory}")
        except Exception as e:
            print(f"✗ ERROR: Could not write to {directory}: {e}")
            all_passed = False
    
    # Test writing a DataFrame to output directory
    try:
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        test_csv_path = os.path.join(os.path.dirname(OUTPUT_PATH), "test_dataframe.csv")
        test_df.to_csv(test_csv_path, index=False)
        os.remove(test_csv_path)
        print(f"✓ Successfully wrote test DataFrame to {os.path.dirname(OUTPUT_PATH)}")
    except Exception as e:
        print(f"✗ ERROR: Could not write test DataFrame: {e}")
        all_passed = False
            
    # Also test writing a small test plot to ensure matplotlib can save figures
    if os.path.exists(FIGURES_DIR):
        try:
            test_fig_path = os.path.join(FIGURES_DIR, "test_plot.png")
            plt.figure(figsize=(2, 2))
            plt.plot([1, 2, 3], [1, 4, 9])
            plt.savefig(test_fig_path)
            plt.close()
            os.remove(test_fig_path)
            print(f"✓ Successfully created test plot in {FIGURES_DIR}")
        except Exception as e:
            print(f"✗ ERROR: Could not create test plot: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main function to execute the data aggregation pipeline"""
    print(f"=== GDELT GKG Data Aggregation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # ADDED - First verify all directories exist using config functions
    if not setup_and_verify():
        print("ERROR: Directory setup verification failed. Aborting processing.")
        return
    
    # Test directory writing before proceeding
    if not test_directory_writing():
        print("ERROR: Directory writing test failed. Aborting processing.")
        return
    
    # Load the parsed data
    print(f"Loading data from {INPUT_PATH}...")
    try:
        df = pd.read_csv(INPUT_PATH)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"Loaded {len(df)} records with {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Perform aggregation
    print("Aggregating data to 15-minute intervals...")
    agg_df = aggregate_gkg_to_15min(df)
    
    if agg_df is None:
        print("Aggregation failed. Check the logs for errors.")
        return
    
    # Handle missing intervals
    print("Ensuring all 15-minute intervals are present...")
    complete_df = handle_missing_intervals(agg_df)
    
    # Analyze features
    corr_matrix = analyze_features(complete_df)
    
    # Generate exploratory visualizations
    generate_time_series_plots(complete_df)
    
    # Save the aggregated data
    print(f"Saving aggregated data to {OUTPUT_PATH}...")
    # IMPROVED - Ensure output directory exists before saving
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    complete_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Data aggregation complete. Created {len(complete_df)} 15-minute intervals with {complete_df.shape[1]} features.")
    
    # Print summary statistics
    print("\nSummary of key aggregated metrics:")
    print(complete_df[['article_count', 'tone_tone_mean', 'energy_crisis_indicator', 
                     'weather_alert_indicator', 'social_event_indicator']].describe().T)
    
    print(f"\nOutput file saved to: {OUTPUT_PATH}")
    print(f"Visualization files saved to: {FIGURES_DIR}")
    print(f"=== Processing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    return complete_df

if __name__ == "__main__":
    main()