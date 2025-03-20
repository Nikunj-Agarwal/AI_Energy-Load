"""
Theme Analysis for GDELT GKG Data
---------------------------------
This script analyzes themes in GDELT GKG data, focusing on energy-related themes 
that might impact energy load forecasting. It supports:

1. Raw GKG data analysis: Extracts and analyzes themes from raw GDELT GKG files
2. Batch processing: Can process multiple files from the batch output directory
3. Pre-processed data: Can work with data already processed by gkg_data_sparsing.py

Usage:
- By default, processes the file specified in INPUT_FILE
- Set PROCESS_BATCH_FILES=True to process all files in BATCH_DIR
- Set USE_PREPROCESSED=True to use data that's already been through the sparsing script

The script generates visualizations and CSV outputs with theme analysis results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import os
from pathlib import Path
from datetime import datetime
import glob
from tqdm import tqdm

#============================================================
# CONFIGURATION
#============================================================
# Input/Output Paths
INPUT_FILE = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets\delhi_gkg_data_2021_jan1_3.csv"
BATCH_DIR = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR\raw_data\batch_outputs"
PREPROCESSED_FILE = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR\processed_data\processed_gkg_parsed_data.csv"
MERGED_OUTPUT_FILE = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR\processed_data\theme_analysis_merged.csv"

# Create a better organized output structure
BASE_OUTPUT_DIR = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR\analysis_results"
# The actual output directory will be created with a timestamp

# Processing options
PROCESS_BATCH_FILES = True   # Set to True to process all files in BATCH_DIR
USE_PREPROCESSED = False     # Set to True to use already preprocessed data
SAMPLE_SIZE = None           # Set to a number to limit processing (for testing)

# Theme Categories - Must align with categories in gkg_data_sparsing.py
THEME_CATEGORIES = {
    'Political': ['ELECTION', 'GOVERN', 'POLIT', 'DEMO', 'LEG', 'VOTE', 'PARLIAMENT', 'PRESIDENT', 'MINISTER'],
    'Economic': ['ECON', 'MARKET', 'TRADE', 'BUSINESS', 'FINANC', 'TAX', 'INVEST', 'GDP', 'INFLATION'],
    'Religious': ['RELIG', 'MUSLIM', 'HINDU', 'SIKH', 'TEMPLE', 'CHURCH', 'MOSQUE', 'FAITH', 'GOD'],
    'Energy': ['ENERGY', 'POWER', 'ELECTRIC', 'OIL', 'GAS', 'COAL', 'FUEL', 'RENEWABLE', 'GRID'],
    'Infrastructure': ['INFRA', 'TRANSPORT', 'CONSTRUCT', 'BUILDING', 'ROAD', 'HIGHWAY', 'RAIL', 'AIRPORT'],
    'Health': ['HEALTH', 'COVID', 'DISEASE', 'PANDEMIC', 'HOSPITAL', 'MEDIC', 'VACCINE', 'DRUG'],
    'Social': ['SOCIAL', 'PROTEST', 'RALLY', 'CELEBR', 'FESTIVAL', 'COMMUNITY', 'SOCIETY', 'PUBLIC', 'CITIZEN'],
    'Environment': ['ENV', 'CLIMATE', 'WEATHER', 'POLLUT', 'WATER', 'GREEN', 'ECOLOGY', 'TEMPERATURE'],
    'Education': ['EDU', 'SCHOOL', 'UNIVERSITY', 'STUDENT', 'LEARNING', 'COLLEGE', 'TEACHER', 'PROFESSOR']
}

# Energy-Related Keywords - Terms that might directly impact energy consumption
ENERGY_KEYWORDS = [
    'POWER', 'ENERGY', 'ELECTRIC', 'GRID', 'OUTAGE', 'BLACKOUT', 'LOAD', 'CONSUMPTION',
    'WEATHER', 'TEMPERATURE', 'HEAT', 'COLD', 'WINTER', 'SUMMER', 'RAINFALL',
    'FESTIVAL', 'CELEBRATION', 'EVENT', 'HOLIDAY', 'GATHERING',
    'INFRA', 'CONSTRUCTION', 'DEVELOPMENT',
    'PROTEST', 'RALLY', 'STRIKE', 'DISRUPTION',
    'OFFICE', 'WORKPLACE', 'WORK', 'BUSINESS', 'INDUSTRY'
]

# Time periods for temporal analysis
TIME_PERIODS = {
    'Morning': (5, 11),     # 5 AM to 11:59 AM
    'Afternoon': (12, 16),  # 12 PM to 4:59 PM
    'Evening': (17, 21),    # 5 PM to 9:59 PM
    'Night': (22, 4)        # 10 PM to 4:59 AM
}

#============================================================
# DATA LOADING
#============================================================
def load_gkg_data(file_path):
    """Load GDELT GKG data from CSV file"""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records with {df.shape[1]} columns")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_theme_column(df):
    """Determine which theme column to use"""
    if 'V2Themes' in df.columns:
        return 'V2Themes'
    elif 'Themes' in df.columns:
        return 'Themes'
    elif any(col.startswith('theme_') for col in df.columns):
        # This is preprocessed data from sparsing.py
        return 'preprocessed'
    else:
        print("Error: No theme columns found in the dataset")
        return None

def load_batch_files(batch_dir=BATCH_DIR, limit=None):
    """Load and combine data from multiple batch files"""
    print(f"Looking for batch files in {batch_dir}...")
    batch_files = glob.glob(os.path.join(batch_dir, "*.csv"))
    
    if not batch_files:
        print("No batch files found!")
        return None
    
    if limit:
        batch_files = batch_files[:limit]
        
    print(f"Found {len(batch_files)} batch files. Processing...")
    
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
        return None
    
    # Combine all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(batch_files)} files with a total of {total_rows} rows")
    
    return merged_df

#============================================================
# THEME EXTRACTION & ANALYSIS
#============================================================
def extract_themes(df, theme_column):
    """Extract all themes from the dataset"""
    print(f"Extracting themes from {theme_column} column...")
    
    # For preprocessed data, use the theme columns directly
    if (theme_column == 'preprocessed'):
        print("Using preprocessed theme data...")
        theme_counts = {}
        for category in THEME_CATEGORIES:
            col_name = f'theme_{category}'
            if col_name in df.columns:
                # Sum the theme occurrences
                theme_counts[category] = df[col_name].sum()
        return Counter(theme_counts)
    
    # For raw data, extract themes from the text column
    # Filter out null values
    themes_data = df[theme_column].dropna()
    
    # Split themes (they are typically separated by semicolons)
    all_themes = []
    for themes_str in tqdm(themes_data, desc="Processing themes", total=len(themes_data)):
        # Split by semicolon if present, otherwise try comma
        if ';' in str(themes_str):
            themes = themes_str.split(';')
        else:
            themes = themes_str.split(',')
        
        # Clean up each theme and add to list
        for theme in themes:
            theme = theme.strip()
            if theme:
                # Extract the theme name (before any comma)
                if ',' in theme:
                    theme = theme.split(',')[0]
                all_themes.append(theme)
    
    # Count theme occurrences
    theme_counter = Counter(all_themes)
    return theme_counter

def categorize_themes(theme_counter, categories=THEME_CATEGORIES):
    """Group themes into predefined categories"""
    print("Categorizing themes...")
    category_counts = {category: 0 for category in categories}
    
    # If already using preprocessed data, theme_counter has category names as keys
    if set(theme_counter.keys()).issubset(set(categories.keys())):
        return dict(theme_counter)
    
    # Otherwise, categorize raw themes
    for theme, count in theme_counter.items():
        for category, keywords in categories.items():
            if any(keyword in theme.upper() for keyword in keywords):
                category_counts[category] += count
                break  # Assign to first matching category
    
    return category_counts

def identify_energy_related_themes(theme_counter, keywords=ENERGY_KEYWORDS):
    """Identify themes that might affect energy load"""
    if not theme_counter:
        return None
    
    print("Identifying energy-related themes...")
    potential_energy_themes = []
    
    # With preprocessed data, we may not have individual themes
    if set(theme_counter.keys()).issubset(set(THEME_CATEGORIES.keys())):
        # Just return the count of the Energy category
        return [('Energy Category', theme_counter.get('Energy', 0))]
    
    # For raw themes, search for energy keywords
    for theme in theme_counter:
        if any(keyword in theme.upper() for keyword in keywords):
            potential_energy_themes.append((theme, theme_counter[theme]))
    
    # Sort by count
    potential_energy_themes.sort(key=lambda x: x[1], reverse=True)
    return potential_energy_themes

def analyze_temporal_patterns(df):
    """Analyze how themes appear throughout the day"""
    print("Analyzing temporal patterns of themes...")
    
    # Check if we have datetime information
    if 'datetime' not in df.columns:
        if 'DATE' in df.columns:
            # Try to convert DATE to datetime
            try:
                df['datetime'] = pd.to_datetime(df['DATE'], format='%Y%m%d%H%M%S')
            except:
                print("Could not convert DATE to datetime. Skipping temporal analysis.")
                return None
        else:
            print("No datetime information found. Skipping temporal analysis.")
            return None
    
    # Extract hour information
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour
    
    # Assign time periods
    def assign_period(hour):
        for period, (start, end) in TIME_PERIODS.items():
            if start <= end:  # Normal period (e.g., 9-17)
                if start <= hour <= end:
                    return period
            else:  # Period crosses midnight (e.g., 22-4)
                if hour >= start or hour <= end:
                    return period
        return 'Other'  # Fallback
    
    df['time_period'] = df['hour'].apply(assign_period)
    
    # Analyze themes by time period
    period_theme_counts = {}
    
    # Check if we have preprocessed theme columns
    theme_cols = [col for col in df.columns if col.startswith('theme_')]
    
    if theme_cols:
        # Use preprocessed theme columns
        for period in TIME_PERIODS.keys():
            period_df = df[df['time_period'] == period]
            if len(period_df) > 0:
                period_counts = {}
                for col in theme_cols:
                    category = col.replace('theme_', '')
                    period_counts[category] = period_df[col].sum()
                period_theme_counts[period] = period_counts
    else:
        # Use raw theme column
        theme_column = get_theme_column(df)
        if theme_column not in ['V2Themes', 'Themes']:
            return None
            
        for period in TIME_PERIODS.keys():
            period_df = df[df['time_period'] == period]
            if len(period_df) > 0:
                themes = extract_themes(period_df, theme_column)
                categories = categorize_themes(themes)
                period_theme_counts[period] = categories
    
    return period_theme_counts

#============================================================
# VISUALIZATION
#============================================================
def ensure_output_dir(output_dir):
    """Ensure the output directory exists"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_top_themes(theme_counter, n=20, output_dirs=None):
    """Plot the top N most common themes"""
    if output_dirs is None:
        output_dir = ensure_output_dir(OUTPUT_DIR)
    else:
        output_dir = output_dirs["visualizations"]["themes"]
    
    plt.figure(figsize=(14, 8))
    top_themes = dict(theme_counter.most_common(n))
    
    # Check if we're using category names from preprocessed data
    if set(theme_counter.keys()).issubset(set(THEME_CATEGORIES.keys())):
        # Use categories directly
        categories = list(top_themes.keys())
        counts = list(top_themes.values())
    else:
        # Use raw themes
        categories = list(top_themes.keys())
        counts = list(top_themes.values())
    
    # Create the bar chart with improved styling
    bars = plt.bar(categories, counts, color=sns.color_palette("viridis", len(categories)))
    
    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {n} Themes in Delhi GKG Dataset', fontsize=16)
    plt.xlabel('Theme', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'top_themes_delhi.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()
    
    return output_path  # Return the path for use in reports

def plot_theme_categories(category_counts, output_dir=OUTPUT_DIR):
    """Plot theme categories as a pie chart and bar chart"""
    output_dir = ensure_output_dir(output_dir)
    
    # Sort categories by count
    sorted_categories = dict(sorted(category_counts.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True))
    
    # 1. Pie Chart
    plt.figure(figsize=(12, 10))
    
    # Use a better color palette
    colors = sns.color_palette('viridis', len(sorted_categories))
    
    # Create pie chart with better formatting
    wedges, texts, autotexts = plt.pie(
        sorted_categories.values(), 
        labels=sorted_categories.keys(), 
        autopct='%1.1f%%', 
        startangle=90,
        colors=colors,
        wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 1}
    )
    
    # Improve text appearance
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
    
    plt.axis('equal')
    plt.title('Distribution of Theme Categories', fontsize=16)
    
    output_path = os.path.join(output_dir, 'theme_categories_pie.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved pie chart to {output_path}")
    plt.close()
    
    # 2. Bar Chart (more precise for comparing values)
    plt.figure(figsize=(14, 8))
    categories = list(sorted_categories.keys())
    counts = list(sorted_categories.values())
    
    bars = plt.bar(categories, counts, color=sns.color_palette("viridis", len(categories)))
    
    # Add counts above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title('Theme Categories by Count', fontsize=16)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'theme_categories_bar.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved bar chart to {output_path}")
    plt.close()

def plot_energy_themes(energy_themes, output_dir=OUTPUT_DIR):
    """Plot energy-related themes"""
    if not energy_themes:
        print("No energy-related themes to plot")
        return
    
    output_dir = ensure_output_dir(output_dir)
    
    # Use only top 15 for better visualization
    top_energy = energy_themes[:15]
    themes, counts = zip(*top_energy)
    
    plt.figure(figsize=(14, 8))
    
    # Create horizontal bar chart with improved styling
    bars = plt.barh(themes, counts, color=sns.color_palette("YlOrRd", len(themes)))
    
    # Add counts to the right of bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{int(width)}', ha='left', va='center', fontsize=10)
    
    plt.title('Top Energy-Related Themes', fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel('Theme', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'energy_related_themes.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_temporal_patterns(period_theme_counts, output_dir=OUTPUT_DIR):
    """Plot theme patterns across different time periods"""
    if not period_theme_counts:
        print("No temporal data to plot")
        return
    
    output_dir = ensure_output_dir(output_dir)
    
    # Create a DataFrame for easier plotting
    periods = []
    categories = []
    counts = []
    
    for period, theme_dict in period_theme_counts.items():
        for category, count in theme_dict.items():
            periods.append(period)
            categories.append(category)
            counts.append(count)
    
    temporal_df = pd.DataFrame({
        'Period': periods,
        'Category': categories,
        'Count': counts
    })
    
    # Plot as heatmap
    plt.figure(figsize=(12, 8))
    pivot_df = temporal_df.pivot(index='Category', columns='Period', values='Count')
    
    # Normalize by column (time period) to see relative importance
    normalized_df = pivot_df.div(pivot_df.sum(axis=0), axis=1)
    
    # Sort categories by total count
    category_order = pivot_df.sum(axis=1).sort_values(ascending=False).index
    
    # Plot using the sorted order
    heatmap_df = normalized_df.reindex(category_order)
    sns.heatmap(heatmap_df, annot=pivot_df.reindex(category_order), 
                fmt='g', cmap='YlGnBu', linewidths=.5)
    
    plt.title('Theme Distribution Across Time Periods', fontsize=16)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'temporal_theme_patterns.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved temporal pattern plot to {output_path}")
    plt.close()
    
    # Also create a grouped bar chart for absolute counts
    plt.figure(figsize=(14, 10))
    
    # Use a subset of categories for clarity
    top_categories = pivot_df.sum(axis=1).sort_values(ascending=False).index[:6]
    plot_df = pivot_df.loc[top_categories]
    
    ax = plot_df.plot(kind='bar', figsize=(14, 8), rot=0, width=0.7)
    
    plt.title('Theme Counts by Time Period', fontsize=16)
    plt.xlabel('Theme Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Time Period', fontsize=10)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'temporal_theme_barchart.png')
    plt.savefig(output_path, dpi=300)
    print(f"Saved temporal bar chart to {output_path}")
    plt.close()

#============================================================
# REPORTING
#============================================================
def print_theme_statistics(theme_counter, category_counts, energy_themes):
    """Print theme analysis statistics"""
    print("\n" + "="*60)
    print("THEME ANALYSIS RESULTS")
    print("="*60)
    
    # General statistics
    print(f"\nFound {len(theme_counter)} unique themes/categories in the dataset")
    
    # Top themes
    print(f"\nTop themes/categories:")
    for theme, count in theme_counter.most_common(20):
        print(f"{theme}: {count}")
    
    # Category counts
    print("\nTheme counts by category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count}")
    
    # Energy-related themes
    print("\nPotential energy-related themes:")
    if energy_themes:
        for theme, count in energy_themes:
            print(f"{theme}: {count}")
    else:
        print("No energy-related themes found.")

def save_results_to_csv(theme_counter, category_counts, energy_themes, period_theme_counts=None, output_dir=OUTPUT_DIR):
    """Save analysis results to CSV files"""
    output_dir = ensure_output_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all themes
    themes_df = pd.DataFrame(theme_counter.most_common(), columns=['Theme', 'Count'])
    themes_path = os.path.join(output_dir, f'all_themes_{timestamp}.csv')
    themes_df.to_csv(themes_path, index=False)
    
    # Save categories
    categories_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
    categories_path = os.path.join(output_dir, f'theme_categories_{timestamp}.csv')
    categories_df.to_csv(categories_path, index=False)
    
    # Save energy themes
    if energy_themes:
        energy_df = pd.DataFrame(energy_themes, columns=['Theme', 'Count'])
        energy_path = os.path.join(output_dir, f'energy_themes_{timestamp}.csv')
        energy_df.to_csv(energy_path, index=False)
    
    # Save temporal patterns if available
    if period_theme_counts:
        # Create a DataFrame for time period analysis
        periods = []
        categories = []
        counts = []
        
        for period, theme_dict in period_theme_counts.items():
            for category, count in theme_dict.items():
                periods.append(period)
                categories.append(category)
                counts.append(count)
        
        temporal_df = pd.DataFrame({
            'Period': periods,
            'Category': categories,
            'Count': counts
        })
        
        temporal_path = os.path.join(output_dir, f'temporal_theme_patterns_{timestamp}.csv')
        temporal_df.to_csv(temporal_path, index=False)
    
    print(f"\nResults saved to {output_dir}")

def generate_html_report(output_dirs, theme_counter, category_counts, energy_themes, period_theme_counts=None, viz_paths=None):
    """Generate an HTML report with embedded visualizations and analysis results"""
    report_path = os.path.join(output_dirs["reports"], "theme_analysis_report.html")
    
    # Check if visualization paths are provided
    if not viz_paths:
        print("Error: Visualization paths are missing. HTML report cannot be generated.")
        return None

    # Check if all required visualization paths exist
    required_viz_keys = ['category_pie', 'category_bar', 'top_themes', 'energy_themes']
    if period_theme_counts:
        required_viz_keys.extend(['temporal_heatmap', 'temporal_bar'])
    missing_viz = [key for key in required_viz_keys if key not in viz_paths]
    if missing_viz:
        print(f"Error: Missing visualization paths for {missing_viz}. HTML report may be incomplete.")
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>GDELT Theme Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .section {{ margin-bottom: 30px; padding: 20px; background-color: #f9f9f9; border-radius: 5px; }}
            .viz-container {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
    </head>
    <body>
        <h1>GDELT GKG Theme Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <div class="section">
            <h2>Overview</h2>
            <p>This report presents an analysis of themes found in GDELT GKG data related to Delhi.</p>
            <p>Total unique themes/categories: {len(theme_counter)}</p>
        </div>
        
        <div class="section">
            <h2>Theme Categories</h2>
            <div class="viz-container">
                <img src="{os.path.relpath(viz_paths.get('category_pie', ''), output_dirs['reports'])}" alt="Theme Categories Pie Chart">
            </div>
            <div class="viz-container">
                <img src="{os.path.relpath(viz_paths.get('category_bar', ''), output_dirs['reports'])}" alt="Theme Categories Bar Chart">
            </div>
            
            <h3>Category Distribution</h3>
            <table>
                <tr>
                    <th>Category</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
    """
    
    # Add category data to the table
    total_count = sum(category_counts.values())
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_count * 100) if total_count > 0 else 0
        html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
    """
    
    # Ensure all visualizations are properly referenced
    if 'top_themes' in viz_paths:
        html_content += f"""
        <div class="section">
            <h2>Top Themes</h2>
            <div class="viz-container">
                <img src="{os.path.relpath(viz_paths['top_themes'], output_dirs['reports'])}" alt="Top Themes">
            </div>
        </div>
        """
    
    # Add Energy-related themes section
    if energy_themes:
        html_content += """
        <div class="section">
            <h2>Energy-Related Themes</h2>
            <div class="viz-container">
                <img src="{}" alt="Energy-Related Themes">
            </div>
            
            <h3>Top Energy Themes</h3>
            <table>
                <tr>
                    <th>Theme</th>
                    <th>Count</th>
                </tr>
        """.format(os.path.relpath(viz_paths['energy_themes'], output_dirs['reports']))
        
        # Add energy theme data
        for theme, count in energy_themes[:15]:  # Show top 15
            html_content += f"""
                <tr>
                    <td>{theme}</td>
                    <td>{count}</td>
                </tr>
            """
        
        html_content += """
            </table>
        </div>
        """
    
    # Add Temporal Analysis section if available
    if period_theme_counts and 'temporal_heatmap' in viz_paths:
        html_content += """
        <div class="section">
            <h2>Temporal Analysis</h2>
            <div class="viz-container">
                <img src="{}" alt="Temporal Theme Patterns Heatmap">
            </div>
            <div class="viz-container">
                <img src="{}" alt="Temporal Theme Patterns Bar Chart">
            </div>
        </div>
        """.format(
            os.path.relpath(viz_paths['temporal_heatmap'], output_dirs['reports']),
            os.path.relpath(viz_paths['temporal_bar'], output_dirs['reports'])
        )
    
    # Close the HTML document
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Generated HTML report at: {report_path}")
    return report_path

#============================================================
# MAIN EXECUTION
#============================================================
def process_file(file_path):
    """Process a single GKG data file"""
    # Load data
    df = load_gkg_data(file_path)
    if df is None:
        return None, None, None, None
    
    # Sample data if requested
    if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=42)
        print(f"Using sample of {SAMPLE_SIZE} records")
    
    # Get theme column
    theme_column = get_theme_column(df)
    if theme_column is None:
        return None, None, None, None
    
    # Extract and analyze themes
    theme_counter = extract_themes(df, theme_column)
    
    # Categorize themes
    category_counts = categorize_themes(theme_counter)
    
    # Identify energy-related themes
    energy_themes = identify_energy_related_themes(theme_counter)
    
    # Analyze temporal patterns
    period_theme_counts = analyze_temporal_patterns(df)
    
    return theme_counter, category_counts, energy_themes, period_theme_counts

def create_organized_output_dirs():
    """Create an organized directory structure for outputs with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE_OUTPUT_DIR, f"analysis_{timestamp}")
    
    # Create main run directory
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories for different outputs
    viz_dir = os.path.join(run_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # Subdirectories for different visualization types
    theme_viz_dir = os.path.join(viz_dir, "themes")
    category_viz_dir = os.path.join(viz_dir, "categories")
    energy_viz_dir = os.path.join(viz_dir, "energy")
    temporal_viz_dir = os.path.join(viz_dir, "temporal")
    
    os.makedirs(theme_viz_dir, exist_ok=True)
    os.makedirs(category_viz_dir, exist_ok=True)
    os.makedirs(energy_viz_dir, exist_ok=True)
    os.makedirs(temporal_viz_dir, exist_ok=True)
    
    # Directory for data files
    data_dir = os.path.join(run_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Directory for reports
    report_dir = os.path.join(run_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Create a structure to hold all directory paths
    dirs = {
        "root": run_dir,
        "visualizations": {
            "main": viz_dir,
            "themes": theme_viz_dir,
            "categories": category_viz_dir,
            "energy": energy_viz_dir,
            "temporal": temporal_viz_dir
        },
        "data": data_dir,
        "reports": report_dir
    }
    
    return dirs

def main(input_file=INPUT_FILE):
    """Main execution function"""
    print(f"=== GDELT GKG Theme Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Create organized directory structure
    output_dirs = create_organized_output_dirs()
    print(f"Analysis outputs will be saved to: {output_dirs['root']}")
    
    # Determine data source
    if PROCESS_BATCH_FILES:
        print("Mode: Processing batch files")
        df = load_batch_files(BATCH_DIR)
        if df is not None:
            # Optionally save the merged file
            df.to_csv(MERGED_OUTPUT_FILE, index=False)
            print(f"Saved merged data to {MERGED_OUTPUT_FILE}")
    elif USE_PREPROCESSED:
        print("Mode: Using preprocessed data")
        df = load_gkg_data(PREPROCESSED_FILE)
    else:
        print(f"Mode: Processing single file: {input_file}")
        df = load_gkg_data(input_file)
    
    if df is None:
        print("Error: No data to process")
        return
    
    # Process the data
    theme_counter, category_counts, energy_themes, period_theme_counts = process_file(df)
    
    if theme_counter is None:
        print("Error: Theme extraction failed")
        return
    
    # Keep track of visualization paths for the report
    viz_paths = {}
    
    # Create visualizations with the new directory structure
    viz_paths['top_themes'] = plot_top_themes(theme_counter, output_dirs=output_dirs)
    viz_paths['category_pie'], viz_paths['category_bar'] = plot_theme_categories(category_counts, output_dirs=output_dirs)
    viz_paths['energy_themes'] = plot_energy_themes(energy_themes, output_dirs=output_dirs)
    
    if period_theme_counts:
        viz_paths['temporal_heatmap'], viz_paths['temporal_bar'] = plot_temporal_patterns(period_theme_counts, output_dirs=output_dirs)
    
    # Save results to CSV in the data directory
    save_results_to_csv(theme_counter, category_counts, energy_themes, period_theme_counts, output_dir=output_dirs["data"])
    
    # Generate HTML report
    report_path = generate_html_report(output_dirs, theme_counter, category_counts, energy_themes, period_theme_counts, viz_paths)
    
    print(f"=== Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"Results saved to: {output_dirs['root']}")
    print(f"HTML Report: {report_path}")
    
    return theme_counter, category_counts, energy_themes

if __name__ == "__main__":
    main()