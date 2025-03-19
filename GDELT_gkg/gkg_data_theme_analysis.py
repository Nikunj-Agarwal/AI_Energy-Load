"""
Theme Analysis for GDELT GKG Data
---------------------------------
This script analyzes themes in GDELT GKG data, focusing on energy-related themes 
that might impact energy load forecasting.
"""
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import os
from pathlib import Path

#============================================================
# CONFIGURATION
#============================================================
# Input/Output Paths
INPUT_FILE = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets\delhi_gkg_data_2021_jan1_3.csv"
OUTPUT_DIR = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_graphs"

# Theme Categories
THEME_CATEGORIES = {
    'Political': ['ELECTION', 'GOVERN', 'POLIT', 'DEMO', 'LEG', 'VOTE'],
    'Economic': ['ECON', 'BUSINESS', 'MARKET', 'TRADE', 'INVEST'],
    'Religious': ['RELIG', 'MUSLIM', 'HINDU', 'SIKH', 'TEMPLE'],
    'Energy': ['ENERGY', 'POWER', 'ELECTRIC', 'OIL'],
    'Infrastructure': ['INFRA', 'TRANSPORT', 'CONSTRUCT'],
    'Health': ['HEALTH', 'COVID', 'DISEASE', 'PANDEMIC'],
    'Social': ['SOCIAL', 'PROTEST', 'RALLY', 'CELEBR'],
    'Environment': ['ENV', 'CLIMATE', 'WEATHER', 'POLLUT'],
    'Education': ['EDU', 'SCHOOL', 'UNIVERSITY', 'STUDENT']
}

# Energy-Related Keywords
ENERGY_KEYWORDS = [
    'POWER', 'ENERGY', 'ELECTRIC', 'GRID', 
    'WEATHER', 'TEMPERATURE', 'HEAT', 'COLD',
    'FESTIVAL', 'CELEBRATION', 'EVENT', 
    'INFRA', 'OUTAGE', 'BLACKOUT',
    'PROTEST', 'RALLY', 'GATHERING'
]

#============================================================
# DATA LOADING
#============================================================
def load_gkg_data(file_path):
    """Load GDELT GKG data from CSV file"""
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} records")
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
    else:
        print("Error: No theme columns found in the dataset")
        return None

#============================================================
# THEME EXTRACTION & ANALYSIS
#============================================================
def extract_themes(df, theme_column):
    """Extract all themes from the dataset"""
    print(f"Extracting themes from {theme_column} column...")
    
    # Filter out null values
    themes_data = df[theme_column].dropna()
    
    # Split themes (they are typically separated by semicolons)
    all_themes = []
    for themes_str in themes_data:
        # Split by semicolon if present, otherwise try comma
        if ';' in str(themes_str):
            themes = themes_str.split(';')
        else:
            themes = themes_str.split(',')
        
        # Clean up each theme and add to list
        for theme in themes:
            theme = theme.strip()
            if theme:
                all_themes.append(theme)
    
    # Count theme occurrences
    theme_counter = Counter(all_themes)
    return theme_counter

def categorize_themes(theme_counter, categories=THEME_CATEGORIES):
    """Group themes into predefined categories"""
    print("Categorizing themes...")
    category_counts = {category: 0 for category in categories}
    
    for theme, count in theme_counter.items():
        for category, keywords in categories.items():
            if any(keyword in theme.upper() for keyword in keywords):
                category_counts[category] += count
    
    return category_counts

def identify_energy_related_themes(theme_counter, keywords=ENERGY_KEYWORDS):
    """Identify themes that might affect energy load"""
    if not theme_counter:
        return None
    
    print("Identifying energy-related themes...")
    potential_energy_themes = []
    
    for theme in theme_counter:
        if any(keyword in theme.upper() for keyword in keywords):
            potential_energy_themes.append((theme, theme_counter[theme]))
    
    # Sort by count
    potential_energy_themes.sort(key=lambda x: x[1], reverse=True)
    return potential_energy_themes

#============================================================
# VISUALIZATION
#============================================================
def ensure_output_dir(output_dir):
    """Ensure the output directory exists"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_top_themes(theme_counter, n=20, output_dir=OUTPUT_DIR):
    """Plot the top N most common themes"""
    output_dir = ensure_output_dir(output_dir)
    
    plt.figure(figsize=(12, 8))
    top_themes = dict(theme_counter.most_common(n))
    plt.bar(top_themes.keys(), top_themes.values())
    plt.xticks(rotation=90)
    plt.title(f'Top {n} Themes in Delhi GKG Dataset')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'top_themes_delhi.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.show()

def plot_theme_categories(category_counts, output_dir=OUTPUT_DIR):
    """Plot theme categories as a pie chart"""
    output_dir = ensure_output_dir(output_dir)
    
    # Sort categories by count
    sorted_categories = dict(sorted(category_counts.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True))
    
    plt.figure(figsize=(10, 8))
    plt.pie(sorted_categories.values(), labels=sorted_categories.keys(), 
            autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Theme Categories')
    
    output_path = os.path.join(output_dir, 'theme_categories_pie.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.show()

def plot_energy_themes(energy_themes, output_dir=OUTPUT_DIR):
    """Plot energy-related themes"""
    if not energy_themes:
        print("No energy-related themes to plot")
        return
    
    output_dir = ensure_output_dir(output_dir)
    
    # Use only top 15 for better visualization
    top_energy = energy_themes[:15]
    themes, counts = zip(*top_energy)
    
    plt.figure(figsize=(12, 8))
    plt.barh(themes, counts)
    plt.title('Top Energy-Related Themes')
    plt.xlabel('Count')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'energy_related_themes.png')
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.show()

#============================================================
# REPORTING
#============================================================
def print_theme_statistics(theme_counter, category_counts, energy_themes):
    """Print theme analysis statistics"""
    print("\n" + "="*60)
    print("THEME ANALYSIS RESULTS")
    print("="*60)
    
    # General statistics
    print(f"\nFound {len(theme_counter)} unique themes in the dataset")
    
    # Top themes
    print(f"\nTop 20 most common themes:")
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

def save_results_to_csv(theme_counter, category_counts, energy_themes, output_dir=OUTPUT_DIR):
    """Save analysis results to CSV files"""
    output_dir = ensure_output_dir(output_dir)
    
    # Save all themes
    themes_df = pd.DataFrame(theme_counter.most_common(), columns=['Theme', 'Count'])
    themes_path = os.path.join(output_dir, 'all_themes.csv')
    themes_df.to_csv(themes_path, index=False)
    
    # Save categories
    categories_df = pd.DataFrame(list(category_counts.items()), columns=['Category', 'Count'])
    categories_path = os.path.join(output_dir, 'theme_categories.csv')
    categories_df.to_csv(categories_path, index=False)
    
    # Save energy themes
    if energy_themes:
        energy_df = pd.DataFrame(energy_themes, columns=['Theme', 'Count'])
        energy_path = os.path.join(output_dir, 'energy_themes.csv')
        energy_df.to_csv(energy_path, index=False)
    
    print(f"\nResults saved to {output_dir}")

#============================================================
# MAIN EXECUTION
#============================================================
def main(input_file=INPUT_FILE, output_dir=OUTPUT_DIR):
    """Main execution function"""
    # Create output directory
    ensure_output_dir(output_dir)
    
    # Load data
    df = load_gkg_data(input_file)
    if df is None:
        return
    
    # Get theme column
    theme_column = get_theme_column(df)
    if theme_column is None:
        return
    
    # Extract and analyze themes
    theme_counter = extract_themes(df, theme_column)
    
    # Categorize themes
    category_counts = categorize_themes(theme_counter)
    
    # Identify energy-related themes
    energy_themes = identify_energy_related_themes(theme_counter)
    
    # Create visualizations
    plot_top_themes(theme_counter)
    plot_theme_categories(category_counts)
    plot_energy_themes(energy_themes)
    
    # Print and save results
    print_theme_statistics(theme_counter, category_counts, energy_themes)
    save_results_to_csv(theme_counter, category_counts, energy_themes)
    
    return theme_counter, category_counts, energy_themes

if __name__ == "__main__":
    main()