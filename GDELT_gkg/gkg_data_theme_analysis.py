import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# Update this line
INPUT_FILE = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\GDELT_gkg\gkg_datasets\delhi_gkg_data_2021_jan1_3.csv"

def extract_and_analyze_themes(file_path):
    """Extract and analyze all themes in the dataset"""
    # Load the dataset
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    
    # Check if V2Themes column exists
    if 'V2Themes' not in df.columns and 'Themes' not in df.columns:
        print("Error: No theme columns found in the dataset")
        return None
    
    # Use V2Themes if available, otherwise fall back to Themes
    theme_column = 'V2Themes' if 'V2Themes' in df.columns else 'Themes'
    
    # Extract all themes
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
            # Extract the actual theme name (remove any metadata)
            # GDELT GKG themes often have patterns like "WB_XXX" or "TAX_XXX"
            theme = theme.strip()
            if theme:
                all_themes.append(theme)
    
    # Count theme occurrences
    theme_counter = Counter(all_themes)
    
    # Display total number of unique themes
    print(f"\nFound {len(theme_counter)} unique themes in the dataset")
    
    # Display top themes
    top_n = 50
    print(f"\nTop {top_n} most common themes:")
    for theme, count in theme_counter.most_common(top_n):
        print(f"{theme}: {count}")
    
    # Plot top themes
    plt.figure(figsize=(12, 8))
    top_themes = dict(theme_counter.most_common(20))
    plt.bar(top_themes.keys(), top_themes.values())
    plt.xticks(rotation=90)
    plt.title('Top 20 Themes in Delhi GKG Dataset')
    plt.tight_layout()
    plt.savefig('top_themes_delhi.png')
    plt.show()
    
    # Group themes by category
    theme_categories = {
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
    
    # Count themes by category
    category_counts = {category: 0 for category in theme_categories}
    for theme, count in theme_counter.items():
        for category, keywords in theme_categories.items():
            if any(keyword in theme.upper() for keyword in keywords):
                category_counts[category] += count
    
    # Print category counts
    print("\nTheme counts by category:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count}")
    
    # Return theme counter for further analysis
    return theme_counter

def identify_energy_related_themes(theme_counter):
    """Identify themes that might affect energy load"""
    if not theme_counter:
        return None
        
    # Keywords related to energy consumption
    energy_keywords = [
        'POWER', 'ENERGY', 'ELECTRIC', 'GRID', 
        'WEATHER', 'TEMPERATURE', 'HEAT', 'COLD',
        'FESTIVAL', 'CELEBRATION', 'EVENT', 
        'INFRA', 'OUTAGE', 'BLACKOUT',
        'PROTEST', 'RALLY', 'GATHERING'
    ]
    
    # Find themes containing these keywords
    potential_energy_themes = []
    for theme in theme_counter:
        if any(keyword in theme.upper() for keyword in energy_keywords):
            potential_energy_themes.append((theme, theme_counter[theme]))
    
    # Sort by count
    potential_energy_themes.sort(key=lambda x: x[1], reverse=True)
    
    # Print the results
    print("\nPotential energy-related themes:")
    if potential_energy_themes:
        for theme, count in potential_energy_themes:
            print(f"{theme}: {count}")
    else:
        print("No energy-related themes found.")
        
    return potential_energy_themes

if __name__ == "__main__":
    theme_counter = extract_and_analyze_themes(INPUT_FILE)
    if theme_counter:
        identify_energy_related_themes(theme_counter)