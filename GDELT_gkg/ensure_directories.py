"""
Directory Structure Setup Utility
---------------------------------
This script ensures that all necessary directories for the GDELT processing pipeline exist.
Run this script before starting the pipeline to set up the correct folder structure.
"""

import os
import sys

# Base directory for all outputs
OUTPUT_DIR = r"C:\Users\nikun\Desktop\MLPR\AI_Energy-Load\OUTPUT_DIR"

# Directory structure
DIRECTORIES = {
    # Raw data storage
    "raw_data": {
        "": "...",
        "batch_outputs": "...",
        "cache": "..."
    },
    
    # Processed data
    "processed_data": {
        "": "..."
    },
    
    # Aggregated data
    "aggregated_data": {
        "": "..."
    },
    
    # Analysis outputs
    "analysis_results": {
        "": "..."
    },
    
    # Visualizations
    "figures": {
        "": "..."
    }
}

def ensure_directory_structure():
    """Create the entire directory structure if it doesn't exist"""
    print(f"Setting up directory structure in {OUTPUT_DIR}")
    
    # Check if OUTPUT_DIR is writable
    if not os.access(os.path.dirname(OUTPUT_DIR), os.W_OK):
        print(f"Error: Cannot write to {OUTPUT_DIR}. Check permissions.")
        return
    
    # Create main output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created main output directory: {OUTPUT_DIR}")
    
    # Create all subdirectories
    for main_dir, subdirs in DIRECTORIES.items():
        # Create the main category directory
        main_path = os.path.join(OUTPUT_DIR, main_dir)
        if not os.path.exists(main_path):
            os.makedirs(main_path)
            print(f"Created directory: {main_path}")
        
        # Create subdirectories
        for subdir, description in subdirs.items():
            if subdir:  # Skip the empty string key which is just for the main directory description
                sub_path = os.path.join(main_path, subdir)
                if not os.path.exists(sub_path):
                    os.makedirs(sub_path)
                    print(f"Created subdirectory: {sub_path}")
    
    print("Directory structure setup complete")

def verify_directory_structure():
    """Verify that all required directories exist"""
    missing_dirs = []
    
    for main_dir, subdirs in DIRECTORIES.items():
        main_path = os.path.join(OUTPUT_DIR, main_dir)
        if not os.path.exists(main_path):
            missing_dirs.append(main_path)
            
        for subdir in subdirs.keys():
            if subdir:  # Skip the empty string key
                sub_path = os.path.join(main_path, subdir)
                if not os.path.exists(sub_path):
                    missing_dirs.append(sub_path)
    
    if missing_dirs:
        print("WARNING: The following directories are missing:")
        for dir_path in missing_dirs:
            print(f"  - {dir_path}")
        return False
    else:
        print("All required directories exist")
        return True

if __name__ == "__main__":
    print("GDELT GKG Processing Directory Setup Utility")
    print("--------------------------------------------")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        verify_directory_structure()
    else:
        ensure_directory_structure()
        # Also verify after creation to confirm
        verify_directory_structure()
    
    print("\nDirectory structure overview:")
    for main_dir, subdirs in DIRECTORIES.items():
        print(f"\n{main_dir}/")
        print(f"  {subdirs['']}")  # Print the main directory description
        for subdir, description in subdirs.items():
            if subdir:  # Skip the empty string key
                print(f"  {subdir}/")
                print(f"    {description}")
