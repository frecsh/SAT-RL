import os
import shutil
import glob

"""
This script copies selected visualization outputs from the images directory 
to the examples directory for inclusion in the GitHub repository.
"""

# Create examples directory if it doesn't exist
os.makedirs('examples', exist_ok=True)

# Define image directory
IMAGES_DIR = 'images'

# Define patterns to copy
patterns = [
    # Main comparison results
    "problem_results_*.png",
    # Communication visualizations
    "comm_variables_*.png",
    "comm_clauses_*.png",
    "comm_heatmap_*.png",
    # Oracle comparisons
    "oracle_weight_comparison_*.png",
    "oracle_learning_curves_*.png",
    "oracle_clause_difficulty_*.png",
    # Threshold comparisons
    "threshold_success_*.png",
    "threshold_episodes_*.png"
]

# Copy matched files to examples directory
copied_files = []
for pattern in patterns:
    # Look in images directory
    for file in glob.glob(os.path.join(IMAGES_DIR, pattern)):
        dest_file = os.path.join('examples', os.path.basename(file))
        try:
            shutil.copy2(file, dest_file)
            copied_files.append(dest_file)
            print(f"Copied: {file} → {dest_file}")
        except Exception as e:
            print(f"Failed to copy {file}: {str(e)}")
    
    # Also check the root directory in case some files haven't been moved yet
    for file in glob.glob(pattern):
        dest_file = os.path.join('examples', os.path.basename(file))
        try:
            shutil.copy2(file, dest_file)
            copied_files.append(dest_file)
            print(f"Copied: {file} → {dest_file}")
        except Exception as e:
            print(f"Failed to copy {file}: {str(e)}")

print(f"\nCopied {len(copied_files)} example files to the examples/ directory")
print("These will be included in the GitHub repository")