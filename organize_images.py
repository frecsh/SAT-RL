import os
import shutil
import glob

"""
This script organizes all image files into an 'images' directory and 
updates the save_examples.py script to look for images there.
"""

# Create images directory if it doesn't exist
IMAGES_DIR = 'images'
os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"Created '{IMAGES_DIR}' directory")

# Also ensure examples directory exists
EXAMPLES_DIR = 'examples'
os.makedirs(EXAMPLES_DIR, exist_ok=True)

# Create models directory for neural network models
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Created '{MODELS_DIR}' directory for saving model checkpoints")

# Define image file patterns to move
image_patterns = [
    "*.png",
    "*.jpg",
    "*.jpeg"
]

# Move image files to the images directory
moved_count = 0
for pattern in image_patterns:
    for img_file in glob.glob(pattern):
        # Skip if the file is already in the images or examples directory
        if img_file.startswith(IMAGES_DIR) or img_file.startswith(EXAMPLES_DIR):
            continue
            
        dest_path = os.path.join(IMAGES_DIR, os.path.basename(img_file))
        try:
            shutil.move(img_file, dest_path)
            moved_count += 1
            print(f"Moved: {img_file} → {dest_path}")
        except Exception as e:
            print(f"Failed to move {img_file}: {str(e)}")

print(f"\nMoved {moved_count} image files to the '{IMAGES_DIR}' directory")

# Update .gitignore to exclude all images except examples
with open('.gitignore', 'r') as f:
    content = f.read()

if 'images/' not in content:
    with open('.gitignore', 'a') as f:
        f.write(f"\n# Image files\n{IMAGES_DIR}/\n!{EXAMPLES_DIR}/*.png\n")
    print("Updated .gitignore to exclude images directory")

# Also add models directory to gitignore but exclude .gitkeep
if 'models/' not in content:
    with open('.gitignore', 'a') as f:
        f.write(f"\n# Model files\n{MODELS_DIR}/*.pth\n{MODELS_DIR}/*.h5\n")
    # Create a .gitkeep file so the directory is tracked
    with open(os.path.join(MODELS_DIR, '.gitkeep'), 'w') as f:
        pass

print("\nOrganization complete!")
print(f"All image files are now in the '{IMAGES_DIR}' directory.")
print("Remember to update your scripts to save new images to this directory.")
print(f"Neural network models will be saved in the '{MODELS_DIR}' directory.")