#!/bin/bash
# Script to clean and reinstall pre-commit hooks

# Step 1: Remove pre-commit cache
echo "Removing pre-commit cache..."
pre-commit clean
rm -rf ~/.cache/pre-commit

# Step 2: Uninstall pre-commit hooks
echo "Uninstalling pre-commit hooks..."
pre-commit uninstall

# Step 3: Force upgrade pre-commit
echo "Upgrading pre-commit..."
pip install --upgrade pre-commit

# Step 4: Reinstall pre-commit hooks
echo "Reinstalling pre-commit hooks..."
pre-commit install

# Step 5: Recreate pre-commit environment
echo "Initializing pre-commit environment..."
pre-commit gc
pre-commit autoupdate

echo "Pre-commit has been thoroughly cleaned and reinstalled!"
echo "Try running 'pre-commit run --all-files' to verify everything works."
