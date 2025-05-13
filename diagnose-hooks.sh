#!/bin/bash
# Script to diagnose which pre-commit hook is causing issues

# Step 1: Make backup of original config
cp .pre-commit-config.yaml .pre-commit-config.yaml.bak
echo "Backed up original config to .pre-commit-config.yaml.bak"

# Step 2: Clean environment
./pre-commit-clean.sh

# Step 3: Test each hook individually
echo "Testing hooks one by one to identify problematic one..."

# Core hooks to test
HOOKS=(
    "trailing-whitespace"
    "end-of-file-fixer"
    "check-yaml"
    "black"
    "isort"
    "flake8"
    "autoflake"
    "autopep8"
    "mypy"
    "pyupgrade"
    "pydocstyle"
    "prettier"
    "interrogate"
    "bandit"
    "vulture"
    "toml-sort"
    "yesqa"
)

# Test each hook
for hook in "${HOOKS[@]}"; do
    echo "------------------------------"
    echo "Testing hook: $hook"
    echo "------------------------------"
    cp debug-pre-commit-config.yaml .pre-commit-config.yaml
    pre-commit clean
    if pre-commit run "$hook" --verbose || echo "Hook $hook FAILED with exit code $?"; then
        echo "✅ Hook $hook PASSED"
    else
        echo "❌ Hook $hook FAILED!"
    fi
done

# Step 4: Restore original config
echo "Restoring original config..."
mv .pre-commit-config.yaml.bak .pre-commit-config.yaml

echo "Diagnosis complete. Check the output above to see which hooks failed."
