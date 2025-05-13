#!/bin/bash

# Script to set up the development environment for the SAT-RL project

echo "Setting up development environment for SAT-RL project..."

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip not found. Please install Python and pip first."
    exit 1
fi

# Check if pre-commit is installed, install if not
if ! command -v pre-commit &> /dev/null; then
    echo "Installing pre-commit..."
    pip install pre-commit
fi

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Install project dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing project dependencies..."
    pip install -r requirements.txt
fi

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov pytest-benchmark black flake8 isort mypy pydocstyle

echo "Development environment setup complete!"
echo "Pre-commit hooks are now active and will run automatically on each commit."
echo "To run them manually on all files, use: pre-commit run --all-files"
