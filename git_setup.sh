#!/bin/bash

# This script initializes git, adds your files, and prepares for the first commit
# Before running, create your repository on GitHub and note the URL

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up git repository for SAT+RL project...${NC}"

# Check if git is already initialized
if [ -d .git ]; then
    echo -e "${RED}Git repository already initialized.${NC}"
else
    echo -e "Initializing git repository..."
    git init
    echo -e "${GREEN}Git repository initialized.${NC}"
fi

# Create examples directory if it doesn't exist
if [ ! -d examples ]; then
    echo -e "Creating examples directory..."
    mkdir -p examples
    echo -e "${GREEN}Created examples directory.${NC}"
fi

# First run the script to copy example files
echo -e "Copying example files..."
python save_examples.py

# Add all files
echo -e "Adding files to git..."
git add .

echo -e "${GREEN}Files added to git staging area.${NC}"
echo -e "${BLUE}Ready for first commit.${NC}"
echo
echo -e "Now run the following commands to push to GitHub:"
echo -e "${GREEN}git commit -m \"Initial commit of SAT+RL project\"${NC}"
echo -e "${GREEN}git branch -M main${NC}"
echo -e "${GREEN}git remote add origin <your-github-repo-url>${NC}"
echo -e "${GREEN}git push -u origin main${NC}"