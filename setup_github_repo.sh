#!/bin/bash
# Setup script for GitHub repository ris-localization-mvdr

set -e

REPO_NAME="ris-localization-mvdr"
GITHUB_USER="TahmidZee"

echo "=========================================="
echo "Setting up GitHub repository: $REPO_NAME"
echo "=========================================="
echo ""

# Check if repo already exists locally
if git remote | grep -q "^origin$"; then
    echo "⚠️  Remote 'origin' already exists. Removing..."
    git remote remove origin
fi

# Option 1: Try using GitHub CLI (if available)
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found. Creating repository..."
    gh repo create "$REPO_NAME" --public --description "RIS Localization with MVDR (K-free) - Deep learning + MVDR spectral estimation for multi-source localization" --source=. --remote=origin --push
    echo "✅ Repository created and code pushed!"
    exit 0
fi

# Option 2: Manual instructions
echo "⚠️  GitHub CLI not found. Please create the repository manually:"
echo ""
echo "1. Go to: https://github.com/new"
echo "2. Repository name: $REPO_NAME"
echo "3. Description: RIS Localization with MVDR (K-free) - Deep learning + MVDR spectral estimation for multi-source localization"
echo "4. Set to Public"
echo "5. DO NOT initialize with README, .gitignore, or license (we already have these)"
echo "6. Click 'Create repository'"
echo ""
read -p "Press Enter after you've created the repository on GitHub..."

# Add remote and push
echo ""
echo "Adding remote and pushing code..."
git remote add origin "git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
git branch -M main
git push -u origin main

echo ""
echo "✅ Repository setup complete!"
echo "Repository URL: https://github.com/${GITHUB_USER}/${REPO_NAME}"
