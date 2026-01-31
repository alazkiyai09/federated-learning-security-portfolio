#!/bin/bash
# Script to push the portfolio to GitHub after creating the repo

echo "=== Pushing to GitHub ==="
echo "Make sure you've created the repository at:"
echo "https://github.com/new"
echo ""
echo "Repository name: federated-learning-security-portfolio"
echo "Description: 30-Day Portfolio: Federated Learning Security Research"
echo "Visibility: Public"
echo ""
echo "After creating the repo, press Enter to continue..."
read

# Add remote (if not already added)
git remote add origin git@github.com:alazkiyai09/federated-learning-security-portfolio.git 2>/dev/null

# Push to GitHub
git push -u origin main

echo ""
echo "✓ Code pushed successfully!"
echo "Repository URL: https://github.com/alazkiyai09/federated-learning-security-portfolio"
