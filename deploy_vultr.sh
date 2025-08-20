#!/bin/bash

# Simple Vultr Deployment Script
echo "🚀 Simple Vultr Deployment Starting..."

# Install Python and essential packages
echo "📦 Installing Python and essential packages..."
apt update
apt install -y python3 python3-pip python3-venv

# Check Python version
echo "🐍 Python version:"
python3 --version

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip first
echo "📚 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Download spaCy model (for old version)
echo "🤖 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Setup systemd service
echo "⚙️ Setting up service..."
cp product-recommendation.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable product-recommendation.service
systemctl start product-recommendation.service

echo "✅ Simple deployment completed!"
echo "🌐 Your app is running at: http://144.202.30.217:5000"
echo "📊 Check status: systemctl status product-recommendation"
