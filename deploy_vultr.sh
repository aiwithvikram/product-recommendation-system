#!/bin/bash

# Simple Vultr Deployment Script
echo "🚀 Simple Vultr Deployment Starting..."

# Install only essential packages
echo "📦 Installing essential packages..."
apt install -y python3 python3-pip python3-venv

# Create virtual environment
echo "🐍 Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Download spaCy model
echo "🤖 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Setup systemd service
echo "⚙️ Setting up service..."
cp product-recommendation.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable product-recommendation.service
systemctl start product-recommendation.service

echo "✅ Simple deployment completed!"
echo "🌐 Your app is running at: http://155.138.203.223:5000"
echo "📊 Check status: systemctl status product-recommendation"
