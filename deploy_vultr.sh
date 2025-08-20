#!/bin/bash

# Simple Vultr Deployment Script
echo "🚀 Simple Vultr Deployment Starting..."

# Update system and install dependencies
echo "📦 Updating system and installing dependencies..."
apt update
apt install -y software-properties-common wget curl build-essential

# Add deadsnakes PPA for latest Python
echo "🐍 Adding Python PPA..."
add-apt-repository ppa:deadsnakes/ppa -y
apt update

# Install Python 3.11 and essential packages
echo "📦 Installing Python 3.11 and essential packages..."
apt install -y python3.11 python3.11-venv python3.11-pip python3.11-dev

# Create virtual environment with Python 3.11
echo "🐍 Setting up Python 3.11 environment..."
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip first
echo "📚 Upgrading pip..."
pip install --upgrade pip

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
echo "🌐 Your app is running at: http://144.202.30.217:5000"
echo "📊 Check status: systemctl status product-recommendation"
