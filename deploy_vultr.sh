#!/bin/bash

# AlmaLinux/RHEL Deployment Script
echo "🚀 AlmaLinux/RHEL Deployment Starting..."

# Update system and install dependencies
echo "📦 Updating system and installing dependencies..."
yum update -y
yum install -y wget curl gcc gcc-c++ make openssl-devel bzip2-devel libffi-devel zlib-devel

# Install Python 3.11 from source (AlmaLinux 8 doesn't have Python 3.11 in repos)
echo "🐍 Installing Python 3.11 from source..."
cd /tmp
wget https://www.python.org/ftp/python/3.11.13/Python-3.11.13.tgz
tar -xzf Python-3.11.13.tgz
cd Python-3.11.13

# Configure and compile Python
./configure --enable-optimizations --prefix=/usr/local
make -j $(nproc)
make altinstall

# Create symlink
ln -sf /usr/local/bin/python3.11 /usr/local/bin/python3.11
ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3.11

# Go back to app directory
cd /root/product-recommendation-system

# Remove old virtual environment if it exists
echo "🧹 Cleaning up old environment..."
rm -rf venv

# Create virtual environment with Python 3.11
echo "🐍 Setting up Python 3.11 environment..."
/usr/local/bin/python3.11 -m venv venv
source venv/bin/activate

# Verify we're using Python 3.11
echo "🐍 Python version in venv:"
python --version

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

echo "✅ AlmaLinux deployment completed!"
echo "🌐 Your app is running at: http://144.202.30.217:5000"
echo "📊 Check status: systemctl status product-recommendation"
