#!/bin/bash

# Vultr Server Deployment Script
# Run this on your Vultr server

echo "🚀 Starting Vultr Server Deployment..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install required packages
echo "🔧 Installing required packages..."
apt install -y python3 python3-pip python3-venv nginx git

# Create application directory
echo "📁 Setting up application directory..."
mkdir -p /root/product-recommendation-system
cd /root/product-recommendation-system

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python packages
echo "📚 Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "🤖 Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Download NLTK data
echo "📚 Downloading NLTK data..."
python -c "
import nltk
import os
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
print('NLTK data downloaded successfully')
"

# Setup systemd service
echo "⚙️ Setting up systemd service..."
cp product-recommendation.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable product-recommendation.service

# Setup Nginx
echo "🌐 Setting up Nginx..."
cat > /etc/nginx/sites-available/product-recommendation << 'EOF'
server {
    listen 80;
    server_name your-domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable site
ln -sf /etc/nginx/sites-available/product-recommendation /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test Nginx config
nginx -t

# Start services
echo "🚀 Starting services..."
systemctl start product-recommendation.service
systemctl restart nginx

# Show status
echo "📊 Service Status:"
systemctl status product-recommendation.service --no-pager -l

echo "✅ Deployment completed!"
echo "🌐 Your app should be running at: http://your-server-ip"
echo "📝 Don't forget to:"
echo "   1. Replace 'your-domain.com' in Nginx config with your actual domain"
echo "   2. Configure firewall: ufw allow 80, ufw allow 443"
echo "   3. Set up SSL certificate with Let's Encrypt if needed"
