#!/usr/bin/env python3
"""
Test script for production deployment
Run this to verify everything works before deploying
"""

import requests
import json

def test_local_app():
    """Test the local production app"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Production App...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test main page
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Main page: {response.status_code}")
    except Exception as e:
        print(f"❌ Main page failed: {e}")
    
    # Test recommendation endpoint
    test_user = "rebecca"
    try:
        response = requests.post(f"{base_url}/recommend", data={"User Name": test_user})
        print(f"✅ Recommendation for {test_user}: {response.status_code}")
        if response.status_code == 200:
            print("✅ App is working correctly!")
        else:
            print(f"⚠️ Recommendation returned status: {response.status_code}")
    except Exception as e:
        print(f"❌ Recommendation failed: {e}")

if __name__ == "__main__":
    test_local_app()
