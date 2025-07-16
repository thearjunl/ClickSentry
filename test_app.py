#!/usr/bin/env python3
"""
Test script for ClickSentry - Phishing URL Detection
"""

import requests
import json
import time

def test_web_interface():
    """Test the web interface"""
    print("🌐 Testing Web Interface...")
    
    try:
        response = requests.get("http://localhost:5000")
        if response.status_code == 200:
            print("✓ Web interface is accessible")
            return True
        else:
            print(f"✗ Web interface returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to web interface. Is the server running?")
        return False

def test_api_endpoint():
    """Test the API endpoint"""
    print("\n🔌 Testing API Endpoint...")
    
    test_urls = [
        "https://www.google.com",
        "http://secure-paypal-verification.com",
        "https://www.amazon.com",
        "http://192.168.1.1/paypal-login"
    ]
    
    for url in test_urls:
        try:
            response = requests.post(
                "http://localhost:5000/api/check",
                json={"url": url},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✓ {url} -> {result['result']} ({result['confidence']}%)")
            else:
                print(f"✗ API request failed for {url}: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to API. Is the server running?")
            return False
        except Exception as e:
            print(f"✗ Error testing {url}: {e}")
    
    return True

def test_health_check():
    """Test the health check endpoint"""
    print("\n❤️ Testing Health Check...")
    
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("model_loaded", False):
                print("✓ Health check passed - Model loaded")
                return True
            else:
                print("✗ Health check failed - Model not loaded")
                return False
        else:
            print(f"✗ Health check returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to health endpoint. Is the server running?")
        return False

def main():
    """Main test function"""
    print("🧪 ClickSentry Test Suite")
    print("=" * 30)
    
    # Wait a moment for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(2)
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_web_interface():
        tests_passed += 1
    
    if test_api_endpoint():
        tests_passed += 1
    
    if test_health_check():
        tests_passed += 1
    
    # Summary
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! ClickSentry is working correctly.")
    else:
        print("❌ Some tests failed. Please check the server and try again.")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)