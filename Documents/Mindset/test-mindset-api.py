#!/usr/bin/env python3
"""
Simple test script for MINDSET API.
"""
import sys
import os
import json
import requests
from pprint import pprint

# Get the API URL from command line or use default
api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

print(f"Testing MINDSET API at: {api_url}")
print("==================================")

try:
    # Test root endpoint
    print("\n1. Testing root endpoint...")
    response = requests.get(f"{api_url}/")
    print(f"Status: {response.status_code}")
    pprint(response.json())

    # Test health endpoint
    print("\n2. Testing health endpoint...")
    response = requests.get(f"{api_url}/health")
    print(f"Status: {response.status_code}")
    pprint(response.json())

    # Test articles endpoint
    print("\n3. Testing articles endpoint...")
    response = requests.get(f"{api_url}/articles")
    print(f"Status: {response.status_code}")
    articles = response.json()
    print(f"Retrieved {len(articles)} articles")
    if articles:
        print("First article:")
        pprint(articles[0])

    # Test analysis endpoint
    print("\n4. Testing analysis endpoint...")
    test_text = "This is a test article about government policies and climate change. The devastating effects of global warming require immediate action."
    response = requests.post(
        f"{api_url}/analyze", 
        json={"text": test_text}
    )
    print(f"Status: {response.status_code}")
    print("Transparency metrics:")
    pprint(response.json())

    print("\nAll tests completed successfully!")

except Exception as e:
    print(f"Error testing API: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure the API is running and accessible")
    print("2. Check if the endpoint is correct")
    print("3. Verify network connectivity")
