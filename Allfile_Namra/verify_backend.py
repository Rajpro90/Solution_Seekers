import requests
import json
import threading
import time
import sys
import os

# Add local project path and backend path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from backend.run import app

def run_server():
    app.run(port=5000, use_reloader=False)

def test_endpoints():
    base_url = "http://127.0.0.1:5000"
    
    # Wait for server to start
    time.sleep(2)
    
    print("\n--- Testing Health Check ---")
    try:
        resp = requests.get(f"{base_url}/")
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.json()}")
    except Exception as e:
        print(f"Health Check Failed: {e}")

    print("\n--- Testing API Prediction (POST /api/predict/aqi) ---")
    payload = {"latitude": 28.7041, "longitude": 77.1025} # Delhi
    try:
        resp = requests.post(f"{base_url}/api/predict/aqi", json=payload)
        print(f"Status: {resp.status_code}")
        if resp.status_code == 200:
            print("Success! Prediction received.")
            # Print truncated response to avoid spam
            data = resp.json()
            print(f"Location: {data.get('location')}")
            print(f"Forecast (Day 1): {data.get('forecast', [{}])[0]}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Prediction Failed: {e}")

    print("\n--- Testing History API (GET /api/history) ---")
    try:
        resp = requests.get(f"{base_url}/api/history")
        print(f"Status: {resp.status_code}")
        data = resp.json()
        print(f"Records found: {len(data)}")
    except Exception as e:
        print(f"History Fetch Failed: {e}")

if __name__ == "__main__":
    # Start server in background thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    test_endpoints()
    print("\nTest completed. You can now press Ctrl+C to exit if running manually.")
