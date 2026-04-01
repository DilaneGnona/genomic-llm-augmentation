import requests
url = "https://api.vectorengine.ai/v1/models"
# No auth needed just to see if it's reachable
try:
    print(f"Connecting to {url}...")
    response = requests.get(url, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Content: {response.text[:200]}")
except Exception as e:
    print(f"Failed: {e}")
