import requests
import json
import os
from dotenv import load_dotenv

print("Script starting...")
load_dotenv()
api_key_kimi = os.getenv("KIMI_API_KEY")
url = "https://api.vectorengine.ai/v1/chat/completions"

print(f"Testing Kimi (moonshot-v1-8k)...")
data = {
    "model": "moonshot-v1-8k",
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.7
}
try:
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key_kimi}"}, json=data, timeout=30)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {response.json()['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
