import urllib.request
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
url = "https://api.vectorengine.ai/v1/chat/completions"

data = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Say hi"}],
    "temperature": 0.7
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers)
try:
    print("Requesting via urllib...")
    with urllib.request.urlopen(req, timeout=10) as response:
        print(f"Status: {response.status}")
        print(f"Response: {response.read().decode('utf-8')[:200]}")
except Exception as e:
    print(f"Failed: {e}")
