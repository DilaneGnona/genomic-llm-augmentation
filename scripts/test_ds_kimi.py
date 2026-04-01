print("Script starting...")
import requests
import json
import os
from dotenv import load_dotenv

print("Loading env...")
load_dotenv()
api_key_ds = os.getenv("DEEPSEEK_API_KEY")
api_key_kimi = os.getenv("KIMI_API_KEY")
url = "https://api.vectorengine.ai/v1/chat/completions"

def test_model(model_name, api_key, display_name):
    print(f"Testing {display_name} ({model_name})...")
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Generate 2 lines of CSV for SNPs: Sample_ID, SNP1, Target"}],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data, timeout=30)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()['choices'][0]['message']['content'][:100]}...")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Failed: {e}")

test_model("deepseek-chat", api_key_ds, "DeepSeek")
print("-" * 20)
test_model("moonshot-v1-8k", api_key_kimi, "Kimi")
