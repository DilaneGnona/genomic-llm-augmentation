import requests
import json
import os
from dotenv import load_dotenv
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("Script starting...")
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
url = "https://api.vectorengine.ai/v1/chat/completions"

print("Testing DeepSeek with verify=False...")
data = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Generate 2 lines of CSV for SNPs: Sample_ID, SNP1, Target"}],
    "temperature": 0.7
}
try:
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data, timeout=30, verify=False)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response:\n{response.json()['choices'][0]['message']['content']}")
    else:
        print(f"Error: {response.text}")
except Exception as e:
    print(f"Failed: {e}")
