import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
url = "https://api.vectorengine.ai/v1/chat/completions"

prompt = "Generate 5 lines of CSV data for genetic SNPs. Columns: Sample_ID, SNP1, SNP2, Target. Use 0,1,2 for SNPs."

data = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0.7
}

print("Sending request...")
response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=data)
print(f"Status: {response.status_code}")
if response.status_code == 200:
    content = response.json()['choices'][0]['message']['content']
    print("Content received:")
    print(content)
else:
    print(f"Error: {response.text}")
