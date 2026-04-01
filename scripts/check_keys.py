import os
from dotenv import load_dotenv
load_dotenv()

keys = [
    "KIMI_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "ZHIPU_API_KEY"
]

for k in keys:
    val = os.getenv(k)
    if val:
        print(f"{k}: {val[:10]}...")
    else:
        print(f"{k}: NOT FOUND")
