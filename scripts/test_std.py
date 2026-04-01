import os
print("OS imported")
api_key_ds = os.environ.get("DEEPSEEK_API_KEY")
print(f"API Key found: {api_key_ds is not None}")
