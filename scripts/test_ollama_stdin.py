import subprocess

try:
    print("Testing subprocess with stdin...")
    p = subprocess.run(['ollama', 'run', 'glm-4.6:cloud'], input="hello", capture_output=True, encoding='utf-8', timeout=60)
    print(f"Return code: {p.returncode}")
    print(f"Stdout: {p.stdout[:100]}")
    print(f"Stderr: {p.stderr}")
except Exception as e:
    print(f"Error: {e}")
