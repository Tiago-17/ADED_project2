import os
import requests
import time

URL = "http://localhost:8080/completion"
MODELS_DIR = "models"
models = [m for m in os.listdir(MODELS_DIR) if m.endswith(".gguf")]

PROMPT = "What is the capital of Portugal?"

for model in models:
    print("\n" + "="*60)
    print(f"Testing model: {model}")
    print("="*60)

    start = time.time()

    try:
        r = requests.post(
            URL,
            json={
                "prompt": PROMPT,
                "n_predict": 100,
                "temperature": 0.7,
                "stream": False
            },
            timeout=120
        )

        if r.status_code != 200:
            print("Error:", r.text)
            continue

        data = r.json()
        output = data.get("content") or data.get("response")

        end = time.time()

        print("Response:", output)
        print(f"Time: {end - start:.2f}s")

    except Exception as e:
        print("Request failed:", e)