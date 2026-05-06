import requests
import time

PROMPT = "What is the capital of Portugal?"

# Modelos com portas correspondentes
models = {
    "qwen2.5-0.5b": "http://localhost:8080/completion",
    "tinyllama-1.1b": "http://localhost:8081/completion",
    "meta-llama-3.1-8b": "http://localhost:8082/completion"
}

for model_name, url in models.items():
    print("\n" + "="*60)
    print(f"Testing model: {model_name}")
    print(f"URL: {url}")
    print("="*60)

    start = time.time()

    try:
        r = requests.post(
            url,
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
