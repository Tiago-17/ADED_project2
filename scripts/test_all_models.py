import requests
import time

PROMPT = "What is the capital of Portugal?"

# Modelos com portas correspondentes
models = {
    "qwen2.5-0.5b": {
        "url": "http://localhost:8080/completion",
        "payload": {
            "prompt": PROMPT,
            "n_predict": 100,
            "temperature": 0.7,
            "stream": False,
        },
    },
    "tinyllama-1.1b": {
        "url": "http://localhost:8081/v1/chat/completions",
        "payload": {
            "messages": [
                {"role": "user", "content": PROMPT},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "stream": False,
        },
    },
    "meta-llama-3.1-8b": {
        "url": "http://localhost:8082/completion",
        "payload": {
            "prompt": PROMPT,
            "n_predict": 100,
            "temperature": 0.7,
            "stream": False,
        },
    },
}


def extract_output(data):
    if not isinstance(data, dict):
        return None

    for key in ("content", "response", "text", "completion"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                value = message.get("content")
                if isinstance(value, str) and value.strip():
                    return value

            value = first_choice.get("text")
            if isinstance(value, str) and value.strip():
                return value

    return None

for model_name, url in models.items():
    if isinstance(url, dict):
        endpoint = url["url"]
        payload = url["payload"]
    else:
        endpoint = url
        payload = {
            "prompt": PROMPT,
            "n_predict": 100,
            "temperature": 0.7,
            "stream": False,
        }

    print("\n" + "="*60)
    print(f"Testing model: {model_name}")
    print(f"URL: {endpoint}")
    print("="*60)

    start = time.time()

    try:
        r = requests.post(
            endpoint,
            json=payload,
            timeout=120
        )

        if r.status_code != 200:
            print("Error:", r.text)
            continue

        data = r.json()
        output = extract_output(data)

        end = time.time()

        if output is None:
            print("Response: <empty>")
            print("Raw JSON:", data)
        else:
            print("Response:", output)
        print(f"Time: {end - start:.2f}s")

    except Exception as e:
        print("Request failed:", e)
