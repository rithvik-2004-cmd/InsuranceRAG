
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    print("API key not found. Check your .env file.")
else:
    print("API key loaded successfully!")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "z-ai/glm-5.3-flash",
        "messages": [
            {"role": "user", "content": "Explain insurance in one sentence."}
        ],
        "max_tokens": 150,
        "temperature": 0.2,
        "stream": False
    }

try:
    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=(15, 120)
    )

    print("Status code:", response.status_code)
    response.raise_for_status()

    result = response.json()
    message = result["choices"][0]["message"]

    print("Answer:", message.get("content"))
    print("Reasoning:", message.get("reasoning_content"))
    print("Token usage:", result.get("usage"))

except requests.exceptions.Timeout:
    print("NVIDIA API timed out. We will troubleshoot it separately.")

except requests.exceptions.RequestException as e:
    print("API request failed:", e)