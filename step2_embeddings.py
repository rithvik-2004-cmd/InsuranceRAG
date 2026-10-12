
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

url = "https://integrate.api.nvidia.com/v1/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


data = {
    "model": "nvidia/nemotron-3-embed-1b",
    "input": [
        "A deductible is the amount paid by the policyholder before insurance coverage applies."
    ],
    "input_type": "query",
    "encoding_format": "float",
    "truncate": "NONE"
}

try:
    response = requests.post(
        url,
        headers=headers,
        json=data,
        timeout=60
    )

    print("Status code:", response.status_code)
    response.raise_for_status()

    result = response.json()
    embedding = result["data"][0]["embedding"]

    print("Embedding dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])

except requests.exceptions.RequestException as error:
    print("Embedding request failed:", error)
    if error.response is not None:
        print("Response details:", error.response.text)
