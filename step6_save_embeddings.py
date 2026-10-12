
import json
from step1_chunks import load_document, split_into_chunks


import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY")

EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}


def get_embeddings(texts, input_type):
    data = {
        "model": "nvidia/nemotron-3-embed-1b",
        "input": texts,
        "input_type": input_type,
        "encoding_format": "float",
        "truncate": "NONE"
    }

    response = requests.post(
        EMBED_URL,
        headers=HEADERS,
        json=data,
        timeout=120
    )
    response.raise_for_status()

    return [
        item["embedding"]
        for item in response.json()["data"]
    ]



# 1. Load the insurance policy
policy_text = load_document("insurance_policy.txt")

# 2. Split it into chunks
chunks = split_into_chunks(policy_text)

# 3. Generate embeddings for all chunks
embeddings = get_embeddings(chunks, "passage")

# 4. Combine each chunk with its embedding
records = []

for chunk, embedding in zip(chunks, embeddings):
    records.append({
        "text": chunk,
        "embedding": embedding
    })

# 5. Save everything to a JSON file
with open("insurance_embeddings.json", "w", encoding="utf-8") as file:
    json.dump(records, file)

print(f"Saved {len(records)} chunks and their embeddings.")
