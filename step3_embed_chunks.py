
import os
import requests
from dotenv import load_dotenv
from step1_chunks import load_document, split_into_chunks

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

url = "https://integrate.api.nvidia.com/v1/embeddings"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 1. Read and split the insurance document
text = load_document("insurance_policy.txt")
chunks = split_into_chunks(text)

print("Total chunks:", len(chunks))

# 2. Generate an embedding for each chunk
data = {
    "model": "nvidia/nemotron-3-embed-1b",
    "input": chunks,
    "input_type": "passage",
    "encoding_format": "float",
    "truncate": "NONE"
}

response = requests.post(
    url,
    headers=headers,
    json=data,
    timeout=120
)

print("Status code:", response.status_code)
response.raise_for_status()

result = response.json()
embeddings = [item["embedding"] for item in result["data"]]

# 3. Check the results
print("Total embeddings:", len(embeddings))
print("Dimensions per embedding:", len(embeddings[0]))

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}: {len(embeddings[i])} numbers")
