
import os
import requests
from dotenv import load_dotenv
from step1_chunks import load_document, split_into_chunks
from step4_similarity import dot, magnitude, cosine_similarity

load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

url = "https://integrate.api.nvidia.com/v1/embeddings"

headers = {
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
        url, headers=headers, json=data, timeout=120
    )
    response.raise_for_status()

    result = response.json()
    return [item["embedding"] for item in result["data"]]


# 1. Load the insurance policy
text = load_document("insurance_policy.txt")
chunks = split_into_chunks(text)

# 2. Embed every document chunk
chunk_embeddings = get_embeddings(chunks, "passage")

# 3. Ask a question
question = "What amount do I pay before the insurer starts paying?"

# 4. Embed the question
question_embedding = get_embeddings([question], "query")[0]

# 5. Compare the question with every chunk
results = []

for i, chunk_embedding in enumerate(chunk_embeddings):
    score = cosine_similarity(question_embedding, chunk_embedding)
    results.append((score, i, chunks[i]))

# 6. Sort by highest similarity
results.sort(reverse=True, key=lambda item: item[0])

print("\nQuestion:", question)
print("\nMost relevant chunks:")

for score, index, chunk in results:
    print(f"\nChunk {index + 1} | Similarity: {score:.4f}")
    print(chunk)
