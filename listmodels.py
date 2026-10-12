import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Point the OpenAI client to NVIDIA's base URL and use your NVIDIA API key
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY")
)

# List all available models hosted on NVIDIA NIM
models = client.models.list()

print("--- Available NVIDIA Models ---")
for m in sorted(models.data, key=lambda m: m.id):
    print(m.id)