import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import numpy as np

load_dotenv()

endpoint = os.getenv("AZURE_ENDPOINT")
subscription_key = os.getenv("AZURE_OPENAI_KEY")

api_version = "2024-12-01-preview"
model_name = "text-embedding-ada-002"
deployment = "text-embedding-ada-002"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.embeddings.create(
    input="esempio testo prova",
    model=deployment,
)

vector = response.data[0].embedding
np_vector = np.array(vector)
print(np_vector.shape)




