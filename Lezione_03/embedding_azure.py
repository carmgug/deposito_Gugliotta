import os

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

api_version = "2024-12-01-preview"
endpoint = os.getenv("ENDPOINT")
model_name = "text-embedding-ada-002"
deployment = "text-embedding-ada-002"
subscription_key = os.getenv("SUBSCRIPTION_KEY")


client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.embeddings.create(
    input="text da",
    model=deployment,
)

vector = response.data[0].embedding
np_vector = np.array(vector)
print(np_vector.shape)
