from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    embeddings = OllamaEmbeddings(model="llama3", base_url="http://10.151.11.37/ollama", headers={"Authorization": "Bearer sk-a97954c0344b4424896185ce12d8b804"})
    return embeddings
