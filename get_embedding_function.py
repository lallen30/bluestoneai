from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv() 
import os


def get_embedding_function():
    api_key = os.getenv('OPENAI_API_KEY')
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings
