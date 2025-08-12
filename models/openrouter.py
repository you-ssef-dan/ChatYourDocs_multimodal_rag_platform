# models/openrouter.py

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

def load_openrouter_llm(
    default_model: str = "mistralai/mistral-small-3.2-24b-instruct:free"
):

    api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    return client, default_model
