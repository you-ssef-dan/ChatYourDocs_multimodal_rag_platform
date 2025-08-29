# models/groq.py

from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

def load_groq_llm(default_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    client = Groq(api_key=api_key)
    return client, default_model
