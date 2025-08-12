# models/groq.py

from dotenv import load_dotenv
import os
from groq import Groq

load_dotenv()

def load_groq_llm(default_model: str = "llama3-8b-8192"):
    """
    Initialize a Groq LLM client.

    Args:
        default_model (str): The Groq model ID to use.
                             Example: "llama3-8b-8192", "mixtral-8x7b-32768", etc.

    Returns:
        tuple: (client, default_model)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    client = Groq(api_key=api_key)
    return client, default_model
