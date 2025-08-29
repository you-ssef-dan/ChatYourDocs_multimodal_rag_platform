#retriever.py
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import torch

DEFAULT_PERSIST_DIR = "database"

def get_text_embedding_function():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-xl",
        model_kwargs={'device': device},
        encode_kwargs={
            "prompt": "Represent the document for retrieval:",
            "normalize_embeddings": True
        },
        query_encode_kwargs={
            "prompt": "Represent the question for retrieving supporting documents:",
            "normalize_embeddings": True
        }
    )

def get_image_embedding_function():
    return OpenCLIPEmbeddingFunction()

def get_text_retriever(user_id, chatbot_id, k=5, persist_dir=DEFAULT_PERSIST_DIR):
    embedding_fn = get_text_embedding_function()
    chroma = Chroma(
        collection_name="docs_collection",
        persist_directory=persist_dir,
        embedding_function=embedding_fn
    )
    filter_dict = {
        "$and": [
            {"user_id": {"$eq": str(user_id)}},
            {"chatbot_id": {"$eq": str(chatbot_id)}},
            {"content_type": {"$eq": "text"}}
        ]
    }
    return chroma.as_retriever(search_kwargs={"k": k, "filter": filter_dict})

def get_image_retriever(user_id, chatbot_id, k=5, persist_dir=DEFAULT_PERSIST_DIR):
    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = get_image_embedding_function()
    image_loader = ImageLoader()

    collection = client.get_collection(
        name="images_collection",
        embedding_function=embedding_fn,
        data_loader=image_loader
    )

    def retrieve_by_text(query_text):
        filter_dict = {
            "$and": [
                {"user_id": {"$eq": str(user_id)}},
                {"chatbot_id": {"$eq": str(chatbot_id)}},
                {"content_type": {"$eq": "image"}}
            ]
        }
        result = collection.query(
            query_texts=[query_text],
            n_results=k,
            where=filter_dict,
            include=["uris", "metadatas", "distances"]
        )
        uris = result.get("uris", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        return list(zip(uris, metadatas)) if uris and metadatas else []
    return retrieve_by_text
