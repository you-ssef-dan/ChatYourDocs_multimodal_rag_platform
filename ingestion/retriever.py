# retriever.py

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from .vectordb import vector_db  # Singleton instance for text vectorstore

DEFAULT_PERSIST_DIR = "database"


def get_text_retriever(collection_name: str = "documents", k: int = 5):
    """
    Returns a retriever object for text documents.
    Uses LangChain-compatible retriever from Chroma.
    """
    return vector_db.get_collection(collection_name).as_retriever(search_kwargs={"k": k})


def get_image_retriever(collection_name: str = "images", k: int = 5, persist_dir: str = DEFAULT_PERSIST_DIR):
    """
    Returns a function that performs similarity search on an image collection using a text query.
    Ensures embedding_function is provided to embed the text query.
    """
    client = chromadb.PersistentClient(path=persist_dir) # Use persistent client for image collection
    embedding_fn = OpenCLIPEmbeddingFunction() # Embedding function for text queries

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn  # needed for text queries
    )

    def retrieve_by_text(query_text: str):
        result = collection.query(
            query_texts=[query_text],
            n_results=k,
            include=["uris", "distances"]
        )
        # Return image URIs
        return result.get("uris", [[]])[0]  # safe flattening [[list]] → [list]

    return retrieve_by_text


def search_all(query: str, k: int = 5):
    """
    Performs global search over both text and image collections.
    Returns top-k results from both collections.
    """
    print(f"\n🔎 Global search for: '{query}'")

    # Text results
    text_retriever = get_text_retriever(k=k) # Get text retriever
    text_docs = text_retriever.invoke(query) # Get relevant text documents
    text_results = [doc.page_content for doc in text_docs] # Extract text content

    # Image results
    image_retriever = get_image_retriever(k=k) # Get image retriever
    image_results = image_retriever(query) # get relevant image URIs

    return {
        "texts": text_results,
        "images": image_results
    }


# Optional CLI testing
if __name__ == "__main__":
    query = input("Enter your query: ")
    results = search_all(query)

    print("\n📄 Top text results:")
    for i, res in enumerate(results["texts"], 1):
        print(f"{i}. {res[:200]}...")

    print("\n🖼️ Top image URIs:")
    for i, uri in enumerate(results["images"], 1):
        print(f"{i}. {uri}")
