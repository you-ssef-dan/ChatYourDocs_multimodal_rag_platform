# images_ingestion.py
print("🔧 Initializing image ingestion...")
import chromadb
from tqdm import tqdm
import os
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

def ingest_images(user_id, chatbot_id, file_paths=[], persist_dir="database"):
    if not file_paths:
        print("⚠️ No images to ingest!")
        return

    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()

    # separate collection for images
    collection = client.get_or_create_collection(
        name="images_collection",
        embedding_function=embedding_fn,
        data_loader=image_loader
    )

    ids = [f"img_{user_id}_{chatbot_id}_{i}" for i in range(len(file_paths))]
    uris = file_paths

    metadatas = [{
        "user_id": str(user_id),
        "chatbot_id": str(chatbot_id),
        "content_type": "image",
        "source": os.path.basename(path)
    } for path in file_paths]

    print(f"📦 Adding {len(file_paths)} images for user {user_id}, chatbot {chatbot_id}...")
    collection.add(ids=ids, uris=uris, metadatas=metadatas)

    print(f"✅ Image ingestion complete for user {user_id}, chatbot {chatbot_id}.")
