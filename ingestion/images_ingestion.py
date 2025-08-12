# images_ingestion.py
print("🔧 Initializing image ingestion...")
import chromadb
from tqdm import tqdm
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

print("🔧 Initializing image ingestion...")

def ingest_images(collection_name="images", file_paths=[], persist_dir="database"):
    """
    Ingest image files into ChromaDB using OpenCLIP embeddings.
    Assumes file_paths only contain valid image paths.
    """
    if not file_paths:
        print("⚠️ No images to ingest!")
        return

    client = chromadb.PersistentClient(path=persist_dir)
    embedding_fn = OpenCLIPEmbeddingFunction()
    image_loader = ImageLoader()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        data_loader=image_loader
    )

    ids = [str(i) for i in range(len(file_paths))]
    uris = file_paths

    print(f"📦 Adding {len(uris)} images to '{collection_name}'...")
    for idx, uri in tqdm(zip(ids, uris), total=len(uris), desc="🖼️ Ingesting images"):
        collection.add(ids=[idx], uris=[uri])

    print(f"✅ Ingested {len(uris)} images into '{collection_name}'.")
    print(" image ingestion complete.")
