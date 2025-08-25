# images_ingestion.py
print("🔧 Initializing image ingestion...")
import chromadb
from tqdm import tqdm
import os
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader

print("🔧 Initializing image ingestion...")

def ingest_images(collection_name="images", file_paths=[], persist_dir="database"):
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

    # Print file names instead of just count
    image_names = [os.path.basename(path) for path in uris]
    print(f"📦 Adding images to '{collection_name}':")
    for name in image_names:
        print(f"   • {name}")

    for idx, uri in tqdm(zip(ids, uris), total=len(uris), desc="🖼️ Ingesting images"):
        collection.add(ids=[idx], uris=[uri])

    print("✅ Image ingestion complete.")
