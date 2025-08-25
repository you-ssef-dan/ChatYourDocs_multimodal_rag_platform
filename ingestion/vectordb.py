#vectordb.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

class VectorDB:
    def __init__(self, persist_dir="database"):
        self.embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
    )
    
        self.persist_dir = persist_dir

    def get_collection(self, name):
        return Chroma(
            collection_name=name,
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )

    def store_documents(self, collection_name, documents, metadatas=None):
        """Store documents in the given collection"""
        if not documents:
            return

        chroma = self.get_collection(collection_name)
        contents = [doc.page_content for doc in documents]

        # Handle metadata safely
        if metadatas is None:
            metadatas = []
            for doc in documents:
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                try:
                    filtered = filter_complex_metadata(doc)
                    metadatas.append(filtered.metadata)
                except Exception:
                    metadatas.append({})  # Fallback to empty metadata

        chroma.add_texts(
            texts=contents,
            metadatas=metadatas
        )

        print(f"📥 Ingesting {len(documents)} documents into collection '{collection_name}'...")
        print(f"✅ Successfully stored {len(documents)} documents in '{collection_name}' collection!")

        return chroma

# Singleton instance
vector_db = VectorDB()