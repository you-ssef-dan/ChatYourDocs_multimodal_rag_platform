#vectordb.py
print("🔧 Initializing vector database...")
import os
import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata

class VectorDB:
    def __init__(self, persist_dir="database"):
        self.persist_dir = persist_dir
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # embeddings for docs
        self.text_embedding = HuggingFaceEmbeddings(
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

        self.docs_collection_name = "docs_collection"

    def get_docs_collection(self):
        return Chroma(
            collection_name=self.docs_collection_name,
            persist_directory=self.persist_dir,
            embedding_function=self.text_embedding
        )

    def store_documents(self, documents, user_id, chatbot_id, content_type="text"):
        """Store only text-based documents"""
        if not documents:
            return

        chroma = self.get_docs_collection()
        contents = [doc.page_content for doc in documents]

        metadatas = []
        for doc in documents:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            try:
                filtered = filter_complex_metadata(doc)
                metadata = filtered.metadata.copy()
            except Exception:
                metadata = {}
            metadata.update({
                "user_id": str(user_id),
                "chatbot_id": str(chatbot_id),
                "content_type": content_type,
                "source": doc.metadata.get("source", ""),
                "element_type": doc.metadata.get("element_type", "")
            })
            metadatas.append(metadata)

        chroma.add_texts(texts=contents, metadatas=metadatas)

        print(f"📥 Ingested {len(documents)} {content_type} docs into {self.docs_collection_name}")
        return chroma

# Singleton
vector_db = VectorDB()
