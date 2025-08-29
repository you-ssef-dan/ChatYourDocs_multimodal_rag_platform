#dispatcher.py

print("🔧 Initializing dispatcher...")
import os
from .text_ingestion import ingest_texts
from .images_ingestion import ingest_images

DOC_EXT = {".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls", ".csv",
        ".pptx", ".ppt", ".html", ".htm", ".rtf", ".odt", ".msg"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def detect_and_ingest(user_id, chatbot_id, base_storage_path="uploads"):
    """Ingest documents for a specific user and chatbot from the storage structure"""
    # Build the specific path for this user and chatbot
    user_chatbot_path = os.path.join(base_storage_path, "users", str(user_id), str(chatbot_id), "documents")
    
    # Check if the directory exists
    if not os.path.exists(user_chatbot_path):
        print(f"⚠️ Directory not found: {user_chatbot_path}")
        return
    
    docs, imgs = [], []
    
    try:
        for fn in sorted(os.listdir(user_chatbot_path)):
            ext = os.path.splitext(fn)[1].lower()
            path = os.path.join(user_chatbot_path, fn)
            if os.path.isfile(path):  # Only process files, not directories
                if ext in DOC_EXT:
                    docs.append(path)
                elif ext in IMG_EXT:
                    imgs.append(path)
    except FileNotFoundError:
        print(f"❌ Documents directory not found for user {user_id}, chatbot {chatbot_id}")
        return
    except PermissionError:
        print(f"❌ Permission denied accessing documents for user {user_id}, chatbot {chatbot_id}")
        return

    print(f"📊 Found {len(docs)} documents and {len(imgs)} images for user {user_id}, chatbot {chatbot_id}")

    if imgs:
        print(f"🖼️ Images detected → image ingestion for user {user_id}, chatbot {chatbot_id}")
        ingest_images(user_id=user_id, chatbot_id=chatbot_id, file_paths=imgs)

    if docs:
        print(f"📚 Docs detected → text ingestion for user {user_id}, chatbot {chatbot_id}")
        ingest_texts(user_id=user_id, chatbot_id=chatbot_id, file_paths=docs)

    if not docs and not imgs:
        print(f"⚠️ No documents or images found in {user_chatbot_path}")

if __name__ == "__main__":
    # Example usage options:
    
    # Option 1: Process specific user and chatbot
    user_id = "1"  # Replace with actual user ID
    chatbot_id = "1"  # Replace with actual chatbot ID
    detect_and_ingest(user_id, chatbot_id, "uploads")