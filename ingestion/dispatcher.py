#dispatcher.py
print("🔧 Initializing dispatcher...")
import os
from text_ingestion import ingest_texts
from images_ingestion import ingest_images

DOC_EXT = {".pdf", ".txt", ".docx", ".doc", ".xlsx", ".xls", ".csv",
        ".pptx", ".ppt", ".html", ".htm", ".rtf", ".odt", ".msg"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def detect_and_ingest(folder="documents"):
    docs, imgs = [], []
    
    for fn in sorted(os.listdir(folder)):
        ext = os.path.splitext(fn)[1].lower()
        path = os.path.join(folder, fn)
        if ext in DOC_EXT:
            docs.append(path)
        elif ext in IMG_EXT:
            imgs.append(path)


    if imgs:
        print("🖼️ Images detected → image ingestion")
        ingest_images(collection_name="images", file_paths=imgs)

        
    if docs:
        print("📚 Docs detected → text ingestion")
        ingest_texts(collection_name="documents", file_paths=docs)

    if not docs and not imgs:
        print("⚠️ Nothing to ingest!")

if __name__ == "__main__":
    detect_and_ingest("documents")
