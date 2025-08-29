#main.py
import os
import base64
import shutil
from fastapi import FastAPI, HTTPException, Query, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
from .retriever import get_text_retriever, get_image_retriever
#from models.openrouter import load_openrouter_llm
from models.groq import load_groq_llm
from prompts.prompt import PROMPT
from .dispatcher import detect_and_ingest

app = FastAPI()

# Allow CORS from your Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Directory to save uploaded files
UPLOAD_DIR = Path("uploads/users")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global variables for LLM (will be initialized on first request)
client = None
MODEL_ID = None

SYSTEM_MSG = (
    """
    You are a visual AI assistant. Analyze both the TEXT CONTEXT and IMAGES to answer the question.
    Pay special attention to visual details in images. When asked about something related to images.
    Use ONLY the provided context to answer the question.
    If the answer is not contained in the context, respond with: "I don't know." 
    """
)

def initialize_llm():
    """Initialize the LLM client on first use"""
    global client, MODEL_ID
    if client is None:
        #client, MODEL_ID = load_openrouter_llm()
        client, MODEL_ID = load_groq_llm()

def encode_image_to_data_uri(path: str) -> str:
    """Reads an image file and returns a base64 data URI."""
    ext = os.path.splitext(path)[1].lower()
    mime = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
    }.get(ext, 'application/octet-stream')
    with open(path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"

def retrieve_text_context(query: str, user_id: str, chatbot_id: str):
    """Retrieves and formats relevant texts for a specific user and chatbot."""
    text_ret = get_text_retriever(user_id, chatbot_id, k=5)
    docs = text_ret.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    sources = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    return context, sources

def retrieve_image_uris(query: str, include_images: bool, user_id: str, chatbot_id: str):
    """Retrieves image paths and encodes them as data URIs if requested."""
    if not include_images:
        return [], []
    
    image_ret = get_image_retriever(user_id, chatbot_id, k=5)
    image_results = image_ret(query)
    
    paths = []
    uris = []
    
    for uri, metadata in image_results:
        if os.path.exists(uri):
            paths.append(uri)
            uris.append(encode_image_to_data_uri(uri))
        else:
            # Try to find the image in the user's document directory
            user_docs_dir = os.path.join("storage", "users", user_id, chatbot_id, "documents")
            filename = os.path.basename(uri)
            potential_path = os.path.join(user_docs_dir, filename)
            if os.path.exists(potential_path):
                paths.append(potential_path)
                uris.append(encode_image_to_data_uri(potential_path))
    
    return paths, uris

def build_message_payload(context: str, question: str, image_uris: list[str]):
    """Constructs the message list with correct multimodal structure"""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": []}
    ]
    
    # Add image analysis instructions
    if image_uris:
        messages[1]["content"].append({
            "type": "text", 
            "text": "ANALYZE THESE IMAGES CAREFULLY:"
        })
        for uri in image_uris:
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": uri}
            })
    
    # Add text context and question
    prompt_text = PROMPT.format(context=context, question=question)
    messages[1]["content"].append({
        "type": "text",
        "text": prompt_text
    })
    
    return messages

def call_llm(messages: list):
    """Calls the LLM and returns the generated text."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=1024
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")

def run_rag(query: str, user_id: str, chatbot_id: str, include_images: bool = True):
    # Initialize LLM on first use
    initialize_llm()
    
    # 1) Text context
    context, text_sources = retrieve_text_context(query, user_id, chatbot_id)
    # 2) Image context
    image_paths, image_uris = retrieve_image_uris(query, include_images, user_id, chatbot_id)
    
    print(f"Retrieved {len(image_uris)} images for user {user_id}, chatbot {chatbot_id}")
    
    # 3) Build messages
    messages = build_message_payload(context, query, image_uris)
    # 4) Call model
    result = call_llm(messages)
    
    return {
        "result": result,
        "sources": {
            "text": text_sources,
            "images": image_paths
        }
    }

@app.get("/ask")
async def ask(
    q: str = Query(..., alias="query"),
    user_id: str = Query(..., description="User ID"),
    chatbot_id: str = Query(..., description="Chatbot ID"),
    include_images: bool = True
):
    if not q:
        raise HTTPException(status_code=400, detail="The 'query' parameter is required.")
    try:
        return run_rag(q, user_id, chatbot_id, include_images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Multimodal RAG API is running!"}


@app.post("/chatbots")
async def create_chatbot(
    name: str = Form(...),
    user_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Receive chatbot name and multiple files.
    Saves files to the uploads/ folder.
    """
    chatbot_id = "1"
    saved_files = []

    # Construct the folder path: uploads/{user_id}/{chatbot_id}/
    user_folder = UPLOAD_DIR / str(user_id) / chatbot_id / "documents"
    user_folder.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = user_folder / file.filename

        # Save uploaded file inside project folder
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_files.append(file.filename)
    print(f"Saved files for user {user_id}, chatbot '2': {saved_files}")

    detect_and_ingest(user_id, chatbot_id, "uploads")
    return {
        "message": "Chatbot created successfully",
        "user_id": user_id,
        "name": name,
        "files": saved_files
    }