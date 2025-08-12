import os
import base64
from fastapi import FastAPI, HTTPException, Query
from .retriever import get_text_retriever, get_image_retriever
from models.openrouter import load_openrouter_llm
#from models.groq import load_groq_llm
from prompts.prompt import PROMPT

app = FastAPI()

# — Initialisation des retrievers et du client LLM —
text_ret = get_text_retriever(k=5)
image_ret = get_image_retriever(k=3)
client, MODEL_ID = load_openrouter_llm()
# client, MODEL_ID = load_groq_llm()

SYSTEM_MSG = (
    """
    You are a visual AI assistant. Analyze both the TEXT CONTEXT and IMAGES to answer the question.
    Pay special attention to visual details in images. When asked about something related to images.
    Use ONLY the provided context to answer the question.
    If the answer is not contained in the context, respond with: "I don't know." 
    """
)


def encode_image_to_data_uri(path: str) -> str:
    """Lit un fichier image et renvoie une URI base64."""
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


def retrieve_text_context(query: str):
    """Récupère et formate les textes pertinents."""
    docs = text_ret.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    sources = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    return context, sources


def retrieve_image_uris(query: str, include_images: bool):
    """Récupère les chemins d'images et les encode en data URIs si demandé."""
    if not include_images:
        return [], []
    paths = image_ret(query)
    uris = []
    for p in paths:
        if os.path.exists(p):
            uris.append(encode_image_to_data_uri(p))
        else:
            continue
    return paths, uris


def build_message_payload(context: str, question: str, image_uris: list[str]):
    """Construit la liste des messages avec structure multimodale correcte"""
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
    """Appelle le modèle et renvoie le texte généré."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            max_tokens=1024
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}")


def run_rag(query: str, include_images: bool = True):
    
    # 1) Contexte texte
    context, text_sources = retrieve_text_context(query)
    # 2) Contexte image
    image_paths, image_uris = retrieve_image_uris(query, include_images)
    
    print(f"Retrieved {len(image_uris)} images")  # Debug logging
    
    # 3) Construction des messages
    messages = build_message_payload(context, query, image_uris)
    # 4) Appel du modèle
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
    include_images: bool = True
):
    if not q:
        raise HTTPException(status_code=400, detail="Le paramètre 'query' est requis.")
    try:
        return run_rag(q, include_images)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Multimodal RAG API is running!"}