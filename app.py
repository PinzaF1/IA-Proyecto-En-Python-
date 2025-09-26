from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), override=True)  # carga .env

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

app = FastAPI()

class ChatIn(BaseModel):
    pregunta: str
    max_tokens: int = 200

def get_client():
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="HF_API_KEY no está configurada en el entorno (.env o variables de sistema).")
    # usa api_key= (no token=)
    return InferenceClient(api_key=api_key)

@app.post("/chat")
def chat(body: ChatIn):
    client = get_client()
    resp = client.chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": body.pregunta}],
        max_tokens=body.max_tokens
    )
    return {
        "ok": True,
        "model": MODEL,
        "answer": resp.choices[0].message["content"]
    }

# Ruta de diagnóstico opcional
@app.get("/health")
def health():
    api_key = os.getenv("HF_API_KEY")
    return {"token_loaded": bool(api_key), "token_preview": (api_key[:6] + "..." if api_key else None)}

