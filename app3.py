from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
import os, re, json

# ========= Config =========
load_dotenv(find_dotenv(), override=True)
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = os.getenv("MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

if not HF_API_KEY:
    raise RuntimeError("Falta HF_API_KEY en variables de entorno (.env o Render).")

client = InferenceClient(api_key=HF_API_KEY)
app = FastAPI(title="ICFES Question Generator (ES)")

# ========= Esquemas =========
class IcfesPreguntaOut(BaseModel):
    area: str
    subtema: Optional[str] = None
    dificultad: str
    estilo_kolb: Optional[str] = None
    pregunta: str = Field(min_length=20, max_length=500)
    opciones: Dict[str, str]            # {"A":"...", "B":"...", "C":"...", "D":"..."}
    respuesta_correcta: str             # "A" | "B" | "C" | "D"
    explicacion: Optional[str] = None

    @validator("respuesta_correcta")
    def valida_opcion(cls, v):
        if v not in ("A", "B", "C", "D"):
            raise ValueError("respuesta_correcta debe ser A/B/C/D")
        return v

class GenInput(BaseModel):
    area: str                     # Matemáticas | Lectura Crítica | Ciencias Naturales | Sociales y Ciudadanas | Inglés
    dificultad: str               # Básica | Intermedia | Avanzada
    subtema: Optional[str] = None
    estilo_kolb: Optional[str] = None  # Divergente | Asimilador | Acomodador | Convergente
    longitud_min: int = 30
    longitud_max: int = 60
    max_tokens: int = 400
    temperatura: float = 0.2      # baja creatividad

class LoteInput(BaseModel):
    items: List[GenInput]
    max_tokens_por_item: int = 400

# ========= Prompts =========
def build_system_prompt():
    return (
        "Eres un generador experto de preguntas tipo ICFES en ESPAÑOL. "
        "Devuelve EXCLUSIVAMENTE un objeto JSON plano (sin bloques ```), "
        "con comillas dobles ASCII y SIN comas finales en objetos o listas. "
        "Esquema EXACTO: "
        '{"area":"","subtema":"","dificultad":"","estilo_kolb":"","pregunta":"","opciones":{"A":"","B":"","C":"","D":""},"respuesta_correcta":"","explicacion":""}. '
        "Reglas: 1) La pregunta debe tener entre LONG_MIN y LONG_MAX palabras. "
        "2) Cuatro opciones A–D, solo una correcta. "
        "3) La explicación debe justificar la opción correcta y descartar las demás. "
        "4) No uses markdown ni texto extra fuera del JSON."
    )

def build_user_prompt(cfg: GenInput) -> str:
    return (
        f"Genera UNA pregunta ICFES.\n"
        f"area: {cfg.area}\n"
        f"subtema: {cfg.subtema or ''}\n"
        f"dificultad: {cfg.dificultad}\n"
        f"estilo_kolb: {cfg.estilo_kolb or ''}\n"
        f"LONG_MIN: {cfg.longitud_min}\n"
        f"LONG_MAX: {cfg.longitud_max}\n"
        "Responde SOLO con el JSON indicado (sin comentarios ni markdown) y en ESPAÑOL."
    )

# ========= Utilidades JSON =========
def parse_strict_json(text: str) -> dict:
    """
    Extrae el primer objeto {...} y normaliza comillas/formatos.
    """
    # 1) Extraer primer bloque {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No se detectó ningún bloque JSON en la salida del modelo.")
    s = m.group(0)

    # 2) Normalizar comillas tipográficas → ASCII
    s = (s.replace("“", '"').replace("”", '"')
           .replace("’", "'").replace("‘", "'"))

    # 3) Quitar etiquetas/rodeos y espacios
    s = s.replace("```json", "").replace("```", "").strip()

    # 4) Eliminar comas colgantes antes de } o ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 5) Forzar claves A-D en mayúsculas si vienen en minúscula (raro)
    s = re.sub(
        r'"opciones"\s*:\s*\{([^}]*)\}',
        lambda m2: re.sub(r'"([a-d])"\s*:', lambda k: f'"{k.group(1).upper()}":', m2.group(0)),
        s, flags=re.S
    )

    # 6) Parsear JSON
    return json.loads(s)

def ensure_required_keys(data: dict):
    required = ["area", "dificultad", "pregunta", "opciones", "respuesta_correcta"]
    for k in required:
        if k not in data:
            raise ValueError(f"Falta clave requerida: {k}")
    if not isinstance(data.get("opciones"), dict):
        raise ValueError("opciones debe ser un objeto con claves A-D")
    for k in ["A", "B", "C", "D"]:
        if k not in data["opciones"]:
            raise ValueError(f"Falta opción {k} en 'opciones'")

# ========= Cliente robusto =========
def chat_completion_json(messages, max_tokens: int, temperature: float):
    # Intento 1: pedir JSON explícitamente
    try:
        return client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
            extra_body={"response_format": {"type": "json_object"}},
        )
    except Exception:
        # Intento 2: sin response_format
        return client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1,
        )

# ========= Generación =========
def generar_una(cfg: GenInput) -> IcfesPreguntaOut:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(cfg)},
    ]

    # Primer intento
    resp = chat_completion_json(messages, max_tokens=cfg.max_tokens, temperature=cfg.temperatura)
    choice = resp.choices[0]
    content = choice.message["content"] if isinstance(choice.message, dict) else choice.message.content
    text = content

    try:
        data = parse_strict_json(text)
        ensure_required_keys(data)
        return IcfesPreguntaOut(**data)
    except Exception:
        # Reintento con temperatura mínima y recordatorio de JSON estricto
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": (
            "EL JSON ES INVÁLIDO. Reenvía ÚNICAMENTE un JSON válido EXACTO con este esquema: "
            '{"area":"","subtema":"","dificultad":"","estilo_kolb":"","pregunta":"","opciones":{"A":"","B":"","C":"","D":""},"respuesta_correcta":"","explicacion":""} '
            "Sin ningún texto adicional, ni comentarios, ni markdown."
        )})
        resp2 = chat_completion_json(messages, max_tokens=cfg.max_tokens, temperature=0.0)
        choice2 = resp2.choices[0]
        content2 = choice2.message["content"] if isinstance(choice2.message, dict) else choice2.message.content
        text2 = content2
        data2 = parse_strict_json(text2)
        ensure_required_keys(data2)
        return IcfesPreguntaOut(**data2)

# ========= Endpoints =========
@app.post("/icfes/generar", response_model=IcfesPreguntaOut)
def icfes_generar(body: GenInput):
    try:
        return generar_una(body)
    except Exception as e:
        # Devuelve 400 (no 500) cuando el modelo no produce JSON válido
        raise HTTPException(status_code=400, detail=f"Salida del modelo no es JSON válido: {e}")

@app.post("/icfes/lote")
def icfes_lote(body: LoteInput):
    resultados, errores = [], []
    for i, cfg in enumerate(body.items, start=1):
        try:
            cfg.max_tokens = min(cfg.max_tokens, body.max_tokens_por_item)
            q = generar_una(cfg)
            resultados.append(q.dict())
        except Exception as e:
            errores.append({"index": i, "error": str(e)})
    return {"ok": len(errores) == 0, "generadas": len(resultados), "resultados": resultados, "errores": errores}

@app.get("/health")
def health():
    return {"model": MODEL, "token_loaded": bool(HF_API_KEY)}
