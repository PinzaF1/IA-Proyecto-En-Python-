from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient
import os, re, json

# ========= Config =========
load_dotenv(find_dotenv(), override=True)
HF_API_KEY = os.getenv("HF_API_KEY")
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

if not HF_API_KEY:
    raise RuntimeError("Falta HF_API_KEY en .env")

client = InferenceClient(api_key=HF_API_KEY)
app = FastAPI(title="ICFES Question Generator")

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
    def valida_opcion(cls, v, values):
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
    temperatura: float = 0.2      # <-- baja creatividad por defecto

class LoteInput(BaseModel):
    items: List[GenInput]
    max_tokens_por_item: int = 400

# ========= Prompts =========
def build_system_prompt():
    return (
        "Eres un generador experto de preguntas tipo ICFES en español. "
        "Debes devolver estrictamente un JSON válido con ESTE esquema exacto: "
        '{"area":"","subtema":"","dificultad":"","estilo_kolb":"","pregunta":"","opciones":{"A":"","B":"","C":"","D":""},"respuesta_correcta":"","explicacion":""}. '
        "Reglas: 1) La pregunta debe tener entre LONG_MIN y LONG_MAX palabras. "
        "2) Cuatro opciones A–D, solo una correcta. "
        "3) La explicación debe justificar por qué la opción correcta es correcta y por qué las demás no. "
        "4) No uses markdown ni comillas tipográficas — solo JSON plano. "
        "5) Toda la salida (pregunta, opciones y explicación) debe estar en ESPAÑOL."
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
        "Responde SOLO con el JSON indicado (sin comentarios, ni markdown)."
    )

# ========= Utilidades JSON =========
def parse_strict_json(text: str) -> dict:
    """
    Intenta extraer y normalizar el primer objeto JSON de un texto potencialmente ruidoso.
    Corrige comillas “tipográficas”, comas colgantes y elimina bloques de formato.
    """
    # 1) Tomar el primer bloque {...}
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("No se detectó ningún bloque JSON en la salida del modelo.")
    s = m.group(0)

    # 2) Normalizar comillas tipográficas → ASCII
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # 3) Quitar etiquetas de código accidentales
    s = s.replace("```json", "").replace("```", "").strip()

    # 4) Eliminar comas colgantes antes de } o ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 5) Parsear
    return json.loads(s)

def ensure_required_keys(data: dict):
    required = ["area", "dificultad", "pregunta", "opciones", "respuesta_correcta"]
    for k in required:
        if k not in data:
            raise ValueError(f"Falta clave requerida: {k}")
    # Opciones deben tener A-D
    if not isinstance(data.get("opciones"), dict):
        raise ValueError("opciones debe ser un objeto con claves A-D")
    for k in ["A", "B", "C", "D"]:
        if k not in data["opciones"]:
            raise ValueError(f"Falta opción {k} en 'opciones'")

# ========= Cliente robusto =========
def chat_completion_json(messages, max_tokens: int, temperature: float):
    """
    Intenta forzar respuesta en JSON si el backend lo soporta.
    Hace fallback si 'response_format' no está disponible.
    """
    # Intento 1: con response_format JSON (si lo soporta el backend)
    try:
        resp = client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"response_format": {"type": "json_object"}},
        )
        return resp
    except Exception:
        # Intento 2: sin extra_body
        resp = client.chat_completion(
            model=MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp

# ========= Generación =========
def generar_una(cfg: GenInput) -> IcfesPreguntaOut:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": build_user_prompt(cfg)},
    ]

    # Primer intento (baja temperatura)
    resp = chat_completion_json(
        messages=messages,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperatura,
    )

    # Algunas versiones de HF devuelven choices como objetos tipo dict/attr
    choice = resp.choices[0]
    content = choice.message["content"] if isinstance(choice.message, dict) else choice.message.content
    text = content

    try:
        data = parse_strict_json(text)
        ensure_required_keys(data)
        return IcfesPreguntaOut(**data)
    except Exception:
        # Reintento: pedir SOLO JSON válido, temperatura mínima
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": (
            "EL JSON ES INVÁLIDO. Reenvía ÚNICAMENTE un JSON válido EXACTO con este esquema: "
            '{"area":"","subtema":"","dificultad":"","estilo_kolb":"","pregunta":"","opciones":{"A":"","B":"","C":"","D":""},"respuesta_correcta":"","explicacion":""}'
        )})

        resp2 = chat_completion_json(
            messages=messages,
            max_tokens=cfg.max_tokens,
            temperature=0.0,
        )
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
        # Descomenta para depurar localmente:
        # print("ERROR>>", repr(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/icfes/lote")
def icfes_lote(body: LoteInput):
    resultados = []
    errores = []
    for i, cfg in enumerate(body.items, start=1):
        try:
            # Asegura max_tokens por item si te interesa limitar:
            cfg.max_tokens = min(cfg.max_tokens, body.max_tokens_por_item)
            q = generar_una(cfg)
            resultados.append(q.dict())
        except Exception as e:
            errores.append({"index": i, "error": str(e)})
    return {
        "ok": len(errores) == 0,
        "generadas": len(resultados),
        "resultados": resultados,
        "errores": errores
    }

@app.get("/health")
def health():
    return {"model": MODEL, "token_loaded": bool(HF_API_KEY)}
