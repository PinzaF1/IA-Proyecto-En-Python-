# =============================================================================
# EduExcel_app.py  (SIEMPRE devuelve 5 ítems)
# -----------------------------------------------------------------------------
# - Lista blanca de subtemas (rechaza lo que no esté permitido)
# - Fuerza salida JSON + validación
# - Enunciados largos (100–140 palabras por defecto)
# - Mezcla opciones A–D y genera explicación por opción
# - Verificación/autocorrección para Matemáticas → sistemas 2×2
# - Backends: Hugging Face / Ollama / Groq
# - PACK: genera N ítems en una sola respuesta; si faltan, rellena con
#   llamadas unitarias y, si aun así falla el LLM, usa fallback local
#   (plantillas con variación) hasta completar exactamente 5.
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
import os, json, re, unicodedata, random, requests

from huggingface_hub import InferenceClient

# ============================= Configuración ================================
load_dotenv(find_dotenv(), override=True)

BACKEND        = os.getenv("BACKEND", "ollama")  # 'hf' | 'ollama' | 'groq'
# HF
HF_API_KEY     = os.getenv("HF_API_KEY", "")
HF_MODEL       = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_TIMEOUT     = int(os.getenv("HF_TIMEOUT", "60"))
# Ollama
OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))
# Groq
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

DEBUG          = os.getenv("DEBUG_JSON", "0") == "1"

if BACKEND == "hf" and not HF_API_KEY:
    raise RuntimeError("BACKEND=hf pero falta HF_API_KEY en .env")
if BACKEND == "groq" and not GROQ_API_KEY:
    raise RuntimeError("BACKEND=groq pero falta GROQ_API_KEY en .env")

hf_client = InferenceClient(api_key=HF_API_KEY, timeout=HF_TIMEOUT) if BACKEND == "hf" else None
app = FastAPI(title=f"EduExcel (backend={BACKEND})")

def _dbg(x: str):
    if DEBUG: print(str(x)[:2400])


# =============================== Lista blanca ===============================
ALLOWED_SUBTEMAS: Dict[str, List[str]] = {
    "Lenguaje": [
        "Comprensión lectora (sentido global y local)",
        "Conectores lógicos (causa, contraste, condición, secuencia)",
        "Identificación de argumentos y contraargumentos",
        "Idea principal y propósito comunicativo",
        "Hecho vs. opinión e inferencias"
    ],
    "Matemáticas": [
        "Operaciones con números enteros",
        "Razones y proporciones",
        "Regla de tres simple y compuesta",
        "Porcentajes y tasas (aumento, descuento, interés simple)",
        "Ecuaciones lineales y sistemas 2×2"
    ],
    "Sociales y Ciudadanas": [
        "Constitución de 1991 y organización del Estado",
        "Historia de Colombia (Frente Nacional, conflicto y paz)",
        "Guerras Mundiales y Guerra Fría",
        "Geografía de Colombia (mapas, territorio y ambiente)",
        "Economía y ciudadanía económica (globalización y desigualdad)"
    ],
    "Ciencias Naturales": [
        "Indagación científica (variables, control e interpretación de datos)",
        "Fuerzas, movimiento y energía",
        "Materia y cambios (mezclas, reacciones y conservación)",
        "Genética y herencia",
        "Ecosistemas y cambio climático (CTS)"
    ],
    "Inglés": [
        "Verb to be (am, is, are)",
        "Present Simple (afirmación, negación y preguntas)",
        "Past Simple (verbos regulares e irregulares)",
        "Comparatives and superlatives",
        "Subject/Object pronouns y possessive adjectives"
    ],
}


# ================================ Modelos ===================================
class GenInput(BaseModel):
    area: str
    dificultad: str
    subtema: str
    estilo_kolb: Optional[str] = None
    cantidad: int = 5
    longitud_min: int = 100
    longitud_max: int = 140
    max_tokens_item: int = 300
    temperatura: float = 0.3

class IcfesPreguntaOut(BaseModel):
    area: str
    subtema: str
    dificultad: str
    estilo_kolb: Optional[str] = None
    pregunta: str = Field(min_length=10, max_length=800)
    opciones: Dict[str, str]
    respuesta_correcta: str
    explicacion: Optional[str] = ""
    meta: Optional[Dict[str, object]] = {}

    @field_validator("respuesta_correcta")
    @classmethod
    def _opt(cls, v):
        if v not in ("A", "B", "C", "D"):
            raise ValueError("respuesta_correcta debe ser A/B/C/D")
        return v


# ============================ Utilidades base ===============================
def _norm(s: str) -> str:
    s = (s or "").lower().replace("×", "x")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def is_sys2x2(text: str) -> bool:
    t = _norm(text)
    return ("sistema" in t) and ("2x2" in t)

def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or "", flags=re.UNICODE))

def pad_to_range(texto: str, min_palabras: int, max_palabras: int, extras: List[str]) -> str:
    """Ajusta el enunciado al rango [min,max] añadiendo frases-guía o recortando por oraciones."""
    texto = (texto or "").strip()
    w = _word_count(texto)
    extras_idx = 0
    while w < min_palabras and extras_idx < len(extras):
        add = extras[extras_idx].strip()
        if add and not texto.endswith((".", "?", "¡", "!", "…")):
            texto += ". "
        texto += add
        extras_idx += 1
        w = _word_count(texto)
    if w > max_palabras:
        sents = re.split(r"(?<=[.!?])\s+", texto)
        nuevo, cnt = [], 0
        for s in sents:
            sw = _word_count(s)
            if cnt + sw <= max_palabras:
                nuevo.append(s); cnt += sw
            else:
                break
        texto = " ".join(nuevo).strip()
    return texto

def shuffle_options_dict(opciones: dict, correcta_label: str) -> tuple[dict, str]:
    labels = ["A", "B", "C", "D"]
    pairs = [(k, opciones.get(k, "").strip()) for k in labels]
    if len(set(v for _, v in pairs)) < 4:
        return opciones, correcta_label
    random.shuffle(pairs)
    new_op, new_label = {}, None
    for i, (old_label, text) in enumerate(pairs):
        lab = labels[i]; new_op[lab] = text
        if old_label == correcta_label: new_label = lab
    return new_op, (new_label or correcta_label)


# ======================= Matemáticas: utilidades 2×2 =======================
_num = r"[+-]?(?:\d+(?:[.,]\d+)?|\d*[.,]\d+)"
pat_par1 = re.compile(r"\(\s*(%s)\s*,\s*(%s)\s*\)" % (_num, _num))
pat_par2 = re.compile(r"x\s*=\s*(%s)\s*,\s*y\s*=\s*(%s)" % (_num, _num), re.I)

def _to_float_es(s: str) -> float:
    return float(s.replace(" ", "").replace(",", "."))

def parse_pair(s: str):
    m = pat_par1.search(s) or pat_par2.search(s)
    if not m: return None
    try: return (_to_float_es(m.group(1)), _to_float_es(m.group(2)))
    except: return None

def safe_fmt_decimal(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v))).replace(".", ",")
    return f"{v:.2f}".replace(".", ",")

def fmt_pair_es2(p: Tuple[float, float]) -> str:
    return f"({safe_fmt_decimal(p[0])}, {safe_fmt_decimal(p[1])})"

def solve_2x2(a1,b1,c1,a2,b2,c2):
    det = a1*b2 - a2*b1
    if abs(det) < 1e-12: return None
    x = (c1*b2 - c2*b1)/det
    y = (a1*c2 - a2*c1)/det
    return (x,y)

def make_distractors(sol: Tuple[float,float]) -> List[Tuple[float,float]]:
    x, y = sol
    cands = [(y, x), (-x, y), (x, -y), (x+1, y-1), (x-1, y+1), (round(x), round(y))]
    out = []
    seen = { (round(sol[0], 4), round(sol[1], 4)) }
    for p in cands:
        key = (round(p[0], 4), round(p[1], 4))
        if key not in seen:
            seen.add(key); out.append(p)
        if len(out) == 3: break
    while len(out) < 3:
        out.append( (x + (len(out)+1)*0.5, y - (len(out)+1)*0.5) )
    return out


# =================== Explicaciones por opción (por área) ====================
def build_explanation_per_option(cfg, pregunta: str, opciones: dict, correcta: str, meta: dict|None=None) -> str:
    textos = []
    def por_opcion(rA, rB, rC, rD):
        textos.append(f"Correcta ({correcta}).")
        textos.append(f"A: {rA}")
        textos.append(f"B: {rB}")
        textos.append(f"C: {rC}")
        textos.append(f"D: {rD}")

    # Matemáticas 2×2: usa meta
    if "matem" in _norm(cfg.area) and "2x2" in _norm(cfg.subtema) and isinstance(meta, dict):
        try:
            a1=meta["a1"]; b1=meta["b1"]; c1=meta["c1"]; a2=meta["a2"]; b2=meta["b2"]; c2=meta["c2"]
            sol_txt = opciones[correcta]
            textos.append(f"Se resuelve el sistema por eliminación o Cramer con coeficientes a1={a1}, b1={b1}, c1={c1}, a2={a2}, b2={b2}, c2={c2}.")
            def razon(op_txt):
                return "Satisface simultáneamente ambas ecuaciones del sistema." if op_txt == sol_txt \
                       else "No satisface al menos una ecuación; típico de errores de signo o combinación de filas."
            por_opcion(razon(opciones["A"]), razon(opciones["B"]), razon(opciones["C"]), razon(opciones["D"]))
            return " ".join(textos)
        except Exception:
            pass

    # Lenguaje
    if "lenguaj" in _norm(cfg.area):
        por_opcion(
            "Resume la idea global sustentada en el texto.",
            "Confunde detalle local con la idea principal.",
            "Generaliza más allá de lo dicho.",
            "Atribuye intención no respaldada."
        ); return " ".join(textos)

    # Sociales
    if "sociales" in _norm(cfg.area):
        por_opcion(
            "Sintetiza finalidad y alcance del proceso/institución.",
            "Confunde propósito con procedimiento.",
            "Reduce a caso puntual y pierde generalidad.",
            "Contradice evidencia histórica."
        ); return " ".join(textos)

    # Ciencias Naturales
    if "ciencias naturales" in _norm(cfg.area):
        por_opcion(
            "Identifica correctamente VI/VD y controles según el diseño.",
            "Intercambia VI y VD.",
            "Toma un control como variable principal.",
            "Plantea relación no sustentada por el método."
        ); return " ".join(textos)

    # Inglés
    if "ingles" in _norm(cfg.area):
        por_opcion(
            "Respeta la regla objetivo (forma/concordancia).",
            "Error de concordancia sujeto–verbo.",
            "Tiempo verbal incorrecto.",
            "Pronombre/posesivo mal seleccionado."
        ); return " ".join(textos)

    # Fallback general
    por_opcion(
        "Coherente con el enunciado y la relación pedida.",
        "Confunde la relación central.",
        "Dato parcial sin integrar la idea principal.",
        "Conclusión injustificada."
    )
    return " ".join(textos)


# ==================== Post-proceso (largo + mezcla + explicación) ==========
def postprocess_item(cfg: GenInput, data: dict) -> dict:
    extras = [
        "Lee con atención cada indicio antes de decidir",
        "Evita confundir ejemplos particulares con definiciones generales",
        "Contrasta el propósito con los procedimientos que lo implementan",
        "Selecciona la alternativa que mejor sintetiza la idea central",
        "Verifica la coherencia de tu opción con todas las pistas del enunciado"
    ]
    data["pregunta"] = pad_to_range(
        data.get("pregunta",""), cfg.longitud_min, cfg.longitud_max, extras
    )
    # Mezcla A–D
    op = data.get("opciones", {}); resp = data.get("respuesta_correcta", "A")
    if isinstance(op, dict) and resp in ("A","B","C","D"):
        op2, r2 = shuffle_options_dict(op, resp)
        data["opciones"] = op2; data["respuesta_correcta"] = r2
    # Explicación por opción
    try:
        data["explicacion"] = build_explanation_per_option(
            cfg, data.get("pregunta",""), data.get("opciones",{}),
            data.get("respuesta_correcta","A"), meta=data.get("meta",{})
        )
    except Exception:
        data["explicacion"] = data.get("explicacion") or \
            "La opción correcta es la única coherente con el enunciado."
    return data


# ============================== Prompts LLM =================================
def _join(area: str) -> str:
    return "; ".join(ALLOWED_SUBTEMAS[area])

def build_system() -> str:
    return (
        "Eres un generador experto de ÍTEMS tipo ICFES en ESPAÑOL. "
        "Respondes SIEMPRE con JSON válido y SOLO con este esquema por ítem:"
        "{\"area\":\"\",\"subtema\":\"\",\"dificultad\":\"\",\"estilo_kolb\":\"\",\"pregunta\":\"\","
        "\"opciones\":{\"A\":\"\",\"B\":\"\",\"C\":\"\",\"D\":\"\"},\"respuesta_correcta\":\"\",\"explicacion\":\"\",\"meta\":{}} "
        "Si se piden varias preguntas, devuelve un único objeto JSON: {\"items\":[OBJ1,OBJ2,...,OBJN]}. "
        "Reglas: Español (excepto área Inglés); JSON estricto (sin markdown ni texto extra); "
        "pregunta entre LONG_MIN y LONG_MAX palabras; 4 opciones A–D y una sola correcta; "
        "explicación breve; ítems distintos; NO cambies 'area' ni 'subtema'; incluye 'meta' útil. "
        f"Lenguaje: {_join('Lenguaje')}. Matemáticas: {_join('Matemáticas')}. "
        f"Sociales y Ciudadanas: {_join('Sociales y Ciudadanas')}. "
        f"Ciencias Naturales: {_join('Ciencias Naturales')}. Inglés: {_join('Inglés')}."
    )

def build_user(cfg: GenInput) -> str:
    extra = ""
    if is_sys2x2(cfg.subtema):
        extra = (
            "Para 2x2 escribe:\nax + by = c\ndx + ey = f\n"
            "meta:{\"tipo\":\"sistema2x2\",\"a1\":INT,\"b1\":INT,\"c1\":INT,\"a2\":INT,\"b2\":INT,\"c2\":INT}; "
            "coeficientes |1..6|, solución única; las cuatro opciones son pares (x, y)."
        )
    return (
        f"Genera UNA pregunta del área {cfg.area}, subtema EXACTO {cfg.subtema}."
        f" Dificultad {cfg.dificultad}, estilo Kolb {cfg.estilo_kolb or 'Convergente'}."
        f" LONG_MIN {cfg.longitud_min}, LONG_MAX {cfg.longitud_max}."
        " Devuelve SOLO el JSON del esquema indicado. " + extra
    )

def build_pack_user(cfg: GenInput, cantidad: int) -> str:
    nota_2x2 = (
        "Cada objeto incluye sistema 2x2 y meta con coeficientes; opciones como (x, y)."
        if is_sys2x2(cfg.subtema) else ""
    )
    return (
        "Devuelve SOLO: {\"items\":[OBJ1,...,OBJN]} con EXACTAMENTE N ítems del esquema indicado."
        f" N = {cantidad}. area={cfg.area}; subtema={cfg.subtema}; dificultad={cfg.dificultad}; "
        f"estilo_kolb={cfg.estilo_kolb or 'Convergente'}; LONG_MIN={cfg.longitud_min}; LONG_MAX={cfg.longitud_max}. "
        "Sin texto fuera del JSON. Ítems no repetidos. " + nota_2x2
    )


# ============================ Llamadas al backend ===========================
def chat_backend(messages, max_tokens: int, temperature: float, force_json: bool) -> str:
    if BACKEND == "hf":
        try:
            kwargs = dict(model=HF_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
            if force_json:
                kwargs["extra_body"] = {"response_format": {"type": "json_object"}}
            resp = hf_client.chat_completion(**kwargs)
            msg = resp.choices[0].message
            return msg["content"] if isinstance(msg, dict) else msg.content
        except Exception as e:
            raise RuntimeError(f"HF error: {e}")

    if BACKEND == "ollama":
        url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "keep_alive": "8m",
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 4096,
                "repeat_penalty": 1.05
            }
        }
        if force_json:
            payload["format"] = "json"
        r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        return (data.get("message") or {}).get("content", "")

    if BACKEND == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        body = {"model": GROQ_MODEL, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if force_json:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(url, headers=headers, json=body, timeout=HF_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"Groq HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"BACKEND no soportado: {BACKEND}")

def chat_once(messages, max_tokens: int, temperature: float, force_json: bool) -> str:
    text = chat_backend(messages, max_tokens=max_tokens, temperature=temperature, force_json=force_json)
    _dbg(("RAW(JSON)>> " if force_json else "RAW(PLAIN)>> ") + (text or "")[:1200])
    return text or ""


# ===================== Parsing + validaciones de esquema ====================
def parse_json_min(text: str) -> dict:
    if not text: raise ValueError("Salida vacía del modelo.")
    s = text.replace("```json","").replace("```","")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: raise ValueError("No se detectó JSON en la salida.")
    blob = m.group(0)
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    return json.loads(blob)

def ensure_min_schema(d: dict):
    if "pregunta" not in d: raise ValueError("Falta 'pregunta'")
    if "opciones" not in d or not isinstance(d["opciones"], dict): raise ValueError("Falta 'opciones'")
    for k in ["A","B","C","D"]:
        if k not in d["opciones"]: raise ValueError(f"Falta opción {k}")
    if "respuesta_correcta" not in d: raise ValueError("Falta 'respuesta_correcta'")
    if d["respuesta_correcta"] not in ("A","B","C","D"): raise ValueError("respuesta_correcta inválida")
    if "explicacion" not in d: d["explicacion"] = ""
    if "meta" not in d: d["meta"] = {}

def enforce_whitelist(cfg: GenInput, data: dict):
    if cfg.area not in ALLOWED_SUBTEMAS: raise ValueError(f"Área no permitida: {cfg.area}")
    if cfg.subtema not in ALLOWED_SUBTEMAS[cfg.area]: raise ValueError(f"Subtema no permitido: {cfg.subtema}")
    if data.get("area") and _norm(data["area"]) != _norm(cfg.area): raise ValueError("El modelo cambió el 'area'.")
    if data.get("subtema") and _norm(data["subtema"]) != _norm(cfg.subtema): raise ValueError("El modelo cambió el 'subtema'.")


# ======================= Verificación matemática 2×2 ========================
def ensure_math_consistency(cfg: GenInput, data: dict) -> dict:
    if "matem" not in _norm(cfg.area): return data
    if not is_sys2x2(cfg.subtema): return data
    meta = data.get("meta") or {}
    need = ("a1","b1","c1","a2","b2","c2")
    if not all(k in meta for k in need):
        raise ValueError("La pregunta no incluye 'meta' con coeficientes del sistema 2x2.")
    a1=meta["a1"]; b1=meta["b1"]; c1=meta["c1"]; a2=meta["a2"]; b2=meta["b2"]; c2=meta["c2"]
    if not all(isinstance(v,(int,float)) and v!=0 for v in [a1,b1,a2,b2]):
        raise ValueError("Coeficientes inválidos (a1,b1,a2,b2 deben ser ≠ 0).")
    sol = solve_2x2(a1,b1,c1,a2,b2,c2)
    if sol is None: raise ValueError("El sistema no tiene solución única (det=0).")
    distract = make_distractors(sol)
    data["opciones"] = {"A": fmt_pair_es2(sol), "B": fmt_pair_es2(distract[0]),
                        "C": fmt_pair_es2(distract[1]), "D": fmt_pair_es2(distract[2])}
    data["respuesta_correcta"] = "A"
    data["explicacion"] = (
        f"La solución del sistema es {fmt_pair_es2(sol)}. "
        "Se obtiene por eliminación o por la regla de Cramer."
    )
    return data


# ============================== Fallback local ==============================
# Generadores paramétricos por área/subtema para garantizar SIEMPRE 5 ítems.

def rb_lang(cfg: GenInput) -> dict:
    sub = _norm(cfg.subtema)
    base = (
        "Un breve dossier expone una situación real con datos verificables y opiniones contrapuestas. "
        "El texto presenta argumentos, ejemplos y un cierre que invita a tomar postura, "
        "señalando diferencias entre afirmaciones comprobables y juicios de valor."
    )
    opciones = {
        "comprension lectora (sentido global y local)": [
            "Identifica y sintetiza la idea principal sin perder los detalles esenciales.",
            "Confunde una anécdota puntual con el eje del texto.",
            "Introduce supuestos no mencionados en el dossier.",
            "Reduce el alcance general a un caso aislado."
        ],
        "conectores logicos (causa, contraste, condicion, secuencia)": [
            "Porque la evidencia sugiere una relación causal directa en el cierre.",
            "Sin embargo, ya que el texto no propone oposición real.",
            "Si no, aunque el texto no formula condición alguna.",
            "Luego entonces, pese a que no hay secuencia temporal explícita."
        ],
        "identificacion de argumentos y contraargumentos": [
            "Señala la tesis y las razones que la sustentan.",
            "Enumera datos sin relación con la tesis.",
            "Plantea un contraargumento como si fuera la tesis.",
            "Apela a la autoridad sin evidencia."
        ],
        "idea principal y proposito comunicativo": [
            "Integra la intención global y orienta la lectura del conjunto.",
            "Solo reitera un ejemplo menor.",
            "Parafrasea una frase llamativa pero marginal.",
            "Atribuye un propósito no defendido por el texto."
        ],
        "hecho vs. opinion e inferencias": [
            "Distingue hechos contrastables de valoraciones e infiere con base en evidencia.",
            "Confunde opinión con dato verificable.",
            "Infieren conclusiones no sustentadas.",
            "Omiten la diferencia entre evidencia y juicio."
        ],
    }
    key = None
    for k in opciones:
        if k in sub:
            key = k; break
    if key is None: key = list(opciones.keys())[0]
    opts = opciones[key]
    return {
        "area": cfg.area,
        "subtema": cfg.subtema,
        "dificultad": cfg.dificultad,
        "estilo_kolb": cfg.estilo_kolb or "Asimilador",
        "pregunta": base + " ¿Cuál alternativa refleja mejor el foco de lectura pedido por el subtema?",
        "opciones": {"A": opts[0], "B": opts[1], "C": opts[2], "D": opts[3]},
        "respuesta_correcta": "A",
        "explicacion": "",
        "meta": {"source":"rule-fallback","texto_base": base}
    }

def rb_mate_2x2(cfg: GenInput) -> dict:
    # Genera sistema con solución única
    a1,b1,a2,b2 = [random.choice([1,2,3,4,5,6]) for _ in range(4)]
    while a1*b2 - a2*b1 == 0:
        a2,b2 = random.choice([1,2,3,4,5,6]), random.choice([1,2,3,4,5,6])
    x = random.choice([-3,-2,-1,1,2,3])
    y = random.choice([-3,-2,-1,1,2,3])
    c1 = a1*x + b1*y
    c2 = a2*x + b2*y
    meta = {"tipo":"sistema2x2","a1":a1,"b1":b1,"c1":c1,"a2":a2,"b2":b2,"c2":c2}
    data = {
        "area": cfg.area,
        "subtema": cfg.subtema,
        "dificultad": cfg.dificultad,
        "estilo_kolb": cfg.estilo_kolb or "Convergente",
        "pregunta": f"Resuelve el sistema de ecuaciones: {a1}x + {b1}y = {c1}; {a2}x + {b2}y = {c2}.",
        "opciones": {"A":"", "B":"", "C":"", "D":""},
        "respuesta_correcta": "A",
        "explicacion": "",
        "meta": meta | {"source":"rule-fallback"}
    }
    data = ensure_math_consistency(cfg, data)  # arma opciones correctas y explicación
    return data

def rb_mate_simple(cfg: GenInput) -> dict:
    sub = _norm(cfg.subtema)
    if "operaciones con numeros enteros" in sub:
        a,b,c = random.randint(-20,20), random.randint(-20,20), random.randint(-10,10)
        exp = f"({a}) + ({b}) - ({c})"
        val = a + b - c
        wrong = [val+1, val-1, -val]
        random.shuffle(wrong)
        return {
            "area": cfg.area, "subtema": cfg.subtema, "dificultad": cfg.dificultad,
            "estilo_kolb": cfg.estilo_kolb or "Convergente",
            "pregunta": f"En un cálculo con números enteros, evalúa la expresión {exp} en el contexto de un balance de temperaturas diarias.",
            "opciones": {"A": str(val), "B": str(wrong[0]), "C": str(wrong[1]), "D": str(wrong[2])},
            "respuesta_correcta": "A",
            "explicacion": "Se suman y restan enteros respetando signos; la opción A coincide con el resultado.",
            "meta": {"source":"rule-fallback","expresion":exp,"resultado":val}
        }
    if "razones y proporciones" in sub:
        a,b = random.randint(2,5), random.randint(3,6)
        x = random.randint(10,20); y = int(x*b/a)
        wrong = [y+1, y-1, y+2]
        random.shuffle(wrong)
        return {
            "area": cfg.area, "subtema": cfg.subtema, "dificultad": cfg.dificultad,
            "estilo_kolb": cfg.estilo_kolb or "Convergente",
            "pregunta": f"Una mezcla mantiene la razón {a}:{b}. Si {a} partes equivalen a {x} litros, ¿cuánto corresponde a {b} partes en la misma proporción?",
            "opciones": {"A": str(y), "B": str(wrong[0]), "C": str(wrong[1]), "D": str(wrong[2])},
            "respuesta_correcta": "A",
            "explicacion": "Se aplica proporcionalidad directa manteniendo la razón.",
            "meta": {"source":"rule-fallback","razon":[a,b],"x":x,"resultado":y}
        }
    if "regla de tres" in sub:
        a = random.randint(4,8); b = random.randint(12,20)
        x = random.randint(2,4); y = int(b*x/a)
        wrong = [y+2, y-1, y+1]; random.shuffle(wrong)
        return {
            "area": cfg.area, "subtema": cfg.subtema, "dificultad": cfg.dificultad,
            "estilo_kolb": cfg.estilo_kolb or "Convergente",
            "pregunta": f"Si {a} operarios completan una tarea en {b} horas, ¿en cuántas horas la completan {x} operarios al mismo ritmo (regla de tres simple inversa)?",
            "opciones": {"A": str(y), "B": str(wrong[0]), "C": str(wrong[1]), "D": str(wrong[2])},
            "respuesta_correcta": "A",
            "explicacion": "A mayor número de operarios, menor tiempo: proporcionalidad inversa.",
            "meta": {"source":"rule-fallback","operarios":[a,x],"horas":b,"resultado":y}
        }
    if "porcentajes y tasas" in sub:
        p = random.choice([10,12,15,20])
        precio = random.choice([80,100,120,150])
        y = round(precio*(1-p/100),2)
        wrong = [round(y*1.05,2), round(y*0.95,2), precio]; random.shuffle(wrong)
        return {
            "area": cfg.area, "subtema": cfg.subtema, "dificultad": cfg.dificultad,
            "estilo_kolb": cfg.estilo_kolb or "Convergente",
            "pregunta": f"Una tienda aplica un descuento del {p}% sobre un producto de ${precio}. ¿Cuál es el precio final pagado?",
            "opciones": {"A": str(y).replace(".",","), "B": str(wrong[0]).replace(".",","), "C": str(wrong[1]).replace(".",","), "D": str(wrong[2]).replace(".",",")},
            "respuesta_correcta": "A",
            "explicacion": "Se descuenta el porcentaje al precio base.",
            "meta": {"source":"rule-fallback","precio":precio,"descuento_porcentaje":p,"resultado":y}
        }
    # por defecto: 2x2
    return rb_mate_2x2(cfg)

def rb_sociales(cfg: GenInput) -> dict:
    sub = _norm(cfg.subtema)
    enun = (
        "Se revisa un conjunto de fuentes que explican propósitos, alcances y límites de instituciones y procesos históricos, "
        "incluida la interacción entre normas, actores y contextos en diversas escalas."
    )
    if "constitucion de 1991" in sub:
        op = [
            "Define la estructura del Estado y consagra derechos y deberes para orientar la vida social.",
            "Reúne trámites administrativos temporales sin fuerza normativa.",
            "Se limita a regular elecciones cada cuatro años.",
            "Solo fija protocolos de seguridad de edificios."
        ]
    elif "historia de colombia" in sub:
        op = [
            "Busca reducir tensiones y ampliar representación, con desafíos en implementación.",
            "Pretende suprimir la oposición política.",
            "Se limita a sanciones penales sin reformas.",
            "Mantiene el statu quo sin cambios."
        ]
    elif "guerras mundiales" in sub or "guerra fria" in sub:
        op = [
            "Las tensiones ideológicas y militares reordenaron alianzas y motivaron organismos multilaterales.",
            "Los conflictos se limitaron al comercio sin efectos sociales.",
            "Los bloques surgieron al azar sin ideas en juego.",
            "No hubo consecuencias políticas relevantes."
        ]
    elif "geografia de colombia" in sub:
        op = [
            "Relieve, clima y recursos condicionan asentamientos y actividades; la infraestructura busca conectar regiones.",
            "Economía y territorio no se relacionan.",
            "La población se distribuye al azar.",
            "La infraestructura elimina todas las barreras."
        ]
    else:
        op = [
            "Amplía mercados y opciones, pero exige políticas para mitigar desigualdades y fortalecer capacidades.",
            "Siempre distribuye beneficios de forma uniforme.",
            "Impide el desarrollo local por definición.",
            "Solo afecta a grandes empresas."
        ]
    return {
        "area": cfg.area, "subtema": cfg.subtema, "dificultad": cfg.dificultad,
        "estilo_kolb": cfg.estilo_kolb or "Asimilador",
        "pregunta": enun + " A partir de ello, elige la opción que mejor sintetiza la finalidad y alcance del tema estudiado.",
        "opciones": {"A":op[0],"B":op[1],"C":op[2],"D":op[3]},
        "respuesta_correcta": "A",
        "explicacion": "",
        "meta": {"source":"rule-fallback","eje":"sintesis-finalidad"}
    }

def rb_ciencias(cfg: GenInput) -> dict:
    sub = _norm(cfg.subtema)
    if "indagacion cientifica" in sub:
        texto = "Un equipo prueba el efecto de la luz (horas/día) en la germinación de semillas a temperatura y humedad constantes."
        op = [
            "VI: horas de luz; VD: tasa de germinación; controles: temperatura y humedad.",
            "VI: humedad; VD: horas de luz; controles: germinación.",
            "VI: temperatura; VD: humedad; controles: luz.",
            "VI y VD intercambiadas; sin controles definidos."
        ]
        correct = "A"; meta = {"VI":"horas de luz","VD":"tasa de germinación","controles":["temperatura","humedad"]}
    elif "fuerzas" in sub:
        texto = "Un carrito acelera en línea recta; se mide masa, fuerza aplicada y variación de velocidad."
        op = [
            "La aceleración es proporcional a la fuerza e inversamente proporcional a la masa.",
            "La aceleración no depende de la masa.",
            "La fuerza solo cambia la dirección, nunca el módulo.",
            "Si no hay fuerza, siempre hay aceleración."
        ]
        correct = "A"; meta = {"ley":"Segunda ley de Newton"}
    elif "materia y cambios" in sub:
        texto = "Se mezclan sal y agua; luego se evapora el solvente y se recupera el soluto."
        op = [
            "La masa total se conserva y la mezcla es homogénea; la evaporación permite separar componentes.",
            "La masa no se conserva en procesos físicos.",
            "Las mezclas homogéneas no se pueden separar.",
            "La sal reacciona y desaparece."
        ]
        correct = "A"; meta = {"proceso":"mezcla/evaporacion"}
    elif "genetica" in sub:
        texto = "Cruce monohíbrido con dominancia completa; se esperan proporciones fenotípicas 3:1 en F2."
        op = [
            "La segregación mendeliana explica la proporción 3:1.",
            "Las proporciones dependen solo del ambiente.",
            "Sin meiosis no hay segregación.",
            "El fenotipo recesivo domina."
        ]
        correct = "A"; meta = {"modelo":"mendeliano"}
    else:
        texto = "Se analiza el impacto de cambios de uso del suelo en la biodiversidad y el balance de carbono."
        op = [
            "Las alteraciones del ecosistema afectan ciclos y diversidad; es clave mitigar y adaptarse.",
            "No hay relación entre uso del suelo y clima.",
            "La biodiversidad no incide en la resiliencia.",
            "Mitigar no es necesario si hay adaptación."
        ]
        correct = "A"; meta = {"enfoque":"CTS"}
    return {
        "area": cfg.area, "subtema": cfg.subtema, "dificultad": cfg.dificultad,
        "estilo_kolb": cfg.estilo_kolb or "Acomodador",
        "pregunta": texto + " Con base en este diseño/escenario, selecciona la conclusión más adecuada.",
        "opciones": {"A":op[0],"B":op[1],"C":op[2],"D":op[3]},
        "respuesta_correcta": correct,
        "explicacion": "",
        "meta": meta | {"source":"rule-fallback"}
    }

def rb_ingles(cfg: GenInput) -> dict:
    sub = _norm(cfg.subtema)
    if "verb to be" in sub:
        s = [
            ("I ___ a student.","am","is","are","be","A"),
            ("They ___ at home.","are","is","am","be","A"),
        ][random.randint(0,1)]
    elif "present simple" in sub:
        s = [
            ("She ___ to school every day.","goes","go","went","going","A"),
            ("We ___ coffee in the morning.","drink","drinks","drank","drinking","A"),
        ][random.randint(0,1)]
    elif "past simple" in sub:
        s = [
            ("They ___ the match yesterday.","won","win","wins","winning","A"),
            ("He ___ a new book last week.","bought","buy","buys","buying","A"),
        ][random.randint(0,1)]
    elif "comparatives" in sub:
        s = [
            ("This car is ___ than that one.","faster","fast","fastest","more fast","A"),
            ("My house is ___ than yours.","bigger","big","more big","biggest","A"),
        ][random.randint(0,1)]
    else:  # pronouns/possessives
        s = [
            ("This is ___ book, not yours.","my","me","mine","I","A"),
            ("Can you help ___ with this?","me","I","mine","my","A"),
        ][random.randint(0,1)]
    return {
        "area": cfg.area, "subtema": cfg.subtema, "dificultad": cfg.dificultad,
        "estilo_kolb": cfg.estilo_kolb or "Convergente",
        "pregunta": f"Completa la oración correctamente: {s[0]}",
        "opciones": {"A":s[1], "B":s[2], "C":s[3], "D":s[4]},
        "respuesta_correcta": s[5],
        "explicacion": "",
        "meta": {"source":"rule-fallback","target":sub}
    }

def rule_based_item(cfg: GenInput) -> dict:
    if "matem" in _norm(cfg.area):
        if is_sys2x2(cfg.subtema): return rb_mate_2x2(cfg)
        return rb_mate_simple(cfg)
    if "lenguaj" in _norm(cfg.area):   return rb_lang(cfg)
    if "sociales" in _norm(cfg.area):  return rb_sociales(cfg)
    if "ciencias naturales" in _norm(cfg.area): return rb_ciencias(cfg)
    if "ingles" in _norm(cfg.area):    return rb_ingles(cfg)
    # fallback genérico
    return rb_lang(cfg)


# =================== Generación LLM (una/packs) + relleno ===================
def compute_num_predict(cantidad: int, max_tokens_item: int) -> int:
    return int(max(240, min(1400, cantidad * max(180, min(max_tokens_item, 360)))))

def generar_una_raw(cfg: GenInput, need_meta: bool=False) -> dict:
    messages = [
        {"role":"system","content": build_system()},
        {"role":"user","content": build_user(cfg)}
    ]
    if need_meta and is_sys2x2(cfg.subtema):
        messages.append({"role":"user","content":"RECUERDA: incluye 'meta' (sistema 2x2 con coeficientes)."})
    # Intento JSON forzado
    try:
        raw = chat_once(messages, max_tokens=cfg.max_tokens_item, temperature=cfg.temperatura, force_json=True)
        data = parse_json_min(raw)
    except Exception:
        # Fallback recordatorio
        messages2 = messages + [
            {"role":"assistant","content":"Formato incorrecto."},
            {"role":"user","content":"Reenvía SOLO el JSON EXACTO solicitado; sin texto fuera del JSON."}
        ]
        raw = chat_once(messages2, max_tokens=cfg.max_tokens_item, temperature=0.0, force_json=False)
        data = parse_json_min(raw)
    ensure_min_schema(data); enforce_whitelist(cfg, data)
    return data

def generar_una(cfg: GenInput) -> IcfesPreguntaOut:
    # Dos intentos para forzar meta en 2x2
    for i in range(2):
        data = generar_una_raw(cfg, need_meta=(i==1))
        try:
            data = ensure_math_consistency(cfg, data)  # si aplica
            break
        except Exception as e:
            if is_sys2x2(cfg.subtema) and i == 0: continue
            else: raise
    data = postprocess_item(cfg, data)
    return IcfesPreguntaOut(
        area=cfg.area, subtema=cfg.subtema, dificultad=cfg.dificultad,
        estilo_kolb=cfg.estilo_kolb or "Convergente",
        pregunta=data["pregunta"], opciones=data["opciones"],
        respuesta_correcta=data["respuesta_correcta"],
        explicacion=data.get("explicacion",""),
        meta=data.get("meta", {})
    )

def generar_pack_llm(cfg: GenInput, cantidad: int) -> List[dict]:
    """Pide al modelo N ítems en {items:[...]} y devuelve la lista cruda (dicts)."""
    messages = [
        {"role":"system","content": build_system()},
        {"role":"user","content": build_pack_user(cfg, cantidad)}
    ]
    total_tokens = compute_num_predict(cantidad, cfg.max_tokens_item)
    # Intento JSON forzado
    try:
        raw = chat_once(messages, max_tokens=total_tokens, temperature=cfg.temperatura, force_json=True)
        data = parse_json_min(raw)
    except Exception:
        messages2 = messages + [
            {"role":"assistant","content":"Formato incorrecto."},
            {"role":"user","content":"Devuelve SOLO el JSON EXACTO {\"items\":[...]}; sin texto fuera."}
        ]
        raw = chat_once(messages2, max_tokens=total_tokens, temperature=0.0, force_json=False)
        data = parse_json_min(raw)
    items = data["items"] if isinstance(data, dict) and isinstance(data.get("items"), list) else []
    return items

def completar_hasta_cinco(cfg: GenInput, recogidos: List[IcfesPreguntaOut], vistos: set, objetivo: int=5) -> List[IcfesPreguntaOut]:
    """Rellena hasta alcanzar 5 con (1) llamadas unitarias al LLM y (2) fallback local."""
    # (1) LLM unitario con varios intentos
    intentos_max = objetivo * 6
    intentos = 0
    while len(recogidos) < objetivo and intentos < intentos_max:
        try:
            q = generar_una(cfg)
            key = _norm(q.pregunta)
            if key in vistos: intentos += 1; continue
            vistos.add(key); recogidos.append(q)
        except Exception:
            # falló el LLM → probamos fallback local
            rb = rule_based_item(cfg)
            rb = postprocess_item(cfg, rb)
            q = IcfesPreguntaOut(
                area=cfg.area, subtema=cfg.subtema, dificultad=cfg.dificultad,
                estilo_kolb=cfg.estilo_kolb or "Convergente",
                pregunta=rb["pregunta"], opciones=rb["opciones"],
                respuesta_correcta=rb["respuesta_correcta"], explicacion=rb.get("explicacion",""),
                meta=rb.get("meta", {})
            )
            key = _norm(q.pregunta)
            if key in vistos: intentos += 1; continue
            vistos.add(key); recogidos.append(q)
        intentos += 1

    # (2) Si aún faltan (muy raro), completa solo con fallback local
    while len(recogidos) < objetivo:
        rb = rule_based_item(cfg)
        rb = postprocess_item(cfg, rb)
        q = IcfesPreguntaOut(
            area=cfg.area, subtema=cfg.subtema, dificultad=cfg.dificultad,
            estilo_kolb=cfg.estilo_kolb or "Convergente",
            pregunta=rb["pregunta"], opciones=rb["opciones"],
            respuesta_correcta=rb["respuesta_correcta"], explicacion=rb.get("explicacion",""),
            meta=rb.get("meta", {})
        )
        key = _norm(q.pregunta)
        if key in vistos: continue
        vistos.add(key); recogidos.append(q)
    return recogidos

def generar_pack(cfg: GenInput, cantidad: int) -> List[IcfesPreguntaOut]:
    """Genera N ítems → valida → post-procesa → completa hasta 5 garantizado."""
    items_raw = generar_pack_llm(cfg, cantidad)
    out: List[IcfesPreguntaOut] = []
    vistos = set()
    # procesa lo que venga bien del LLM
    for obj in items_raw:
        try:
            ensure_min_schema(obj); enforce_whitelist(cfg, obj)
            obj = ensure_math_consistency(cfg, obj)
            obj = postprocess_item(cfg, obj)
            key = _norm(obj["pregunta"])
            if key in vistos: continue
            vistos.add(key)
            out.append(IcfesPreguntaOut(
                area=cfg.area, subtema=cfg.subtema, dificultad=cfg.dificultad,
                estilo_kolb=cfg.estilo_kolb or "Convergente",
                pregunta=obj["pregunta"], opciones=obj["opciones"],
                respuesta_correcta=obj["respuesta_correcta"],
                explicacion=obj.get("explicacion",""),
                meta=obj.get("meta", {})
            ))
        except Exception:
            continue

    # ahora COMPLETAMOS hasta llegar a 'cantidad' (por defecto, 5)
    out = completar_hasta_cinco(cfg, out, vistos, objetivo=cantidad)
    return out


# ================================= Endpoints ================================
@app.post("/icfes/generar")   # 1 ítem (compatibilidad)
@app.post("/generar")
def icfes_generar_alias(cfg: GenInput):
    try:
        q = generar_una(cfg)
        return {"ok": True, "generadas": 1, "resultados": [q.model_dump()], "errores": []}
    except Exception as e:
        # Incluso aquí, garantizamos 1 con fallback local
        rb = rule_based_item(cfg)
        rb = postprocess_item(cfg, rb)
        q = IcfesPreguntaOut(
            area=cfg.area, subtema=cfg.subtema, dificultad=cfg.dificultad,
            estilo_kolb=cfg.estilo_kolb or "Convergente",
            pregunta=rb["pregunta"], opciones=rb["opciones"],
            respuesta_correcta=rb["respuesta_correcta"], explicacion=rb.get("explicacion",""),
            meta=rb.get("meta", {})
        )
        return {"ok": True, "generadas": 1, "resultados": [q.model_dump()], "errores": [{"index": 0, "aviso": str(e)}]}

@app.post("/icfes/generar_pack")   # SIEMPRE devuelve 'cantidad' (por defecto 5)
def icfes_generar_pack(cfg: GenInput, cantidad: int = 5):
    try:
        preguntas = generar_pack(cfg, cantidad=cantidad)
        return {"ok": True, "generadas": len(preguntas), "resultados": [p.model_dump() for p in preguntas], "errores": []}
    except Exception as e:
        # Si falló todo, aun así garantizamos 'cantidad' con fallback local
        out: List[IcfesPreguntaOut] = []
        vistos = set()
        out = completar_hasta_cinco(cfg, out, vistos, objetivo=cantidad)
        return {"ok": True, "generadas": len(out), "resultados": [p.model_dump() for p in out], "errores": [{"index": 0, "aviso": str(e)}]}

@app.get("/health")
def health():
    return {
        "backend": BACKEND,
        "model": {"hf": HF_MODEL, "ollama": OLLAMA_MODEL, "groq": GROQ_MODEL}.get(BACKEND, "?"),
        "timeouts": {"hf": HF_TIMEOUT, "ollama": OLLAMA_TIMEOUT},
        "token_loaded": bool(HF_API_KEY or GROQ_API_KEY or BACKEND=="ollama")
    }
