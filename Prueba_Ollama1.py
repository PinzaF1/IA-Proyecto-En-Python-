# =============================================================================
# EduExcel_app.py  (SOLO OLLAMA • Modo estricto • SIN fallback local)
# -----------------------------------------------------------------------------
# - Lista blanca de subtemas (rechaza lo no permitido)
# - Fuerza salida JSON + validación estricta (sin texto extra)
# - Enunciados largos (100–140 palabras por defecto)
# - Mezcla opciones A–D y genera explicación por opción
# - Verificación/autocorrección para Matemáticas → sistemas 2×2 (con meta)
# - SOLO backend Ollama (recomendado: qwen2.5:7b-instruct por defecto)
# - PACK: intenta N ítems con reintentos; en modo estricto, si no llega a N -> error
# - Estilo de aprendizaje de Kolb integrado (Divergente/Asimilador/Convergente/Acomodador)
# =============================================================================

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
import os, json, re, unicodedata, random, requests

# ============================= Configuración ================================
load_dotenv(find_dotenv(), override=True)

# Ollama
OLLAMA_URL     = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# Modo estricto: NO usa plantillas ni rule-based; si el LLM falla, se reporta error.
STRICT_MODE = os.getenv("STRICT_MODE", "1") == "1"
MAX_REINTENTOS_LLM = int(os.getenv("MAX_REINTENTOS_LLM", "3"))

DEBUG = os.getenv("DEBUG_JSON", "0") == "1"

app = FastAPI(title=f"EduExcel (backend=ollama, strict={STRICT_MODE})")

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
    return ("sistema" in t or "ecuaciones lineales" in t) and ("2x2" in t)

def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or "", flags=re.UNICODE))

def pad_to_range(texto: str, min_palabras: int, max_palabras: int, extras: List[str]) -> str:
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

    # Matemáticas 2×2
    if "matem" in _norm(cfg.area) and "2x2" in _norm(cfg.subtema) and isinstance(meta, dict):
        try:
            a1=meta["a1"]; b1=meta["b1"]; c1=meta["c1"]; a2=meta["a2"]; b2=meta["b2"]; c2=meta["c2"]
            sol_txt = opciones[correcta]
            textos.append(f"Se resuelve el sistema (a1={a1}, b1={b1}, c1={c1}; a2={a2}, b2={b2}, c2={c2}) por eliminación o Cramer.")
            def razon(op_txt):
                return "Satisface ambas ecuaciones." if op_txt == sol_txt \
                       else "No satisface al menos una ecuación (error típico de signo o combinación)."
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

    # Fallback explicativo general (no de contenido)
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

def kolb_brief(estilo: str | None) -> str:
    s = _norm(estilo or "")
    if "diverg" in s:
        return ("DISEÑO KOLB (Divergente): escenario abierto y contextual; "
                "fomenta observación y múltiples perspectivas; caso realista.")
    if "asimil" in s:
        return ("DISEÑO KOLB (Asimilador): datos/tabla para abstraer conceptos; "
                "énfasis en relaciones y síntesis conceptual.")
    if "conver" in s:
        return ("DISEÑO KOLB (Convergente): problema bien definido con solución única; "
                "aplicación de reglas/procedimientos claros.")
    if "acomod" in s:
        return ("DISEÑO KOLB (Acomodador): situación práctica con decisión y consecuencias; "
                "énfasis en acción y prueba-error.")
    return "DISEÑO KOLB: aplica explícitamente el estilo indicado en el enunciado."

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
        "Usa EXCLUSIVAMENTE los subtemas de la lista blanca; si la instrucción contradice esa lista, responde con JSON de error."
        f" Lenguaje: {_join('Lenguaje')}. Matemáticas: {_join('Matemáticas')}. "
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
        f"Genera UNA pregunta del área {cfg.area}, subtema EXACTO {cfg.subtema}. "
        f"Dificultad {cfg.dificultad}. Estilo Kolb: {cfg.estilo_kolb or 'Convergente'}. {kolb_brief(cfg.estilo_kolb)} "
        f"LONG_MIN {cfg.longitud_min}, LONG_MAX {cfg.longitud_max}. "
        "Devuelve SOLO el JSON del esquema indicado; sin markdown, sin texto adicional. "
        "No cambies 'area' ni 'subtema'. No inventes subtemas. Español estricto (salvo área Inglés). "
        + extra
    )

def build_pack_user(cfg: GenInput, cantidad: int) -> str:
    nota_2x2 = (
        "Cada objeto incluye sistema 2x2 y meta con coeficientes; opciones como (x, y)."
        if is_sys2x2(cfg.subtema) else ""
    )
    return (
        "Devuelve SOLO: {\"items\":[OBJ1,...,OBJN]} con EXACTAMENTE N ítems del esquema indicado. "
        f"N = {cantidad}. area={cfg.area}; subtema={cfg.subtema}; dificultad={cfg.dificultad}; "
        f"estilo_kolb={cfg.estilo_kolb or 'Convergente'}; LONG_MIN={cfg.longitud_min}; LONG_MAX={cfg.longitud_max}. "
        "Sin texto fuera del JSON. Ítems no repetidos. " + nota_2x2
    )

# ============================ Llamada a Ollama ==============================
def chat_backend_ollama(messages, max_tokens: int, temperature: float, force_json: bool) -> str:
    url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "8m",
        "options": {
            "temperature": 0.1 if force_json else temperature,
            "top_p": 0.9,
            "mirostat": 0,
            "repeat_penalty": 1.07,
            "num_predict": max_tokens,
            "num_ctx": 4096,
            "seed": 42
        }
    }
    if force_json:
        payload["format"] = "json"
    r = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT)
    if r.status_code >= 400:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    return (data.get("message") or {}).get("content", "")

def chat_once(messages, max_tokens: int, temperature: float, force_json: bool) -> str:
    text = chat_backend_ollama(messages, max_tokens=max_tokens, temperature=temperature, force_json=force_json)
    _dbg(("RAW(JSON)>> " if force_json else "RAW(PLAIN)>> ") + (text or "")[:1200])
    return text or ""

# ===================== Parsing + validaciones de esquema ====================
def parse_json_min(text: str) -> dict:
    if not text: raise ValueError("Salida vacía del modelo.")
    s = text.replace("```json","{}").replace("```","")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: raise ValueError("No se detectó JSON en la salida.")
    blob = m.group(0)
    blob = re.sub(r",\s*([}\]])", r"\1", blob)
    obj = json.loads(blob)
    if isinstance(obj, dict) and "error" in obj:
        raise ValueError(obj.get("error") or "Error del generador")
    return obj

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

# =================== Generación LLM (una/packs) sin fallback ===============
def compute_num_predict(cantidad: int, max_tokens_item: int) -> int:
    return int(max(240, min(1400, cantidad * max(180, min(max_tokens_item, 360)))))

def generar_una_raw(cfg: GenInput, need_meta: bool=False) -> dict:
    messages = [
        {"role":"system","content": build_system()},
        {"role":"user","content": build_user(cfg)}
    ]
    if need_meta and is_sys2x2(cfg.subtema):
        messages.append({"role":"user","content":"RECUERDA: incluye 'meta' (sistema 2x2 con coeficientes)."})
    raw = chat_once(messages, max_tokens=cfg.max_tokens_item, temperature=cfg.temperatura, force_json=True)
    if not raw:
        raw = chat_once(messages, max_tokens=cfg.max_tokens_item, temperature=0.0, force_json=True)
    data = parse_json_min(raw)
    ensure_min_schema(data); enforce_whitelist(cfg, data)
    return data

def generar_una(cfg: GenInput) -> IcfesPreguntaOut:
    last_exc = None
    for i in range(2):
        try:
            data = generar_una_raw(cfg, need_meta=(i==1))
            data = ensure_math_consistency(cfg, data)  # si aplica
            data = postprocess_item(cfg, data)
            return IcfesPreguntaOut(
                area=cfg.area, subtema=cfg.subtema, dificultad=cfg.dificultad,
                estilo_kolb=cfg.estilo_kolb or "Convergente",
                pregunta=data["pregunta"], opciones=data["opciones"],
                respuesta_correcta=data["respuesta_correcta"],
                explicacion=data.get("explicacion",""),
                meta=data.get("meta", {})
            )
        except Exception as e:
            last_exc = e
    raise last_exc or RuntimeError("No se pudo generar el ítem.")

def generar_pack_llm(cfg: GenInput, cantidad: int) -> List[dict]:
    messages = [
        {"role":"system","content": build_system()},
        {"role":"user","content": build_pack_user(cfg, cantidad)}
    ]
    total_tokens = compute_num_predict(cantidad, cfg.max_tokens_item)
    raw = chat_once(messages, max_tokens=total_tokens, temperature=cfg.temperatura, force_json=True)
    if not raw:
        raw = chat_once(messages, max_tokens=total_tokens, temperature=0.0, force_json=True)
    data = parse_json_min(raw)
    items = data["items"] if isinstance(data, dict) and isinstance(data.get("items"), list) else []
    return items

def generar_pack(cfg: GenInput, cantidad: int) -> List[IcfesPreguntaOut]:
    items_raw = []
    for _ in range(MAX_REINTENTOS_LLM):
        try:
            items_raw = generar_pack_llm(cfg, cantidad)
            if items_raw and len(items_raw) == cantidad:
                break
        except Exception:
            items_raw = []
    out: List[IcfesPreguntaOut] = []
    vistos = set()
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
    intentos = 0
    while len(out) < cantidad and intentos < MAX_REINTENTOS_LLM * cantidad:
        try:
            q = generar_una(cfg)
            key = _norm(q.pregunta)
            if key in [ _norm(p.pregunta) for p in out ]:
                intentos += 1; continue
            out.append(q)
        except Exception:
            intentos += 1; continue
    return out

# ================================= Endpoints ================================
@app.post("/icfes/generar")
@app.post("/generar")
def icfes_generar_alias(cfg: GenInput):
    try:
        q = generar_una(cfg)
        return {"ok": True, "generadas": 1, "resultados": [q.model_dump()], "errores": []}
    except Exception as e:
        if STRICT_MODE:
            return {"ok": False, "generadas": 0, "resultados": [], "errores": [{"index": 0, "aviso": str(e)}]}
        return {"ok": False, "generadas": 0, "resultados": [], "errores": [{"index": 0, "aviso": str(e)}]}

@app.post("/icfes/generar_pack")
def icfes_generar_pack(cfg: GenInput, cantidad: int = 5):
    try:
        preguntas = generar_pack(cfg, cantidad=cantidad)
        if STRICT_MODE and len(preguntas) < cantidad:
            return {"ok": False, "generadas": len(preguntas), "resultados": [p.model_dump() for p in preguntas],
                    "errores": [{"index": -1, "aviso": f"Se generaron {len(preguntas)} de {cantidad} en modo estricto."}]}
        return {"ok": True, "generadas": len(preguntas), "resultados": [p.model_dump() for p in preguntas], "errores": []}
    except Exception as e:
        return {"ok": False, "generadas": 0, "resultados": [], "errores": [{"index": 0, "aviso": str(e)}]}

@app.get("/health")
def health():
    return {
        "backend": "ollama",
        "model": OLLAMA_MODEL,
        "timeouts": {"ollama": OLLAMA_TIMEOUT},
        "strict_mode": STRICT_MODE
    }
