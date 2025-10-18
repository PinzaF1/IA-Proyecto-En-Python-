# app10.py — FastAPI + Ollama (Qwen 7B/3B)
# ------------------------------------------------------------
# - Sin "dificultad" en la entrada (se elimina del esquema).
# - Verificación/normalización de área, subtema y estilo Kolb.
# - Parser robusto de JSON del LLM + normalizador de claves.
# - Saneado numérico: elimina '+' delante de enteros positivos.
# - Coherencia explicación <-> respuesta_correcta (arreglo automático).
# - meta siempre dict; si viene "", se convierte en {}.
# - Explicaciones por área (plantilla) si el modelo trae una pobre.
# - Endpoints: /icfes/catalogo, /icfes/validar, /icfes/generar, /icfes/generar_pack, /debug/raw
# ------------------------------------------------------------

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
import os, json, re, requests, random, unicodedata, difflib, time

# ===================== Configuración =====================
load_dotenv(find_dotenv(), override=True)
OLLAMA_URL       = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")  # puedes cambiar a 3B si gustas
OLLAMA_TIMEOUT   = int(os.getenv("OLLAMA_TIMEOUT", "600"))
STRICT_MODE      = os.getenv("STRICT_MODE", "1") == "1"
DEBUG_JSON       = os.getenv("DEBUG_JSON", "0") == "1"
SEED_RANDOMIZE   = os.getenv("SEED_RANDOMIZE", "1") == "1"   # semilla aleatoria por request

app = FastAPI(title=f"ICFES API (ollama={OLLAMA_MODEL}, strict={STRICT_MODE})")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

def _dbg(msg: str):
    if DEBUG_JSON: print(str(msg)[:2000])

# ===================== Catálogo (áreas, subtemas, estilos) =====================
ALLOWED: Dict[str, List[str]] = {
    "Lenguaje": [
        "Comprensión lectora (sentido global y local)",
        "Conectores lógicos (causa, contraste, condición, secuencia)",
        "Identificación de argumentos y contraargumentos",
        "Idea principal y propósito comunicativo",
        "Hecho vs. opinión e inferencias",
    ],
    "Matemáticas": [
        "Operaciones con números enteros",
        "Razones y proporciones",
        "Regla de tres simple y compuesta",
        "Porcentajes y tasas (aumento, descuento, interés simple)",
        "Ecuaciones lineales y sistemas 2×2",
    ],
    "Sociales y Ciudadanas": [
        "Constitución de 1991 y organización del Estado",
        "Historia de Colombia (Frente Nacional, conflicto y paz)",
        "Guerras Mundiales y Guerra Fría",
        "Geografía de Colombia (mapas, territorio y ambiente)",
        "Economía y ciudadanía económica (globalización y desigualdad)",
    ],
    "Ciencias Naturales": [
        "Indagación científica (variables, control e interpretación de datos)",
        "Fuerzas, movimiento y energía",
        "Materia y cambios (mezclas, reacciones y conservación)",
        "Genética y herencia",
        "Ecosistemas y cambio climático (CTS)",
    ],
    "Inglés": [
        "Verb to be (am, is, are)",
        "Present Simple (afirmación, negación y preguntas)",
        "Past Simple (verbos regulares e irregulares)",
        "Comparatives and superlatives",
        "Subject/Object pronouns y possessive adjectives",
    ],
}

KOLB_STYLES  = ["Convergente", "Asimilador", "Acomodador", "Divergente"]

# Guías de enunciado por subtema (para pedir preguntas “más largas”)
SUBTEMA_GUIDE = {
    "Matemáticas": {
        "Operaciones con números enteros":
            "Crea un mini-caso con saldo/temperatura en 2–3 eventos y varias operaciones encadenadas (evita signos + en positivos).",
        "Razones y proporciones":
            "Plantea mezcla/receta con proporción fija; incluye dos datos y pide el tercero (sin + en positivos).",
        "Regla de tres simple y compuesta":
            "Caso de obreros/tiempos o máquinas/producción; explícita si es directa o inversa (sin + en positivos).",
        "Porcentajes y tasas (aumento, descuento, interés simple)":
            "Precio inicial, descuento y un ajuste adicional (impuesto o recargo) para el total (sin + en positivos).",
        "Ecuaciones lineales y sistemas 2×2":
            "Dos ecuaciones con contexto y solución única; opciones como pares ordenados.",
    },
    "Lenguaje": {
        "Comprensión lectora (sentido global y local)":
            "Fragmento de 3–4 frases con datos y opiniones; pide sentido global sin confundir detalles.",
        "Conectores lógicos (causa, contraste, condición, secuencia)":
            "Incluye conectores variados; pregunta por el que mantiene la relación lógica.",
        "Identificación de argumentos y contraargumentos":
            "Incluye tesis, razones y contraargumento explícito; pide reconocerlos.",
        "Idea principal y propósito comunicativo":
            "Señala pistas de intención (informar/persuadir) y cierre; pide la síntesis central.",
        "Hecho vs. opinión e inferencias":
            "Combina datos verificables y juicios de valor; pide distinguir e inferir con evidencia.",
    },
    "Sociales y Ciudadanas": {
        "Constitución de 1991 y organización del Estado":
            "Menciona funciones/órganos y derechos; pide finalidad/alcance.",
        "Historia de Colombia (Frente Nacional, conflicto y paz)":
            "Contexto histórico (años, actores, objetivos) sin anacronismos; interpreta consecuencias.",
        "Guerras Mundiales y Guerra Fría":
            "Tensiones ideológicas y efectos geopolíticos; pregunta por el rasgo central.",
        "Geografía de Colombia (mapas, territorio y ambiente)":
            "Relieve/clima vs. asentamientos/actividades; elegir síntesis coherente.",
        "Economía y ciudadanía económica (globalización y desigualdad)":
            "Beneficios y desafíos; pide conclusión equilibrada y fundamentada.",
    },
    "Ciencias Naturales": {
        "Indagación científica (variables, control e interpretación de datos)":
            "Diseño experimental con VI/VD y dos controles; identificar correctamente.",
        "Fuerzas, movimiento y energía":
            "Caso con masa, fuerza y variación de velocidad; relacionar con la 2ª ley de Newton.",
        "Materia y cambios (mezclas, reacciones y conservación)":
            "Mezcla homogénea con separación por método físico; conservación de masa.",
        "Genética y herencia":
            "Cruce monohíbrido con dominancia completa; proporciones en F2.",
        "Ecosistemas y cambio climático (CTS)":
            "Cambio de uso del suelo y biodiversidad; conclusión basada en evidencia.",
    },
    "Inglés": {
        "Verb to be (am, is, are)":
            "Mini-diálogo con pistas de número/persona; forma correcta.",
        "Present Simple (afirmación, negación y preguntas)":
            "Rutinas diarias; terceras personas; -s y do/does.",
        "Past Simple (verbos regulares e irregulares)":
            "Adverbios de pasado; forma correcta irregular.",
        "Comparatives and superlatives":
            "Comparación de objetos concretos; cuidado con 'more/—er'.",
        "Subject/Object pronouns y possessive adjectives":
            "Ambigüedad sujeto/objeto/posesivo; elige forma adecuada.",
    },
}

# ===================== Normalización y verificación =====================
def _norm(s: str) -> str:
    s = (s or "").strip().lower().replace("×", "x")
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"\s+", " ", s)
    return s

def _closest(x: str, opciones: List[str]) -> Tuple[str, float]:
    import difflib
    matches = difflib.get_close_matches(x, opciones, n=1, cutoff=0.0)
    if not matches: return "", 0.0
    ratio = difflib.SequenceMatcher(None, x, matches[0]).ratio()
    return matches[0], ratio

def catalogo():
    return {
        "areas": list(ALLOWED.keys()),
        "subtemas_por_area": ALLOWED,
        "estilos_kolb": KOLB_STYLES,
        "kolb_descripcion": {
            "Convergente": "Aplicación práctica y solución única, con datos explícitos y pasos claros.",
            "Asimilador":  "Énfasis en conceptos, organización lógica y relaciones; más estructura conceptual.",
            "Acomodador":  "Contexto experiencial y toma de decisiones; casos situados y realistas.",
            "Divergente":  "Múltiples perspectivas y síntesis; escenario rico en matices pero con respuesta única.",
        },
    }

def validar_area(area: str) -> Tuple[Optional[str], Optional[str]]:
    if area in ALLOWED: return area, None
    mapa = { _norm(k): k for k in ALLOWED.keys() }
    nn = _norm(area)
    if nn in mapa: return mapa[nn], None
    mejor, _ = _closest(area, list(ALLOWED.keys()))
    return None, f"Área no permitida: '{area}'. ¿Quisiste decir '{mejor}'?"

def validar_subtema(area: str, subtema: str) -> Tuple[Optional[str], Optional[str]]:
    if area not in ALLOWED:
        return None, f"Área inválida: '{area}'."
    lst = ALLOWED[area]
    if subtema in lst: return subtema, None
    mapa = { _norm(x): x for x in lst }
    nn = _norm(subtema)
    if nn in mapa: return mapa[nn], None
    mejor, _ = _closest(subtema, lst)
    return None, f"Subtema no permitido para {area}: '{subtema}'. Sugerencia: '{mejor}'."

def validar_kolb(estilo: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not estilo: return "Convergente", None
    for k in KOLB_STYLES:
        if _norm(k) == _norm(estilo):
            return k, None
    mejor, _ = _closest(estilo, KOLB_STYLES)
    return None, f"Estilo Kolb no reconocido: '{estilo}'. Sugerencia: '{mejor}'."

# ===================== Modelos entrada/salida =====================
class GenInput(BaseModel):
    area: str
    subtema: str
    estilo_kolb: Optional[str] = None
    longitud_min: int = 130
    longitud_max: int = 180
    max_tokens_item: int = 400
    temperatura: float = 0.2

class ItemOut(BaseModel):
    area: str
    subtema: str
    estilo_kolb: Optional[str] = None
    pregunta: str = Field(min_length=10)
    opciones: Dict[str, str]
    respuesta_correcta: str
    explicacion: Optional[str] = ""
    meta: Optional[Dict[str, object]] = {}

# ===================== Parser/normalizador de salida del LLM =====================
def ensure_schema(d: dict):
    if not isinstance(d, dict): raise ValueError("Salida no es objeto JSON.")
    for k in ["pregunta", "opciones", "respuesta_correcta"]:
        if k not in d: raise ValueError(f"Falta '{k}'")
    if not isinstance(d["opciones"], dict): raise ValueError("'opciones' debe ser objeto")
    for k in ["A","B","C","D"]:
        if k not in d["opciones"]: raise ValueError(f"Falta opción {k}")
    if d["respuesta_correcta"] not in ("A","B","C","D"):
        raise ValueError("respuesta_correcta inválida (A/B/C/D)")
    d.setdefault("explicacion", "")
    # meta: siempre dict
    meta = d.get("meta", {})
    if not isinstance(meta, dict): meta = {}
    d["meta"] = meta

def parse_json_min(text: str) -> dict:
    if not text:
        raise ValueError("Salida vacía del modelo.")
    s = text.replace("```json", "").replace("```", "")
    s = s.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    t = s.strip()
    if t.startswith("{") and t.endswith("}"):
        return json.loads(t)
    m = re.search(r"\{.*\}", s, flags=re.S)
    if not m: raise ValueError("No se detectó JSON.")
    blob = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    return json.loads(blob)

def coerce_single_item(obj: dict) -> dict:
    if isinstance(obj, dict) and "items" in obj and isinstance(obj["items"], list) and obj["items"] and isinstance(obj["items"][0], dict):
        obj = obj["items"][0]
    return obj

def normalize_keys_es(d: dict) -> dict:
    if not isinstance(d, dict): return d
    # Pregunta
    for k in ["pregunta", "enunciado", "statement", "prompt", "stem", "question", "texto", "planteamiento"]:
        if k in d: d["pregunta"] = d.pop(k); break
    # Opciones
    if "opciones" not in d:
        for k in ["opciones", "alternativas", "choices", "options", "respuestas"]:
            if k in d:
                val = d.pop(k)
                if isinstance(val, dict):
                    d["opciones"] = {
                        "A": val.get("A") or val.get("a") or val.get("1") or val.get("option_a") or "",
                        "B": val.get("B") or val.get("b") or val.get("2") or val.get("option_b") or "",
                        "C": val.get("C") or val.get("c") or val.get("3") or val.get("option_c") or "",
                        "D": val.get("D") or val.get("d") or val.get("4") or val.get("option_d") or "",
                    }
                elif isinstance(val, list):
                    labs = ["A","B","C","D"]; d["opciones"] = {labs[i]: (val[i] if i < len(val) else "") for i in range(4)}
                else:
                    d["opciones"] = {"A":"", "B":"", "C":"", "D":""}
                break
    # Respuesta correcta
    if "respuesta_correcta" not in d:
        for k in ["respuesta_correcta", "respuesta", "correcta", "answer", "ans", "correct_option", "correct", "solucion", "solution"]:
            if k in d:
                ans_raw = str(d.pop(k)).strip().upper()
                if ans_raw in ["1","2","3","4"]:
                    ans_raw = {"1":"A","2":"B","3":"C","4":"D"}[ans_raw]
                if ans_raw not in ["A","B","C","D"]:
                    low = ans_raw.lower()
                    if "a" in low: ans_raw = "A"
                    elif "b" in low: ans_raw = "B"
                    elif "c" in low: ans_raw = "C"
                    elif "d" in low: ans_raw = "D"
                d["respuesta_correcta"] = re.sub(r"[^ABCD]", "", ans_raw) or "A"
                break
    # meta siempre dict
    d.setdefault("explicacion", "")
    meta = d.get("meta", {})
    if not isinstance(meta, dict): meta = {}
    d["meta"] = meta
    return d

# ===================== Longitud, saneado, barajado y explicación =====================
def _word_count(s: str) -> int:
    return len(re.findall(r"\w+", s or "", flags=re.UNICODE))

def pad_to_range(texto: str, min_palabras: int, max_palabras: int) -> str:
    extras = [
        "Lee cuidadosamente los indicios antes de decidir",
        "Contrasta propósito, procedimientos y evidencias del caso",
        "Evita confundir ejemplos con definiciones generales",
        "Verifica coherencia entre datos y conclusión elegida",
        "Selecciona la alternativa que mejor sintetiza la idea central"
    ]
    texto = (texto or "").strip()
    w, k = _word_count(texto), 0
    while w < min_palabras and k < len(extras):
        if texto and not texto.endswith((".", "?", "!", "…")): texto += ". "
        texto += extras[k]; k += 1; w = _word_count(texto)
    if w > max_palabras:
        sents = re.split(r"(?<=[.!?])\s+", texto)
        nuevo, cnt = [], 0
        for s in sents:
            sw = _word_count(s)
            if cnt + sw <= max_palabras: nuevo.append(s); cnt += sw
            else: break
        texto = " ".join(nuevo).strip()
    return texto

def remove_plus_on_positive(text: str) -> str:
    """
    Elimina '+' delante de enteros positivos en cadenas (ej: '+8' -> '8'),
    preservando negativos '-5'. Evita tocar signos dentro de palabras.
    """
    # Reemplaza +<numero> que NO tenga un dígito/decimal antes (inicio o separador no-numérico)
    def repl(m):
        num = m.group(2)
        return m.group(1) + num  # quita el '+'
    # Casos como " +8 ", "(+3)", ":+12", " +10°C"
    pattern = re.compile(r'(^|[^0-9\-\.,])\+(\d+(?:[.,]\d+)?)')
    return pattern.sub(repl, text)

def clean_options_signs(opciones: Dict[str, str]) -> Dict[str, str]:
    out = {}
    for k, v in opciones.items():
        if isinstance(v, str):
            out[k] = remove_plus_on_positive(v)
        else:
            out[k] = v
    return out

def shuffle_options(opciones: dict, correcta_label: str) -> Tuple[dict, str]:
    labels = ["A", "B", "C", "D"]
    pairs = [(k, opciones.get(k, "").strip()) for k in labels]
    if len(set(v for _, v in pairs)) < 4: return opciones, correcta_label
    random.shuffle(pairs)
    new_op, new_label = {}, None
    for i, (old_label, text) in enumerate(pairs):
        lab = labels[i]; new_op[lab] = text
        if old_label == correcta_label: new_label = lab
    return new_op, (new_label or correcta_label)

def build_explanation_per_area(area: str, correcta: str) -> str:
    if "matem" in _norm(area):
        base = [
            f"Correcta ({correcta}).",
            "A: Aplica correctamente las operaciones requeridas.",
            "B: Presenta error de signos u orden de operaciones.",
            "C: Confunde la relación de proporcionalidad o el cálculo intermedio.",
            "D: Conclusión que no se deduce del enunciado."
        ]
    elif "lenguaj" in _norm(area):
        base = [
            f"Correcta ({correcta}).",
            "A: Resume la idea central con soporte textual.",
            "B: Confunde un detalle local con la tesis del texto.",
            "C: Generaliza más allá de la evidencia.",
            "D: Atribuye una intención no respaldada."
        ]
    elif "sociales" in _norm(area):
        base = [
            f"Correcta ({correcta}).",
            "A: Sintetiza finalidad/alcance con coherencia histórica.",
            "B: Confunde propósito con procedimiento o episodio aislado.",
            "C: Reduce el análisis a un caso puntual sin generalidad.",
            "D: Contradice la evidencia del proceso descrito."
        ]
    elif "ciencias" in _norm(area):
        base = [
            f"Correcta ({correcta}).",
            "A: Identifica variables y controles coherentes con el método.",
            "B: Intercambia VI y VD o ignora controles.",
            "C: Toma un control como variable principal.",
            "D: Conclusión no sustentada por el diseño."
        ]
    else:  # Inglés u otros
        base = [
            f"Correcta ({correcta}).",
            "A: Respeta la regla objetivo (forma/concordancia).",
            "B: Error de concordancia sujeto–verbo.",
            "C: Tiempo verbal incorrecto o forma no válida.",
            "D: Pronombre/posesivo mal seleccionado."
        ]
    return " ".join(base)

def fix_explanation_coherence(explicacion: str, correcta: str, area: str) -> str:
    """
    Detecta si la explicación afirma otra letra como correcta
    (patrones: 'Correcta (X)', 'La respuesta correcta es X', etc.)
    y reescribe la explicación para que coincida con 'correcta'.
    """
    if not isinstance(explicacion, str):
        return build_explanation_per_area(area, correcta)

    # Si contiene otra letra explícita, ignoramos y reescribimos
    patt = re.compile(r"(correcta\s*\(?\s*([ABCD])\s*\)?|la\s+respuesta\s+correcta\s+es\s+([ABCD]))", re.I)
    m = patt.search(explicacion)
    if m:
        return build_explanation_per_area(area, correcta)

    # Si está vacía o muy corta, construimos una por área
    if len(explicacion.strip()) < 12:
        return build_explanation_per_area(area, correcta)

    # Aseguramos que mencione la correcta al inicio
    if not re.search(rf"\b{re.escape(correcta)}\b", explicacion):
        # Ante la duda, anteponemos un encabezado consistente:
        encabezado = f"Correcta ({correcta}). "
        return encabezado + explicacion.strip()

    return explicacion

# ===================== Prompts (con guía y notas de consistencia) =====================
def system_prompt() -> str:
    return (
        "Eres un generador experto de ÍTEMS tipo ICFES en ESPAÑOL. "
        "DEVUELVES EXCLUSIVAMENTE JSON VÁLIDO (sin Markdown ni texto fuera del JSON). "
        "Esquema por ítem: {\"area\":\"\",\"subtema\":\"\",\"estilo_kolb\":\"\",\"pregunta\":\"\","
        "\"opciones\":{\"A\":\"\",\"B\":\"\",\"C\":\"\",\"D\":\"\"},\"respuesta_correcta\":\"\",\"explicacion\":\"\",\"meta\":{}} "
        "Para varias preguntas: {\"items\":[OBJ1,...,OBJN]}. "
        "Reglas: Español (salvo área Inglés), LONG_MIN..LONG_MAX palabras, 4 opciones A–D y única correcta, "
        "explicación breve y coherente con 'respuesta_correcta'. "
        "No contradigas la 'respuesta_correcta' en la explicación; si detectas inconsistencia, ajusta la explicación."
    )

def user_prompt(cfg: GenInput) -> str:
    estilo = cfg.estilo_kolb or "Convergente"
    guide  = SUBTEMA_GUIDE.get(cfg.area, {}).get(cfg.subtema, "Incluye un mini-caso realista de 2–3 frases.")
    # Nota de consistencia específica para Sociales (evitar anacronismos)
    sociales_note = ""
    if "sociales" in _norm(cfg.area):
        sociales_note = (
            " Evita anacronismos y atribuciones erróneas de actores/fechas. "
            "No confundas 'Frente Nacional' con procesos posteriores; conserva coherencia histórica."
        )
    # Nota para Matemáticas (evitar '+')
    mates_note = ""
    if "matem" in _norm(cfg.area):
        mates_note = " No uses el signo '+' delante de enteros positivos en enunciado u opciones."

    return (
        f"Genera UNA pregunta del área {cfg.area}, subtema EXACTO {cfg.subtema}. "
        f"Estilo Kolb: {estilo}. "
        f"Usa este enfoque: {guide}{sociales_note}{mates_note} "
        f"Texto largo con mini-caso; evita preguntas de una sola frase. "
        f"LONG_MIN {cfg.longitud_min}, LONG_MAX {cfg.longitud_max}. "
        "Devuelve SOLO el JSON del esquema indicado; sin texto adicional. "
        "Varía números, nombres y contexto; evita repetir patrones."
    )

# ===================== Llamada a Ollama =====================
def chat_ollama(messages: List[dict], max_tokens: int, temperature: float, force_json: bool=True) -> str:
    seed_val = random.randint(1, 10_000_000) if SEED_RANDOMIZE else 42
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "8m",
        "options": {
            "temperature": 0.1 if force_json else temperature,  # más estricto con format=json
            "mirostat": 0,
            "repeat_penalty": 1.07,
            "num_predict": max_tokens,
            "num_ctx": 2048,
            "seed": seed_val,
        },
    }
    if force_json:
        payload["format"] = "json"
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content", "") or ""
    _dbg(f"RAW(JSON)>> seed={seed_val} :: " + content[:1000])
    return content

# ===================== Validación de entrada =====================
def validar_input(cfg: GenInput) -> Tuple[GenInput, List[str]]:
    errores = []

    area_ok, err = validar_area(cfg.area)
    if err: errores.append(err)
    sub_ok, err2 = (None, "Primero corrige el área.") if not area_ok else validar_subtema(area_ok, cfg.subtema)
    if err2: errores.append(err2)
    kolb_ok, err3 = validar_kolb(cfg.estilo_kolb)
    if err3: errores.append(err3)

    if errores:
        return cfg, errores

    cfg2 = GenInput(
        area=area_ok,
        subtema=sub_ok,
        estilo_kolb=kolb_ok,
        longitud_min=cfg.longitud_min,
        longitud_max=cfg.longitud_max,
        max_tokens_item=cfg.max_tokens_item,
        temperatura=cfg.temperatura,
    )
    return cfg2, []

# ===================== Generación =====================
def generar_una(cfg: GenInput) -> ItemOut:
    msgs = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(cfg)},
    ]
    raw = chat_ollama(msgs, max_tokens=cfg.max_tokens_item, temperature=cfg.temperatura, force_json=True)

    if "{" not in raw or "pregunta" not in raw:
        msgs.append({"role": "user", "content":
            "RECUERDA: devuelve SOLO UN OBJETO JSON EXACTO del esquema indicado. "
            "No escribas nada fuera del JSON. No uses 'items'. Incluye la clave 'pregunta'."})
        raw = chat_ollama(msgs, max_tokens=cfg.max_tokens_item, temperature=0.0, force_json=True)

    data = parse_json_min(raw)
    data = coerce_single_item(data)
    data = normalize_keys_es(data)
    ensure_schema(data)

    # Longitud y post-proceso
    # 1) Ajuste de longitud
    data["pregunta"] = pad_to_range(data.get("pregunta",""), cfg.longitud_min, cfg.longitud_max)

    # 2) Saneado de signos '+' en positivos (pregunta y opciones)
    data["pregunta"] = remove_plus_on_positive(data["pregunta"])
    data["opciones"] = clean_options_signs(data.get("opciones", {}))

    # 3) Barajar opciones y arrastrar la correcta
    op = data.get("opciones", {}); rc = data.get("respuesta_correcta", "A")
    op2, rc2 = shuffle_options(op, rc)
    data["opciones"] = op2
    data["respuesta_correcta"] = rc2

    # 4) Coherencia explicacion <-> respuesta_correcta
    data["explicacion"] = fix_explanation_coherence(data.get("explicacion",""), rc2, cfg.area)

    # 5) Forzar coherencia de cabeceras
    data["area"] = cfg.area
    data["subtema"] = cfg.subtema
    data["estilo_kolb"] = cfg.estilo_kolb or "Convergente"

    # 6) meta siempre dict + info mínima útil
    meta = data.get("meta", {})
    if not isinstance(meta, dict): meta = {}
    meta.setdefault("modelo", OLLAMA_MODEL)
    meta.setdefault("seed_randomize", SEED_RANDOMIZE)
    data["meta"] = meta

    return ItemOut(**data)

def fallback_rule_based(cfg: GenInput) -> ItemOut:
    # Fallback simple, largo y coherente
    pregunta = (
        "Un caso práctico presenta datos y condiciones para analizar la relación central del problema. "
        "Evita sesgos de interpretación y valora la evidencia disponible antes de decidir."
    )
    opciones = {"A":"Conclusión coherente con la relación pedida.",
                "B":"Error por focalizar un detalle local.",
                "C":"Generalización sin soporte.",
                "D":"Afirmación no derivada de la evidencia."}
    correcta = "A"
    pregunta = pad_to_range(pregunta, cfg.longitud_min, cfg.longitud_max)
    opciones, correcta = shuffle_options(opciones, correcta)
    explicacion = build_explanation_per_area(cfg.area, correcta)
    return ItemOut(
        area=cfg.area, subtema=cfg.subtema, estilo_kolb=cfg.estilo_kolb or "Convergente",
        pregunta=pregunta, opciones=opciones, respuesta_correcta=correcta,
        explicacion=explicacion, meta={"source":"fallback","modelo":OLLAMA_MODEL}
    )

# ===================== Endpoints =====================
@app.get("/icfes/catalogo")
def icfes_catalogo():
    """Lista las 5 áreas, sus subtemas y estilos Kolb con descripciones."""
    return {"ok": True, "catalogo": catalogo()}

@app.post("/icfes/validar")
def icfes_validar(cfg: GenInput):
    """Verifica SOLO la validez de área/subtema/estilo, sin generar preguntas."""
    cfg2, errores = validar_input(cfg)
    if errores:
        return {"ok": False, "errores": errores, "sugerencias": catalogo()}
    return {"ok": True, "normalizado": cfg2.model_dump(), "mensaje": "Parámetros válidos."}

@app.post("/icfes/generar")
def icfes_generar(cfg: GenInput):
    """Genera 1 ítem. Valida/normaliza parámetros antes de generar."""
    cfg2, errores = validar_input(cfg)
    if errores:
        return {"ok": False, "generadas": 0, "resultados": [], "errores": [{"index": 0, "aviso": e} for e in errores]}
    try:
        item = generar_una(cfg2)
        return {"ok": True, "generadas": 1, "resultados": [item.model_dump()], "errores": []}
    except Exception as e:
        if STRICT_MODE:
            return {"ok": False, "generadas": 0, "resultados": [], "errores": [{"index": 0, "aviso": str(e)}]}
        fb = fallback_rule_based(cfg2)
        return {"ok": True, "generadas": 1, "resultados": [fb.model_dump()], "errores": [{"index": 0, "aviso": str(e)}]}

@app.post("/icfes/generar_pack")
def icfes_generar_pack(cfg: GenInput, cantidad: int = Query(5, ge=1, le=20)):
    """Genera N ítems. Valida/normaliza primero. Evita repeticiones con seed aleatoria."""
    cfg2, errores = validar_input(cfg)
    if errores:
        return {"ok": False, "generadas": 0, "resultados": [], "errores": [{"index": 0, "aviso": e} for e in errores]}
    resultados, errs, vistos = [], [], set()
    for i in range(cantidad):
        try:
            it = generar_una(cfg2).model_dump()
            if it["pregunta"] in vistos:
                time.sleep(0.05)
                it = generar_una(cfg2).model_dump()
            vistos.add(it["pregunta"])
            resultados.append(it)
        except Exception as e:
            errs.append({"index": i, "aviso": str(e)})
            if STRICT_MODE: continue
    while len(resultados) < cantidad and not STRICT_MODE:
        fb = fallback_rule_based(cfg2).model_dump()
        if fb["pregunta"] in vistos: continue
        vistos.add(fb["pregunta"]); resultados.append(fb)
    ok = (len(errs) == 0)
    return {"ok": ok, "generadas": len(resultados), "resultados": resultados, "errores": errs}

@app.post("/debug/raw")
def debug_raw(cfg: GenInput):
    """Muestra salida RAW del modelo (para depurar formato). Valida/normaliza antes."""
    cfg2, errores = validar_input(cfg)
    if errores:
        return {"ok": False, "errores": errores}
    msgs = [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(cfg2)},
    ]
    raw1 = chat_ollama(msgs, max_tokens=cfg2.max_tokens_item, temperature=cfg2.temperatura, force_json=True)
    if "{" not in raw1 or "pregunta" not in raw1:
        msgs.append({"role": "user", "content":
            "RECUERDA: devuelve SOLO UN OBJETO JSON EXACTO del esquema indicado. "
            "No escribas nada fuera del JSON. No uses 'items'. Incluye la clave 'pregunta'."})
        raw2 = chat_ollama(msgs, max_tokens=cfg2.max_tokens_item, temperature=0.0, force_json=True)
        return {"ok": True, "raw1": raw1, "raw2": raw2}
    return {"ok": True, "raw": raw1}
