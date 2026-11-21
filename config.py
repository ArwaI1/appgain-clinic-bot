import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "clinic_faq.json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file.")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gemini-3.5-flash-lite")
FALLBACK_GENERATION_MODEL = os.getenv("FALLBACK_GENERATION_MODEL", "gemini-3.8-flash")
EMBEDDING_DIMENSIONS = 768
SIMILARITY_THRESHOLD = 0.25
TOP_K = 4
MAX_HISTORY_TURNS = 4
REQUEST_TIMEOUT_MS = 40000