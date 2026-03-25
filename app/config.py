import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-1.5-flash"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_RATE_LIMIT_DELAY = float(os.getenv("EMBEDDING_RATE_LIMIT_DELAY", "0.5"))

# --- Redis ---
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_CACHE_EXPIRE = int(os.getenv("REDIS_CACHE_EXPIRE", "3600"))

# --- Onyx ---
ONYX_HOST = os.getenv("ONYX_HOST", "onyx")
ONYX_PORT = int(os.getenv("ONYX_PORT", "8080"))
ONYX_DIMENSION = 768
ONYX_METRIC = "cosine"

# --- Collection names ---
SEMANTIC_CACHE_COLLECTION = "gemini_semantic_cache"
KNOWLEDGE_BASE_COLLECTION = "knowledge_base"

# --- Default thresholds ---
CACHE_THRESHOLD = 0.92
RAG_THRESHOLD = 0.7
RAG_TOP_K = 5

# --- Default chunking ---
DEFAULT_CHILD_SIZE = 500
DEFAULT_PARENT_SIZE = 2000
DEFAULT_OVERLAP = 50
