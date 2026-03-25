import os
import time
import logging
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("GeminiUtils")

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

EMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-1.5-flash"
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "20"))
EMBEDDING_RATE_LIMIT_DELAY = float(os.getenv("EMBEDDING_RATE_LIMIT_DELAY", "0.5"))


def get_embedding(text: str):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts in batches with rate limiting.
    Processes EMBEDDING_BATCH_SIZE texts at a time with a delay between batches.
    """
    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        logger.info(f"Embedding batch {i // EMBEDDING_BATCH_SIZE + 1}/{-(-len(texts) // EMBEDDING_BATCH_SIZE)} ({len(batch)} texts)")

        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type="retrieval_query"
        )
        all_embeddings.extend(result['embedding'])

        # Rate limit: wait between batches (skip after last batch)
        if i + EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(EMBEDDING_RATE_LIMIT_DELAY)

    return all_embeddings


def get_gemini_response(prompt: str):
    model = genai.GenerativeModel(GENERATIVE_MODEL)
    response = model.generate_content(prompt)
    return response.text
