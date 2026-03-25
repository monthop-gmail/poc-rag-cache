import time
import logging
import google.generativeai as genai
from app.config import (
    GEMINI_API_KEY, EMBEDDING_MODEL, GENERATIVE_MODEL,
    EMBEDDING_BATCH_SIZE, EMBEDDING_RATE_LIMIT_DELAY,
)

logger = logging.getLogger("LLM.Gemini")

genai.configure(api_key=GEMINI_API_KEY)


def get_embedding(text: str) -> list[float]:
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches with rate limiting."""
    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        batch_num = i // EMBEDDING_BATCH_SIZE + 1
        total_batches = -(-len(texts) // EMBEDDING_BATCH_SIZE)
        logger.info(f"Embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type="retrieval_query"
        )
        all_embeddings.extend(result['embedding'])

        if i + EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(EMBEDDING_RATE_LIMIT_DELAY)

    return all_embeddings


def get_gemini_response(prompt: str) -> str:
    model = genai.GenerativeModel(GENERATIVE_MODEL)
    response = model.generate_content(prompt)
    return response.text
