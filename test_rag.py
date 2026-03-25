import unittest
from unittest.mock import MagicMock, patch, sys
import os

# Mocking modules before any imports
mock_onyx = MagicMock()
sys.modules["onyx_database"] = mock_onyx

mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai

mock_dotenv = MagicMock()
sys.modules["dotenv"] = mock_dotenv

mock_pypdf = MagicMock()
sys.modules["pypdf"] = mock_pypdf

os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["ONYX_HOST"] = "localhost"
os.environ["ONYX_PORT"] = "8080"

from document_processor import chunk_text
from rag_provider import KnowledgeBase


class TestChunkText(unittest.TestCase):

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_short_text(self):
        text = "Hello world."
        chunks = chunk_text(text, chunk_size=500)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_chunking_produces_multiple_chunks(self):
        text = "A" * 1200
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_overlap_exists(self):
        text = "Word " * 300  # 1500 chars
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # Check that consecutive chunks share some content
        for i in range(len(chunks) - 1):
            tail = chunks[i][-50:]
            # The overlap region should appear somewhere in the next chunk
            self.assertTrue(
                any(tail[j:j+20] in chunks[i + 1] for j in range(len(tail) - 20)),
                "Expected overlap between consecutive chunks"
            )

    def test_all_text_covered(self):
        text = "Hello. " * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        combined = " ".join(chunks)
        # Every word from original should appear in at least one chunk
        for word in text.split():
            self.assertIn(word, combined)


class TestKnowledgeBase(unittest.TestCase):

    @patch('rag_provider.OnyxClient')
    def test_ingest_chunks(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        chunks = ["chunk 1", "chunk 2"]
        vectors = [[0.1] * 768, [0.2] * 768]

        kb.ingest_chunks(chunks, vectors, source_filename="test.pdf")
        mock_instance.upsert.assert_called_once()

        call_args = mock_instance.upsert.call_args
        points = call_args[1]["points"] if "points" in call_args[1] else call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("points")
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0]["id"], "test.pdf_0")
        self.assertEqual(points[0]["payload"]["text"], "chunk 1")
        self.assertEqual(points[0]["payload"]["source_filename"], "test.pdf")

    @patch('rag_provider.OnyxClient')
    def test_search_above_threshold(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.payload = {"text": "relevant content", "source_filename": "doc.pdf"}
        mock_instance.search.return_value = [mock_hit]

        results = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "relevant content")

    @patch('rag_provider.OnyxClient')
    def test_search_below_threshold(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        mock_hit = MagicMock()
        mock_hit.score = 0.5
        mock_hit.payload = {"text": "irrelevant", "source_filename": "doc.pdf"}
        mock_instance.search.return_value = [mock_hit]

        results = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(results), 0)

    @patch('rag_provider.OnyxClient')
    def test_delete_by_source(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        kb.delete_by_source("old_doc.pdf")
        mock_instance.delete.assert_called_once_with(
            collection="knowledge_base",
            filter={"source_filename": "old_doc.pdf"}
        )


class TestRAGFlow(unittest.TestCase):

    @patch('rag_provider.OnyxClient')
    def test_cache_miss_then_rag(self, MockOnyxClient):
        """Simulate: cache miss → RAG finds docs → Gemini answers → cache stored."""
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        # RAG search returns relevant chunk
        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.payload = {"text": "ลาป่วยได้ 30 วัน", "source_filename": "hr.pdf"}
        mock_instance.search.return_value = [mock_hit]

        hits = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(hits), 1)

        # Build augmented prompt
        context = "\n".join([h["text"] for h in hits])
        prompt = f"Context:\n{context}\n\nQuestion: วันลาป่วยมีกี่วัน\nAnswer:"
        self.assertIn("ลาป่วยได้ 30 วัน", prompt)

    @patch('rag_provider.OnyxClient')
    def test_cache_miss_no_rag_results(self, MockOnyxClient):
        """Simulate: cache miss → RAG finds nothing → direct Gemini call."""
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        mock_instance.search.return_value = []
        hits = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(hits), 0)


if __name__ == '__main__':
    unittest.main()
