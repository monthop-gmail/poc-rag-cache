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

mock_redis = MagicMock()
sys.modules["redis"] = mock_redis

os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["ONYX_HOST"] = "localhost"
os.environ["ONYX_PORT"] = "8080"

from document_processor import chunk_text, chunk_document, split_into_sections, ChunkResult
from rag_provider import KnowledgeBase


# =====================================================================
# Test: Basic chunking (backward compat)
# =====================================================================

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
        for i in range(len(chunks) - 1):
            tail = chunks[i][-50:]
            self.assertTrue(
                any(tail[j:j+20] in chunks[i + 1] for j in range(len(tail) - 20)),
                "Expected overlap between consecutive chunks"
            )

    def test_all_text_covered(self):
        text = "Hello. " * 200
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        combined = " ".join(chunks)
        for word in text.split():
            self.assertIn(word, combined)


# =====================================================================
# Test: Section splitting
# =====================================================================

class TestSplitIntoSections(unittest.TestCase):

    def test_no_headings(self):
        text = "Just a plain paragraph with no headings."
        sections = split_into_sections(text)
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "(untitled)")

    def test_markdown_headings(self):
        text = "## Policy\nSome content here.\n## Benefits\nMore content."
        sections = split_into_sections(text)
        names = [s[0] for s in sections]
        self.assertTrue(any("Policy" in n for n in names))
        self.assertTrue(any("Benefits" in n for n in names))

    def test_thai_headings(self):
        text = "บทนำ\nข้อความเบื้องต้น\nหมวด 1 การลา\nรายละเอียดการลา\nหมวด 2 สวัสดิการ\nรายละเอียดสวัสดิการ"
        sections = split_into_sections(text)
        names = [s[0] for s in sections]
        self.assertTrue(any("หมวด 1" in n for n in names))
        self.assertTrue(any("หมวด 2" in n for n in names))

    def test_numbered_headings(self):
        text = "1. Overview\nSome text.\n2. Details\nMore text."
        sections = split_into_sections(text)
        self.assertGreater(len(sections), 1)

    def test_intro_before_first_heading(self):
        text = "This is an introduction.\n## Chapter 1\nContent here."
        sections = split_into_sections(text)
        self.assertEqual(sections[0][0], "(intro)")

    def test_empty_text(self):
        self.assertEqual(split_into_sections(""), [])


# =====================================================================
# Test: Parent-child chunking (chunk_document)
# =====================================================================

class TestChunkDocument(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(chunk_document(""), [])

    def test_short_document(self):
        text = "Short doc."
        results = chunk_document(text)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ChunkResult)
        self.assertEqual(results[0].child_text, "Short doc.")
        self.assertEqual(results[0].parent_text, "Short doc.")

    def test_parent_is_larger_than_child(self):
        # Create text with a heading and long body
        body = "Some sentence. " * 100  # ~1500 chars
        text = f"## Policy\n{body}"
        results = chunk_document(text, child_size=200, parent_size=1000, overlap=30)

        self.assertGreater(len(results), 1)
        # Parent should be >= child
        for r in results:
            self.assertGreaterEqual(len(r.parent_text), len(r.child_text))

    def test_section_name_preserved(self):
        text = "## การลา\nพนักงานลาป่วยได้ 30 วัน\n## สวัสดิการ\nค่ารักษา OPD 2000 บาท"
        results = chunk_document(text, child_size=500, parent_size=2000)
        sections = set(r.section for r in results)
        self.assertTrue(any("การลา" in s for s in sections))
        self.assertTrue(any("สวัสดิการ" in s for s in sections))

    def test_child_indexes_are_sequential(self):
        text = "Word. " * 500
        results = chunk_document(text, child_size=100, parent_size=500, overlap=20)
        indexes = [r.chunk_index for r in results]
        self.assertEqual(indexes, list(range(len(results))))

    def test_conditional_text_stays_in_one_parent(self):
        """Key test: a condition and its clause should be in the same parent."""
        text = (
            "## การลา\n"
            "พนักงานลาป่วยได้ 30 วันต่อปี แต่ถ้าทำงานไม่ถึง 1 ปี ลาได้ไม่เกิน 15 วัน"
        )
        results = chunk_document(text, child_size=200, parent_size=2000)
        # With parent_size=2000, the whole section fits in one parent
        parents = set(r.parent_text for r in results)
        for parent in parents:
            if "30 วัน" in parent:
                self.assertIn("15 วัน", parent,
                    "Condition '15 วัน' should be in the same parent as '30 วัน'")


# =====================================================================
# Test: KnowledgeBase with parent-child
# =====================================================================

class TestKnowledgeBase(unittest.TestCase):

    @patch('rag_provider.OnyxClient')
    def test_ingest_chunks_with_parent_child(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        chunk_results = [
            ChunkResult(child_text="child 1", parent_text="parent A full text", section="## Intro", chunk_index=0),
            ChunkResult(child_text="child 2", parent_text="parent A full text", section="## Intro", chunk_index=1),
        ]
        vectors = [[0.1] * 768, [0.2] * 768]

        kb.ingest_chunks(chunk_results, vectors, source_filename="test.pdf")
        mock_instance.upsert.assert_called_once()

        call_args = mock_instance.upsert.call_args
        points = call_args[1]["points"]
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0]["id"], "test.pdf_0")
        self.assertEqual(points[0]["payload"]["child_text"], "child 1")
        self.assertEqual(points[0]["payload"]["parent_text"], "parent A full text")
        self.assertEqual(points[0]["payload"]["section"], "## Intro")

    @patch('rag_provider.OnyxClient')
    def test_search_returns_parent_text(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.payload = {
            "child_text": "ลาป่วยได้ 30 วัน",
            "parent_text": "พนักงานลาป่วยได้ 30 วันต่อปี แต่ถ้าทำงานไม่ถึง 1 ปี ลาได้ไม่เกิน 15 วัน",
            "section": "หมวด 1 การลา",
            "source_filename": "hr.pdf"
        }
        mock_instance.search.return_value = [mock_hit]

        results = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(results), 1)
        # LLM gets parent text (full context), not just child
        self.assertIn("15 วัน", results[0]["text"])
        self.assertIn("30 วัน", results[0]["text"])
        self.assertEqual(results[0]["section"], "หมวด 1 การลา")

    @patch('rag_provider.OnyxClient')
    def test_search_deduplicates_same_parent(self, MockOnyxClient):
        """Two children from same parent should return parent only once."""
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        parent_text = "Full parent context about leave policy"
        hit1 = MagicMock()
        hit1.score = 0.90
        hit1.payload = {"child_text": "child A", "parent_text": parent_text, "section": "S1", "source_filename": "hr.pdf"}

        hit2 = MagicMock()
        hit2.score = 0.88
        hit2.payload = {"child_text": "child B", "parent_text": parent_text, "section": "S1", "source_filename": "hr.pdf"}

        mock_instance.search.return_value = [hit1, hit2]

        results = kb.search([0.1] * 768, threshold=0.7)
        # Should be deduplicated to 1 result
        self.assertEqual(len(results), 1)

    @patch('rag_provider.OnyxClient')
    def test_search_below_threshold(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        mock_hit = MagicMock()
        mock_hit.score = 0.5
        mock_hit.payload = {"child_text": "x", "parent_text": "x", "section": "", "source_filename": "doc.pdf"}
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


# =====================================================================
# Test: RAG Flow (end-to-end logic)
# =====================================================================

class TestRAGFlow(unittest.TestCase):

    @patch('rag_provider.OnyxClient')
    def test_rag_returns_parent_context_to_llm(self, MockOnyxClient):
        """RAG should send parent (full context) to LLM, not just child fragment."""
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        mock_hit = MagicMock()
        mock_hit.score = 0.85
        mock_hit.payload = {
            "child_text": "ลาป่วยได้ 30 วัน",
            "parent_text": "พนักงานลาป่วยได้ 30 วันต่อปี แต่ถ้าทำงานไม่ถึง 1 ปี ลาได้ไม่เกิน 15 วัน",
            "section": "หมวด 1 การลา",
            "source_filename": "hr.pdf"
        }
        mock_instance.search.return_value = [mock_hit]

        hits = kb.search([0.1] * 768, threshold=0.7)

        # Build prompt like rag_routes does
        context_parts = []
        for h in hits:
            section = h.get("section", "")
            source = h["source"]
            header = f"[{source} | {section}]" if section else f"[{source}]"
            context_parts.append(f"{header}:\n{h['text']}")
        context = "\n\n---\n\n".join(context_parts)
        prompt = f"Context:\n{context}\n\nQuestion: วันลาป่วยมีกี่วัน\nAnswer:"

        # Parent text should be in the prompt (includes condition)
        self.assertIn("30 วัน", prompt)
        self.assertIn("15 วัน", prompt)
        self.assertIn("หมวด 1 การลา", prompt)

    @patch('rag_provider.OnyxClient')
    def test_cache_miss_no_rag_results(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        mock_instance.search.return_value = []
        hits = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(hits), 0)


# =====================================================================
# Test: Cache Invalidation
# =====================================================================

class TestCacheInvalidation(unittest.TestCase):

    @patch('onyx_provider.OnyxClient')
    def test_onyx_cache_clear_all(self, MockOnyxClient):
        """OnyxCache.clear_all should delete and recreate the collection."""
        from onyx_provider import OnyxCache
        mock_instance = MockOnyxClient.return_value
        cache = OnyxCache()

        cache.clear_all()
        mock_instance.delete_collection.assert_called_once_with(name="gemini_semantic_cache")
        # Should recreate after delete
        self.assertEqual(mock_instance.create_collection.call_count, 2)  # init + clear_all


# =====================================================================
# Test: Metrics
# =====================================================================

class TestMetrics(unittest.TestCase):

    def test_metrics_dict_structure(self):
        from rag_routes import _metrics
        expected_keys = {
            "total_queries", "l1_hits", "l2_hits", "rag_hits",
            "cache_misses", "total_latency_ms",
            "total_documents", "total_chunks_ingested"
        }
        self.assertEqual(set(_metrics.keys()), expected_keys)

    def test_metrics_initial_values_are_zero(self):
        from rag_routes import _metrics
        for key, value in _metrics.items():
            self.assertEqual(value, 0 if isinstance(value, int) else 0.0,
                f"Metric '{key}' should start at 0")


# =====================================================================
# Test: Batch Embedding
# =====================================================================

class TestBatchEmbedding(unittest.TestCase):

    def test_get_embeddings_batch_calls_genai(self):
        from gemini_utils import get_embeddings_batch
        texts = ["hello", "world", "test"]

        # Mock genai.embed_content to return fake embeddings
        mock_genai.embed_content.return_value = {
            'embedding': [[0.1] * 768, [0.2] * 768, [0.3] * 768]
        }

        results = get_embeddings_batch(texts)
        self.assertEqual(len(results), 3)
        mock_genai.embed_content.assert_called()

    def test_get_embeddings_batch_handles_multiple_batches(self):
        from gemini_utils import get_embeddings_batch, EMBEDDING_BATCH_SIZE

        # Create more texts than one batch
        n = EMBEDDING_BATCH_SIZE + 5
        texts = [f"text {i}" for i in range(n)]

        # Mock to return correct number of embeddings per batch
        def fake_embed(model, content, task_type):
            return {'embedding': [[0.1] * 768] * len(content)}

        mock_genai.embed_content.reset_mock()
        mock_genai.embed_content.side_effect = fake_embed

        results = get_embeddings_batch(texts)
        self.assertEqual(len(results), n)
        # Should be called twice (one full batch + one partial)
        self.assertEqual(mock_genai.embed_content.call_count, 2)

        mock_genai.embed_content.side_effect = None  # reset

    def test_get_embeddings_batch_empty(self):
        from gemini_utils import get_embeddings_batch
        results = get_embeddings_batch([])
        self.assertEqual(results, [])


# =====================================================================
# Test: Configurable Chunking
# =====================================================================

class TestConfigurableChunking(unittest.TestCase):

    def test_custom_child_size(self):
        text = "Word. " * 300  # ~1800 chars
        results_small = chunk_document(text, child_size=100, parent_size=2000, overlap=20)
        results_large = chunk_document(text, child_size=500, parent_size=2000, overlap=20)
        # Smaller child_size = more chunks
        self.assertGreater(len(results_small), len(results_large))

    def test_custom_parent_size(self):
        text = "Sentence here. " * 500  # ~7500 chars
        results_small_parent = chunk_document(text, child_size=200, parent_size=500, overlap=30)
        results_large_parent = chunk_document(text, child_size=200, parent_size=3000, overlap=30)
        # Smaller parent = more distinct parents
        parents_small = set(r.parent_text for r in results_small_parent)
        parents_large = set(r.parent_text for r in results_large_parent)
        self.assertGreaterEqual(len(parents_small), len(parents_large))


if __name__ == '__main__':
    unittest.main()
