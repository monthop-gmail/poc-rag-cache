import unittest
from unittest.mock import MagicMock, patch, sys
import os

# Mock external modules before any app imports
sys.modules["onyx_database"] = MagicMock()
mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai
sys.modules["dotenv"] = MagicMock()
sys.modules["pypdf"] = MagicMock()
sys.modules["redis"] = MagicMock()

os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["ONYX_HOST"] = "localhost"
os.environ["ONYX_PORT"] = "8080"

from app.rag.chunking import chunk_text, chunk_document, split_into_sections, ChunkResult
from app.rag.knowledge_base import KnowledgeBase
from app.rag.metrics import _metrics, record, get_stats


# =====================================================================
# Test: Basic chunking
# =====================================================================

class TestChunkText(unittest.TestCase):

    def test_empty_text(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   "), [])

    def test_short_text(self):
        chunks = chunk_text("Hello world.", chunk_size=500)
        self.assertEqual(len(chunks), 1)

    def test_multiple_chunks(self):
        chunks = chunk_text("A" * 1200, chunk_size=500, overlap=50)
        self.assertGreater(len(chunks), 1)

    def test_overlap_exists(self):
        chunks = chunk_text("Word " * 300, chunk_size=500, overlap=50)
        for i in range(len(chunks) - 1):
            tail = chunks[i][-50:]
            self.assertTrue(
                any(tail[j:j+20] in chunks[i + 1] for j in range(len(tail) - 20)),
                "Expected overlap"
            )


# =====================================================================
# Test: Section splitting
# =====================================================================

class TestSplitIntoSections(unittest.TestCase):

    def test_no_headings(self):
        sections = split_into_sections("Plain text without headings.")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0][0], "(untitled)")

    def test_markdown_headings(self):
        sections = split_into_sections("## Policy\nContent.\n## Benefits\nMore.")
        names = [s[0] for s in sections]
        self.assertTrue(any("Policy" in n for n in names))
        self.assertTrue(any("Benefits" in n for n in names))

    def test_thai_headings(self):
        text = "บทนำ\nเนื้อหา\nหมวด 1 การลา\nรายละเอียด\nหมวด 2 สวัสดิการ\nเพิ่มเติม"
        sections = split_into_sections(text)
        names = [s[0] for s in sections]
        self.assertTrue(any("หมวด 1" in n for n in names))
        self.assertTrue(any("หมวด 2" in n for n in names))

    def test_intro_before_heading(self):
        sections = split_into_sections("Intro text.\n## Chapter 1\nContent.")
        self.assertEqual(sections[0][0], "(intro)")

    def test_empty(self):
        self.assertEqual(split_into_sections(""), [])


# =====================================================================
# Test: Parent-child chunking
# =====================================================================

class TestChunkDocument(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(chunk_document(""), [])

    def test_short_doc(self):
        results = chunk_document("Short doc.")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].child_text, "Short doc.")
        self.assertEqual(results[0].parent_text, "Short doc.")

    def test_parent_larger_than_child(self):
        text = f"## Policy\n{'Some sentence. ' * 100}"
        results = chunk_document(text, child_size=200, parent_size=1000, overlap=30)
        self.assertGreater(len(results), 1)
        for r in results:
            self.assertGreaterEqual(len(r.parent_text), len(r.child_text))

    def test_section_name_preserved(self):
        text = "## การลา\nลาป่วยได้ 30 วัน\n## สวัสดิการ\nค่ารักษา OPD"
        results = chunk_document(text, child_size=500, parent_size=2000)
        sections = set(r.section for r in results)
        self.assertTrue(any("การลา" in s for s in sections))
        self.assertTrue(any("สวัสดิการ" in s for s in sections))

    def test_sequential_indexes(self):
        results = chunk_document("Word. " * 500, child_size=100, parent_size=500, overlap=20)
        indexes = [r.chunk_index for r in results]
        self.assertEqual(indexes, list(range(len(results))))

    def test_condition_stays_in_parent(self):
        text = "## การลา\nพนักงานลาป่วยได้ 30 วันต่อปี แต่ถ้าทำงานไม่ถึง 1 ปี ลาได้ไม่เกิน 15 วัน"
        results = chunk_document(text, child_size=200, parent_size=2000)
        for r in results:
            if "30 วัน" in r.parent_text:
                self.assertIn("15 วัน", r.parent_text)


# =====================================================================
# Test: KnowledgeBase
# =====================================================================

class TestKnowledgeBase(unittest.TestCase):

    @patch('app.rag.knowledge_base.OnyxClient')
    def test_ingest_with_parent_child(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        chunks = [
            ChunkResult(child_text="child 1", parent_text="parent A", section="## Intro", chunk_index=0),
            ChunkResult(child_text="child 2", parent_text="parent A", section="## Intro", chunk_index=1),
        ]
        kb.ingest_chunks(chunks, [[0.1] * 768, [0.2] * 768], source_filename="test.pdf")

        points = mock_instance.upsert.call_args[1]["points"]
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0]["payload"]["child_text"], "child 1")
        self.assertEqual(points[0]["payload"]["parent_text"], "parent A")

    @patch('app.rag.knowledge_base.OnyxClient')
    def test_search_returns_parent(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        hit = MagicMock()
        hit.score = 0.85
        hit.payload = {
            "child_text": "ลาป่วยได้ 30 วัน",
            "parent_text": "ลาป่วยได้ 30 วัน แต่ถ้าไม่ถึง 1 ปี ลาได้ 15 วัน",
            "section": "หมวด 1", "source_filename": "hr.pdf"
        }
        mock_instance.search.return_value = [hit]

        results = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(results), 1)
        self.assertIn("15 วัน", results[0]["text"])

    @patch('app.rag.knowledge_base.OnyxClient')
    def test_deduplicates_same_parent(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        parent = "Shared parent text"
        hit1 = MagicMock(score=0.9, payload={"child_text": "a", "parent_text": parent, "section": "S", "source_filename": "f"})
        hit2 = MagicMock(score=0.88, payload={"child_text": "b", "parent_text": parent, "section": "S", "source_filename": "f"})
        mock_instance.search.return_value = [hit1, hit2]

        results = kb.search([0.1] * 768, threshold=0.7)
        self.assertEqual(len(results), 1)

    @patch('app.rag.knowledge_base.OnyxClient')
    def test_below_threshold(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()

        hit = MagicMock(score=0.5, payload={"child_text": "x", "parent_text": "x", "section": "", "source_filename": "f"})
        mock_instance.search.return_value = [hit]

        self.assertEqual(kb.search([0.1] * 768, threshold=0.7), [])

    @patch('app.rag.knowledge_base.OnyxClient')
    def test_delete_by_source(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        kb = KnowledgeBase()
        kb.delete_by_source("old.pdf")
        mock_instance.delete.assert_called_once()


# =====================================================================
# Test: Metrics
# =====================================================================

class TestMetrics(unittest.TestCase):

    def test_record_and_get_stats(self):
        # Reset metrics
        for k in _metrics:
            _metrics[k] = 0 if isinstance(_metrics[k], int) else 0.0

        record("total_queries")
        record("l1_hits")
        record("total_latency_ms", 5.0)

        stats = get_stats()
        self.assertEqual(stats.total_queries, 1)
        self.assertEqual(stats.l1_hits, 1)
        self.assertEqual(stats.cache_hit_rate, 100.0)
        self.assertEqual(stats.avg_latency_ms, 5.0)


# =====================================================================
# Test: Batch Embedding
# =====================================================================

class TestBatchEmbedding(unittest.TestCase):

    def test_batch_calls_genai(self):
        from app.llm.gemini import get_embeddings_batch

        mock_genai.embed_content.reset_mock()
        mock_genai.embed_content.return_value = {'embedding': [[0.1] * 768] * 3}

        results = get_embeddings_batch(["a", "b", "c"])
        self.assertEqual(len(results), 3)

    def test_batch_multiple_batches(self):
        from app.llm.gemini import get_embeddings_batch, EMBEDDING_BATCH_SIZE

        n = EMBEDDING_BATCH_SIZE + 5
        texts = [f"text {i}" for i in range(n)]

        mock_genai.embed_content.reset_mock()
        mock_genai.embed_content.side_effect = lambda **kwargs: {'embedding': [[0.1] * 768] * len(kwargs.get('content', []))}

        results = get_embeddings_batch(texts)
        self.assertEqual(len(results), n)
        self.assertEqual(mock_genai.embed_content.call_count, 2)

        mock_genai.embed_content.side_effect = None

    def test_batch_empty(self):
        from app.llm.gemini import get_embeddings_batch
        self.assertEqual(get_embeddings_batch([]), [])


# =====================================================================
# Test: Configurable Chunking
# =====================================================================

class TestConfigurableChunking(unittest.TestCase):

    def test_smaller_child_more_chunks(self):
        text = "Word. " * 300
        small = chunk_document(text, child_size=100, parent_size=2000, overlap=20)
        large = chunk_document(text, child_size=500, parent_size=2000, overlap=20)
        self.assertGreater(len(small), len(large))

    def test_smaller_parent_more_parents(self):
        text = "Sentence here. " * 500
        small_p = set(r.parent_text for r in chunk_document(text, child_size=200, parent_size=500, overlap=30))
        large_p = set(r.parent_text for r in chunk_document(text, child_size=200, parent_size=3000, overlap=30))
        self.assertGreaterEqual(len(small_p), len(large_p))


if __name__ == '__main__':
    unittest.main()
