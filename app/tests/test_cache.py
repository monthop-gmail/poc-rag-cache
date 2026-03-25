import unittest
from unittest.mock import MagicMock, patch, sys
import os

# Mock external modules
sys.modules["onyx_database"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["dotenv"] = MagicMock()
sys.modules["redis"] = MagicMock()

os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["ONYX_HOST"] = "localhost"
os.environ["ONYX_PORT"] = "8080"

from app.cache.semantic_cache import SemanticCache


class TestSemanticCache(unittest.TestCase):

    @patch('app.cache.semantic_cache.OnyxClient')
    def test_upsert_and_query_hit(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        cache = SemanticCache()

        vector = [0.1] * 768
        query = "What is Onyx?"
        response = "Onyx is a vector database."

        # Test upsert
        cache.upsert(vector, query, response)
        mock_instance.upsert.assert_called_once()

        # Test query hit
        mock_hit = MagicMock()
        mock_hit.score = 0.95
        mock_hit.payload = {"response": response}
        mock_instance.search.return_value = [mock_hit]

        result = cache.query(vector, threshold=0.9)
        self.assertEqual(result, response)

    @patch('app.cache.semantic_cache.OnyxClient')
    def test_query_miss(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        cache = SemanticCache()

        mock_hit = MagicMock()
        mock_hit.score = 0.8
        mock_hit.payload = {"response": "something"}
        mock_instance.search.return_value = [mock_hit]

        result = cache.query([0.1] * 768, threshold=0.9)
        self.assertIsNone(result)

    @patch('app.cache.semantic_cache.OnyxClient')
    def test_clear_all(self, MockOnyxClient):
        mock_instance = MockOnyxClient.return_value
        cache = SemanticCache()

        cache.clear_all()
        mock_instance.delete_collection.assert_called_once()
        self.assertEqual(mock_instance.create_collection.call_count, 2)  # init + clear


if __name__ == '__main__':
    unittest.main()
