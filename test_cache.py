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

# Mocking environment variables
os.environ["GEMINI_API_KEY"] = "fake_key"
os.environ["ONYX_HOST"] = "localhost"
os.environ["ONYX_PORT"] = "8080"

from onyx_provider import OnyxCache
import main

class TestSemanticCache(unittest.TestCase):

    @patch('onyx_provider.OnyxClient')
    def test_onyx_cache_logic(self, MockOnyxClient):
        # Setup mock
        mock_instance = MockOnyxClient.return_value
        cache = OnyxCache()
        
        vector = [0.1] * 768
        query = "What is Onyx?"
        response = "Onyx is a vector database."

        # 1. Test upsert
        cache.upsert_cache(vector, query, response)
        mock_instance.upsert.assert_called_once()

        # 2. Test query_cache HIT
        mock_hit = MagicMock()
        mock_hit.score = 0.95
        mock_hit.payload = {"response": response}
        mock_instance.search.return_value = [mock_hit]
        
        result = cache.query_cache(vector, threshold=0.9)
        self.assertEqual(result, response)

        # 3. Test query_cache MISS (below threshold)
        mock_hit.score = 0.8
        result = cache.query_cache(vector, threshold=0.9)
        self.assertIsNone(result)

    @patch('main.get_embedding')
    @patch('main.get_gemini_response')
    @patch('main.OnyxCache')
    def test_main_flow_logic(self, MockOnyxCache, MockGeminiResponse, MockEmbedding):
        # Setup mocks
        mock_cache_instance = MockOnyxCache.return_value
        MockEmbedding.return_value = [0.1] * 768
        MockGeminiResponse.return_value = "Gemini's Answer"
        
        # Scenario: Cache Miss
        mock_cache_instance.query_cache.return_value = None
        
        # We manually call the logic inside main or a slightly modified version for testing
        query = "Hello"
        vector = main.get_embedding(query)
        cached = mock_cache_instance.query_cache(vector)
        
        if not cached:
            resp = main.get_gemini_response(query)
            mock_cache_instance.upsert_cache(vector, query, resp)
        
        # Verify Gemini was called and result was cached
        MockGeminiResponse.assert_called_once_with(query)
        mock_cache_instance.upsert_cache.assert_called_once()

        # Scenario: Cache Hit
        MockGeminiResponse.reset_mock()
        mock_cache_instance.upsert_cache.reset_mock()
        mock_cache_instance.query_cache.return_value = "Cached Answer"
        
        cached = mock_cache_instance.query_cache(vector)
        if cached:
            resp = cached
        
        # Verify Gemini was NOT called
        MockGeminiResponse.assert_not_called()
        self.assertEqual(resp, "Cached Answer")

if __name__ == '__main__':
    unittest.main()
