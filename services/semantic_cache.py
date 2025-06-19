# Advanced caching with semantic similarity
# services/semantic_cache.py - Semantic Caching Implementation
import numpy as np
import pickle
import gzip
from typing import Optional, Tuple, List, Any
import hashlib
from datetime import datetime, timedelta
import asyncio
import logging


class SemanticCache:
    def __init__(self, sentence_transformer_model):
        self.model = sentence_transformer_model
        self.cache_store = {}
        self.embeddings_store = {}
        self.similarity_threshold = 0.85
        self.max_cache_size = 5000

    async def get_semantic_match(
        self, query: str, max_results: int = 3
    ) -> Optional[Tuple[str, float, Any]]:
        """Find semantically similar cached responses"""

        if not self.embeddings_store:
            return None

        try:
            # Generate embedding for query
            query_embedding = self.model.encode([query])

            # Calculate similarities with cached queries
            best_match = None
            best_similarity = 0.0

            for cache_key, stored_embedding in self.embeddings_store.items():
                similarity = np.dot(query_embedding[0], stored_embedding) / (
                    np.linalg.norm(query_embedding[0])
                    * np.linalg.norm(stored_embedding)
                )

                if (
                    similarity > best_similarity
                    and similarity > self.similarity_threshold
                ):
                    best_similarity = similarity
                    best_match = cache_key

            if best_match and best_match in self.cache_store:
                cached_data = self.cache_store[best_match]

                # Check if not expired
                if datetime.now() < cached_data["expires_at"]:
                    return best_match, best_similarity, cached_data["response"]
                else:
                    # Remove expired entry
                    del self.cache_store[best_match]
                    del self.embeddings_store[best_match]

            return None

        except Exception as e:
            logging.error(f"Error in semantic cache lookup: {str(e)}")
            return None

    async def store_with_semantics(self, query: str, response: Any, ttl: int = 7200):
        """Store response with semantic embedding"""

        try:
            # Generate cache key
            cache_key = hashlib.sha256(query.encode()).hexdigest()

            # Generate embedding
            embedding = self.model.encode([query])[0]

            # Store embedding and response
            self.embeddings_store[cache_key] = embedding

            expires_at = datetime.now() + timedelta(seconds=ttl)

            # Compress response if enabled
            compressed_response = self.compress_data(response)

            self.cache_store[cache_key] = {
                "query": query,
                "response": compressed_response,
                "created_at": datetime.now(),
                "expires_at": expires_at,
                "access_count": 0,
            }

            # Cleanup if cache is too large
            if len(self.cache_store) > self.max_cache_size:
                await self.cleanup_cache()

        except Exception as e:
            logging.error(f"Error storing semantic cache: {str(e)}")

    def compress_data(self, data: Any) -> bytes:
        """Compress data for storage efficiency"""
        try:
            pickled_data = pickle.dumps(data)
            return gzip.compress(pickled_data)
        except:
            return pickle.dumps(data)  # Fallback without compression

    def decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress cached data"""
        try:
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        except:
            return pickle.loads(compressed_data)  # Fallback

    async def cleanup_cache(self):
        """Remove old and least accessed entries"""

        # Sort by access count and age
        sorted_entries = sorted(
            self.cache_store.items(),
            key=lambda x: (x[1]["access_count"], x[1]["created_at"]),
        )

        # Remove bottom 20%
        entries_to_remove = len(sorted_entries) // 5

        for cache_key, _ in sorted_entries[:entries_to_remove]:
            if cache_key in self.cache_store:
                del self.cache_store[cache_key]
            if cache_key in self.embeddings_store:
                del self.embeddings_store[cache_key]

        logging.info(f"Cleaned up {entries_to_remove} semantic cache entries")
