# services/enhanced_router.py - Enhanced Routing with Semantic Classification (Fixed)
import re
import asyncio
import logging
from typing import Optional, Dict, Any
from services.router import LLMRouter as BaseRouter
from services.semantic_classifier import SemanticIntentClassifier
from models.requests import ChatCompletionRequest


class EnhancedLLMRouter(BaseRouter):
    def __init__(self, ollama_client=None):
        super().__init__(ollama_client)
        self.semantic_classifier = None
        self.classification_cache = {}
        self.cache_max_size = 1000  # Memory limit for cache

    async def initialize(self):
        """Initialize enhanced router with semantic classifier"""
        await super().initialize()

        # Initialize semantic classifier with error handling
        try:
            self.semantic_classifier = SemanticIntentClassifier()
            await self.semantic_classifier.initialize()
            logging.info("Semantic classifier initialized successfully")
        except Exception as e:
            logging.warning(
                f"Failed to initialize semantic classifier: {str(e)}")
            logging.info("Falling back to rule-based classification only")
            self.semantic_classifier = None

        logging.info("Enhanced LLM Router initialized")

    def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
        """Enhanced intent classification with semantic fallback"""

        if explicit_intent:
            return explicit_intent

        # Check cache first (with size limit)
        cache_key = hash(text[:200])  # Use first 200 chars for cache key
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]

        # Try rule-based classification first (fast)
        rule_based_intent = self._rule_based_classification(text)
        if rule_based_intent != 'unknown':
            self._cache_classification(cache_key, rule_based_intent)
            return rule_based_intent

        # Fallback to semantic classification if available
        if self.semantic_classifier:
            try:
                # Run semantic classification in background and return heuristic for now
                asyncio.create_task(
                    self._semantic_classify_async(text, cache_key))
                # Return heuristic classification immediately
                heuristic_intent = self._heuristic_classification(text)
                self._cache_classification(cache_key, heuristic_intent)
                return heuristic_intent
            except Exception as e:
                logging.warning(f"Semantic classification failed: {str(e)}")

        # Final fallback to heuristic
        heuristic_intent = self._heuristic_classification(text)
        self._cache_classification(cache_key, heuristic_intent)
        return heuristic_intent

    def _rule_based_classification(self, text: str) -> str:
        """Rule-based classification using patterns"""
        text_lower = text.lower()

        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text_lower):
                return intent

        return 'unknown'

    def _heuristic_classification(self, text: str) -> str:
        """Heuristic-based classification as fallback"""

        text_lower = text.lower()
        word_count = len(text.split())

        # Length-based heuristics
        if word_count < 10:
            return 'factual'  # Short queries are usually factual
        elif word_count > 100:
            if any(word in text_lower for word in ['analyze', 'review', 'evaluate']):
                return 'analysis'
            else:
                return 'creative'

        # Content-based heuristics
        if any(word in text_lower for word in ['resume', 'cv', 'experience']):
            return 'resume'
        elif any(word in text_lower for word in ['interview', 'job', 'career']):
            return 'interview'
        elif any(word in text_lower for word in ['write', 'create', 'compose']):
            return 'creative'
        elif any(word in text_lower for word in ['code', 'function', 'algorithm']):
            return 'coding'
        elif any(word in text_lower for word in ['calculate', 'solve', 'math']):
            return 'math'

        return 'factual'  # Default fallback

    async def _semantic_classify_async(self, text: str, cache_key: int):
        """Async semantic classification that updates cache"""

        try:
            if not self.semantic_classifier:
                return

            intent, confidence = await self.semantic_classifier.classify_intent(text)

            # Only use semantic result if confidence is high enough
            if confidence > getattr(self.semantic_classifier, 'confidence_threshold', 0.7):
                self._cache_classification(cache_key, intent)
                logging.debug(
                    f"Semantic classification updated cache: {intent} (confidence: {confidence:.3f})")

        except Exception as e:
            logging.error(f"Error in async semantic classification: {str(e)}")

    def _cache_classification(self, cache_key: int, intent: str):
        """Cache classification result with size limit"""

        # Implement LRU-like behavior by clearing old entries when cache is full
        if len(self.classification_cache) >= self.cache_max_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self.classification_cache) // 5
            keys_to_remove = list(self.classification_cache.keys())[
                :items_to_remove]
            for key in keys_to_remove:
                del self.classification_cache[key]

        self.classification_cache[cache_key] = intent

    async def route_request(self, request: ChatCompletionRequest) -> str:
        """Enhanced routing with semantic classification"""

        # Use explicit model if valid
        if request.model in self.available_models:
            return request.model

        # Extract text content for classification
        text_content = self._extract_text_content(
            [{"role": msg.role, "content": msg.content}
                for msg in request.messages]
        )

        # Classify intent (this is now synchronous with async background updates)
        intent = self.classify_intent(text_content, request.intent)

        # Select model based on intent and other factors
        selected_model = self._select_model(intent, text_content, request)

        logging.info(
            f"Enhanced routing: intent={intent}, selected_model={selected_model}")
        return selected_model

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""

        stats = {
            "cache_size": len(self.classification_cache),
            "cache_max_size": self.cache_max_size,
            "semantic_classifier_available": self.semantic_classifier is not None
        }

        if self.semantic_classifier:
            stats.update(self.semantic_classifier.get_classification_stats())

        return stats

    async def cleanup(self):
        """Cleanup resources"""
        self.classification_cache.clear()
        await super().cleanup()
