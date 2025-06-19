# services/enhanced_router.py - Fixed Enhanced Routing with Semantic Classification
import re
import asyncio
import logging
import weakref
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

# Safe imports with fallbacks
try:
    from services.router import LLMRouter as BaseRouter
except ImportError:
    try:
        from services.router import LLMRouter as BaseRouter
    except ImportError:
        logging.error("Could not import base LLMRouter - enhanced router disabled")
        BaseRouter = None

try:
    from services.semantic_classifier import SemanticIntentClassifier
except ImportError:
    logging.warning(
        "SemanticIntentClassifier not available - using rule-based classification only"
    )
    SemanticIntentClassifier = None

try:
    from models.requests import ChatCompletionRequest
except ImportError:
    # Fallback for basic request handling
    ChatCompletionRequest = None


class EnhancedLLMRouter:
    """Enhanced LLM Router with semantic classification and intelligent caching"""

    def __init__(self, ollama_client=None):
        # Initialize base router functionality
        if BaseRouter:
            self.base_router = BaseRouter(ollama_client)
        else:
            self.base_router = None
            logging.error(
                "Base router not available - enhanced router will have limited functionality"
            )

        self.ollama_client = ollama_client
        self.semantic_classifier = None
        self.classification_cache = {}
        self.cache_max_size = 1000
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "semantic_calls": 0,
            "rule_based_calls": 0,
        }

        # Task management for async semantic classification
        self.background_tasks = set()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = datetime.now()

        # Enhanced intent patterns (more comprehensive)
        self.intent_patterns = {
            "math": r"\b(?:calculate|compute|solve|equation|math|arithmetic|what.*is.*[\d\+\-\*/]|[\d\+\-\*/].*equals?)\b",
            "factual": r"\b(?:what is|who is|when did|where is|define|explain|fact|tell me about)\b",
            "creative": r"\b(?:write|create|compose|story|poem|creative|imagine|generate|draft)\b",
            "coding": r"\b(?:code|function|algorithm|debug|program|script|python|javascript|api|sql)\b",
            "resume": r"\b(?:resume|cv|experience|skills|qualifications|work history|career summary)\b",
            "interview": r"\b(?:interview|job|career|hiring|prepare.*interview|interview.*question)\b",
            "analysis": r"\b(?:analyze|review|evaluate|assess|compare|examine|pros.*cons|advantages.*disadvantages)\b",
            "translation": r"\b(?:translate|translation|language.*to.*language)\b",
            "summarization": r"\b(?:summarize|summary|tldr|brief|key points)\b",
        }

        # Model capabilities mapping - Updated for 4 models
        self.model_capabilities = {
            "phi3.5": ["math", "reasoning", "logic", "scientific", "analysis"],
            "mistral:7b-instruct-q4_0": [
                "factual",
                "general",
                "translation",
                "summary",
            ],
            "gemma:7b-instruct": [
                "coding",
                "technical",
                "programming",
                "documentation",
            ],
            "llama3:8b-instruct-q4_0": [
                "creative",
                "storytelling",
                "writing",
                "conversation",
            ],
        }

        self._initialized = False

    async def initialize(self):
        """Initialize enhanced router with proper error handling"""
        if self._initialized:
            return

        try:
            # Initialize base router if available
            if self.base_router:
                await self.base_router.initialize()
                # Copy available models from base router
                if hasattr(self.base_router, "available_models"):
                    self.available_models = self.base_router.available_models
                else:
                    self.available_models = {}
                logging.info("Base router initialized successfully")
            else:
                # Fallback initialization
                self.available_models = {
                    "phi3.5": {"priority": 1, "good_for": ["math", "reasoning"]},
                    "mistral:7b-instruct-q4_0": {
                        "priority": 2,
                        "good_for": ["general"],
                    },
                    "gemma:7b-instruct": {"priority": 2, "good_for": ["coding"]},
                    "llama3:8b-instruct-q4_0": {
                        "priority": 3,
                        "good_for": ["creative"],
                    },
                }
                logging.warning("Using fallback model configuration")

            # Initialize semantic classifier with better error handling
            if SemanticIntentClassifier:
                try:
                    self.semantic_classifier = SemanticIntentClassifier()
                    await self.semantic_classifier.initialize()
                    if self.semantic_classifier.model:
                        logging.info("✅ Semantic classifier initialized successfully")
                    else:
                        logging.warning(
                            "⚠️ Semantic classifier model not loaded - using rule-based only"
                        )
                        self.semantic_classifier = None
                except Exception as e:
                    logging.warning(
                        f"⚠️ Semantic classifier initialization failed: {str(e)}"
                    )
                    self.semantic_classifier = None
            else:
                logging.info(
                    "ℹ️ Semantic classifier not available - using rule-based classification"
                )

            self._initialized = True
            logging.info("🎯 Enhanced LLM Router initialized successfully")

        except Exception as e:
            logging.error(f"❌ Enhanced router initialization failed: {str(e)}")
            # Mark as initialized even if some features failed
            self._initialized = True

    async def route_request(self, request) -> str:
        """Enhanced routing with semantic classification"""
        if not self._initialized:
            await self.initialize()

        # Handle different request types
        if hasattr(request, "model") and request.model:
            # Check if explicitly requested model is available
            if request.model in self.available_models:
                return request.model

        # Extract text content for classification
        text_content = self._extract_text_content(request)
        if not text_content:
            return self._get_default_model()

        # Get explicit intent if available
        explicit_intent = getattr(request, "intent", None)

        # Classify intent
        intent = await self.classify_intent(text_content, explicit_intent)

        # Select model based on intent and requirements
        selected_model = self._select_model_for_intent(intent, text_content, request)

        logging.info(
            f"🎯 Enhanced routing: intent='{intent}' → model='{selected_model}'"
        )
        return selected_model

    async def classify_intent(
        self, text: str, explicit_intent: Optional[str] = None
    ) -> str:
        """Enhanced intent classification with caching and fallbacks"""

        if explicit_intent:
            return explicit_intent

        # Check cache first
        cache_key = self._generate_cache_key(text)
        if cache_key in self.classification_cache:
            self.cache_stats["hits"] += 1
            return self.classification_cache[cache_key]

        self.cache_stats["misses"] += 1

        # Try rule-based classification first (fast and reliable)
        rule_based_intent = self._rule_based_classification(text)
        if rule_based_intent != "unknown":
            self.cache_stats["rule_based_calls"] += 1
            self._cache_classification(cache_key, rule_based_intent)
            return rule_based_intent

        # Try semantic classification if available
        if self.semantic_classifier:
            try:
                semantic_intent = await self._semantic_classification(text)
                if semantic_intent != "unknown":
                    self.cache_stats["semantic_calls"] += 1
                    self._cache_classification(cache_key, semantic_intent)
                    return semantic_intent
            except Exception as e:
                logging.warning(f"Semantic classification failed: {str(e)}")

        # Final fallback to heuristic classification
        heuristic_intent = self._heuristic_classification(text)
        self._cache_classification(cache_key, heuristic_intent)
        return heuristic_intent

    def _extract_text_content(self, request) -> str:
        """Extract text content from various request types"""
        text_parts = []

        try:
            # Handle ChatCompletionRequest
            if hasattr(request, "messages") and request.messages:
                for message in request.messages:
                    if hasattr(message, "role") and hasattr(message, "content"):
                        if message.role == "user":
                            text_parts.append(message.content)
                    elif isinstance(message, dict):
                        if message.get("role") == "user":
                            text_parts.append(message.get("content", ""))

            # Handle CompletionRequest
            elif hasattr(request, "prompt") and request.prompt:
                text_parts.append(request.prompt)

            # Handle dict-like requests
            elif isinstance(request, dict):
                if "messages" in request:
                    for msg in request["messages"]:
                        if msg.get("role") == "user":
                            text_parts.append(msg.get("content", ""))
                elif "prompt" in request:
                    text_parts.append(request["prompt"])

        except Exception as e:
            logging.warning(f"Error extracting text content: {str(e)}")

        return " ".join(text_parts) if text_parts else ""

    def _rule_based_classification(self, text: str) -> str:
        """Rule-based classification using enhanced patterns"""
        text_lower = text.lower()

        # Score each intent based on pattern matches
        intent_scores = {}
        for intent, pattern in self.intent_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                intent_scores[intent] = len(matches)

        # Return intent with highest score, or 'unknown'
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]

        return "unknown"

    async def _semantic_classification(self, text: str) -> str:
        """Semantic classification with proper error handling"""
        if not self.semantic_classifier:
            return "unknown"

        try:
            intent, confidence = await self.semantic_classifier.classify_intent(text)

            # Use result only if confidence is high enough
            threshold = getattr(self.semantic_classifier, "confidence_threshold", 0.7)
            if confidence >= threshold:
                logging.debug(
                    f"Semantic classification: {intent} (confidence: {confidence:.3f})"
                )
                return intent
            else:
                logging.debug(
                    f"Semantic classification confidence too low: {confidence:.3f} < {threshold}"
                )
                return "unknown"

        except Exception as e:
            logging.error(f"Semantic classification error: {str(e)}")
            return "unknown"

    def _heuristic_classification(self, text: str) -> str:
        """Improved heuristic classification as final fallback"""
        text_lower = text.lower()
        word_count = len(text.split())

        # Length-based heuristics
        if word_count < 5:
            return "factual"  # Very short queries are usually factual
        elif word_count > 150:
            if any(
                word in text_lower
                for word in ["analyze", "review", "evaluate", "compare"]
            ):
                return "analysis"
            else:
                return "creative"  # Long queries are often creative

        # Content-based heuristics with better patterns
        content_indicators = {
            "resume": [
                "resume",
                "cv",
                "experience",
                "qualifications",
                "work history",
                "skills",
            ],
            "interview": [
                "interview",
                "job application",
                "career",
                "hiring",
                "position",
            ],
            "creative": [
                "write",
                "create",
                "compose",
                "story",
                "poem",
                "article",
                "blog",
            ],
            "coding": [
                "code",
                "function",
                "algorithm",
                "programming",
                "script",
                "debug",
            ],
            "math": ["calculate", "solve", "equation", "formula", "math", "arithmetic"],
            "analysis": [
                "analyze",
                "compare",
                "evaluate",
                "review",
                "assess",
                "examine",
            ],
        }

        for intent, indicators in content_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return intent

        # Default to factual for general questions
        return "factual"

    def _select_model_for_intent(self, intent: str, text: str, request) -> str:
        """Select the best model for the given intent and requirements"""

        # Get models that are good for this intent
        suitable_models = []
        for model_name, capabilities in self.model_capabilities.items():
            if model_name in self.available_models and (
                intent in capabilities or "general" in capabilities
            ):
                model_config = self.available_models[model_name]
                suitable_models.append((model_name, model_config.get("priority", 99)))

        # If no specific models found, use all available models
        if not suitable_models:
            suitable_models = [
                (name, config.get("priority", 99))
                for name, config in self.available_models.items()
            ]

        if not suitable_models:
            return self._get_default_model()

        # Consider request requirements
        max_tokens = getattr(request, "max_tokens", None) or 2048

        # Filter by context length if available
        context_suitable = []
        for model_name, priority in suitable_models:
            model_config = self.available_models[model_name]
            max_context = model_config.get("max_context", 4096)
            if max_context >= max_tokens:
                context_suitable.append((model_name, priority))

        # Use context-suitable models if available, otherwise use all suitable
        final_candidates = context_suitable if context_suitable else suitable_models

        # Select model with highest priority (lowest priority number)
        selected = min(final_candidates, key=lambda x: x[1])
        return selected[0]

    def _get_default_model(self) -> str:
        """Get default model as fallback"""
        if self.available_models:
            # Return model with highest priority (lowest priority number)
            return min(
                self.available_models.items(), key=lambda x: x[1].get("priority", 99)
            )[0]
        else:
            return "mistral:7b-instruct-q4_0"  # Hardcoded fallback

    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key from text (first 200 chars for efficiency)"""
        return str(hash(text[:200].lower().strip()))

    def _cache_classification(self, cache_key: str, intent: str):
        """Cache classification result with size management"""

        # Clean up cache if it's getting too large
        if len(self.classification_cache) >= self.cache_max_size:
            # Remove oldest 20% of entries (simple FIFO)
            items_to_remove = len(self.classification_cache) // 5
            keys_to_remove = list(self.classification_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.classification_cache[key]

        self.classification_cache[cache_key] = intent

        # Periodic cleanup
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() > self._cleanup_interval:
            self._cleanup_background_tasks()
            self._last_cleanup = now

    def _cleanup_background_tasks(self):
        """Clean up completed background tasks"""
        completed_tasks = [task for task in self.background_tasks if task.done()]
        for task in completed_tasks:
            self.background_tasks.discard(task)

    # Delegate other methods to base router if available
    async def ensure_model_loaded(self, model: str) -> bool:
        """Ensure model is loaded"""
        if self.base_router and hasattr(self.base_router, "ensure_model_loaded"):
            return await self.base_router.ensure_model_loaded(model)
        return True  # Assume loaded

    async def process_chat_completion(self, request, model: str):
        """Process chat completion"""
        if self.base_router and hasattr(self.base_router, "process_chat_completion"):
            return await self.base_router.process_chat_completion(request, model)
        else:
            raise NotImplementedError("Base router not available for processing")

    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available models"""
        if self.base_router and hasattr(self.base_router, "get_available_models"):
            return await self.base_router.get_available_models()
        else:
            # Return basic model info
            return [
                {"id": model, "object": "model"}
                for model in self.available_models.keys()
            ]

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get comprehensive classification statistics"""

        total_classifications = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / max(1, total_classifications)) * 100

        stats = {
            "cache_stats": {
                "size": len(self.classification_cache),
                "max_size": self.cache_max_size,
                "hit_rate": hit_rate,
                **self.cache_stats,
            },
            "classification_methods": {
                "semantic_available": self.semantic_classifier is not None,
                "rule_based_calls": self.cache_stats["rule_based_calls"],
                "semantic_calls": self.cache_stats["semantic_calls"],
            },
            "supported_intents": list(self.intent_patterns.keys()),
            "model_capabilities": self.model_capabilities,
        }

        # Add semantic classifier stats if available
        if self.semantic_classifier:
            try:
                semantic_stats = self.semantic_classifier.get_classification_stats()
                stats["semantic_classifier"] = semantic_stats
            except Exception as e:
                stats["semantic_classifier_error"] = str(e)

        return stats

    def get_routing_recommendations(self) -> List[str]:
        """Get routing optimization recommendations"""
        recommendations = []

        total_calls = (
            self.cache_stats["rule_based_calls"] + self.cache_stats["semantic_calls"]
        )
        if total_calls > 0:
            semantic_ratio = self.cache_stats["semantic_calls"] / total_calls
            if semantic_ratio < 0.1:
                recommendations.append(
                    "Consider enabling semantic classification for better routing accuracy"
                )

        cache_hit_rate = self.cache_stats["hits"] / max(
            1, self.cache_stats["hits"] + self.cache_stats["misses"]
        )
        if cache_hit_rate < 0.5:
            recommendations.append(
                "Low cache hit rate - consider increasing cache size or review query patterns"
            )

        if len(self.available_models) < 2:
            recommendations.append(
                "Consider adding more models for better intent-based routing"
            )

        return recommendations

    async def cleanup(self):
        """Cleanup resources"""
        # Cancel all background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()

        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        self.classification_cache.clear()

        # Cleanup base router
        if self.base_router and hasattr(self.base_router, "cleanup"):
            await self.base_router.cleanup()

        logging.info("Enhanced router cleanup completed")


# Factory function for easy creation
def create_enhanced_router(ollama_client=None) -> EnhancedLLMRouter:
    """Factory function to create enhanced router"""
    return EnhancedLLMRouter(ollama_client)
