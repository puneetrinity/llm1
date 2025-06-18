# services/semantic_enhanced_router.py - Enhanced Router with Semantic Classification
import re
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime


class EnhancedLLMRouter:
    """Enhanced LLM Router with semantic intent classification for optimal model selection"""

    def __init__(self, ollama_client=None, base_router=None):
        self.ollama_client = ollama_client
        self.base_router = base_router
        self.semantic_classifier = None
        self.classification_cache = {}
        self.cache_max_size = 1000  # Memory limit for cache

        # Enhanced model configuration optimized for your A5000 setup
        self.model_config = {
            'mistral:7b-instruct-q4_0': {
                'priority': 1,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'memory_mb': 4500,
                'good_for': ['factual', 'math', 'general', 'creative'],
                'specialties': ['quick_facts', 'calculations', 'general_chat']
            },
            'deepseek-v2:7b-q4_0': {
                'priority': 1,  # High priority for coding tasks
                'cost_per_token': 0.00015,
                'max_context': 4096,
                'memory_mb': 4200,
                'good_for': ['coding', 'analysis', 'technical'],
                'specialties': ['code_review', 'debugging', 'programming', 'technical_analysis']
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'memory_mb': 5000,
                'good_for': ['creative', 'interview', 'resume', 'analysis'],
                'specialties': ['hr_tasks', 'interviews', 'writing', 'evaluation']
            }
        }

        # Intent to model mapping - optimized for your use case
        self.intent_model_mapping = {
            # Coding tasks → DeepSeek (code specialist)
            'coding': 'deepseek-v2:7b-q4_0',

            # Analysis tasks → Llama3 (analytical reasoning)
            'analysis': 'llama3:8b-instruct-q4_0',

            # HR/Resume tasks → Llama3 (evaluation & analysis)
            'resume': 'llama3:8b-instruct-q4_0',

            # Interview preparation → Llama3 (conversational & evaluative)
            'interview': 'llama3:8b-instruct-q4_0',

            # Creative writing → Mistral (creative & fluent)
            'creative': 'mistral:7b-instruct-q4_0',

            # Quick facts → Mistral (fast & accurate)
            'factual': 'mistral:7b-instruct-q4_0',

            # Math problems → Mistral (good at calculations)
            'math': 'mistral:7b-instruct-q4_0',

            # Default fallback → Mistral (most versatile)
            'general': 'mistral:7b-instruct-q4_0'
        }

        # Enhanced rule-based patterns for immediate classification
        self.intent_patterns = {
            'coding': r'\b(?:code|debug|function|algorithm|script|program|bug|syntax|review|refactor|optimize|programming|javascript|python|java|sql|api|github|git)\b',
            'resume': r'\b(?:resume|cv|candidate|experience|skills|qualifications|background|hiring|recruit|applicant|work history|employment)\b',
            'interview': r'\b(?:interview|questions|prepare|practice|behavioral|technical interview|job interview|mock|preparation)\b',
            'analysis': r'\b(?:analyze|compare|evaluate|assess|review|pros and cons|advantages|disadvantages|evaluation|assessment|comparison)\b',
            'creative': r'\b(?:write|create|draft|compose|story|blog|article|creative|content|marketing|copy|email|letter)\b',
            'math': r'\b(?:calculate|solve|math|equation|compute|formula|percentage|statistics|algebra|geometry)\b',
            'factual': r'\b(?:what is|define|explain|who is|when|where|how does|tell me about|meaning of)\b'
        }

        self.available_models = {}
        self.routing_stats = {
            'total_requests': 0,
            'intent_counts': {},
            'model_usage': {},
            'cache_hits': 0
        }

    async def initialize(self):
        """Initialize enhanced router with semantic classifier"""
        logging.info(
            "Initializing Enhanced LLM Router with semantic classification...")

        # Initialize base router functionality first
        if self.base_router and hasattr(self.base_router, 'initialize'):
            try:
                await self.base_router.initialize()
                self.available_models = getattr(
                    self.base_router, 'available_models', {})
                logging.info("✅ Base router initialized")
            except Exception as e:
                logging.warning(f"Base router initialization failed: {e}")

        # Check available models if no base router
        if not self.available_models:
            await self._detect_available_models()

        # Initialize semantic classifier
        try:
            from semantic_classifier import SemanticIntentClassifier
            self.semantic_classifier = SemanticIntentClassifier()
            await self.semantic_classifier.initialize()
            logging.info("✅ Semantic classifier initialized successfully")
        except ImportError as e:
            logging.warning(f"Semantic classifier import failed: {e}")
            logging.info(
                "✅ Enhanced router will use rule-based classification only")
        except Exception as e:
            logging.warning(f"Semantic classifier initialization failed: {e}")
            logging.info(
                "✅ Enhanced router will use rule-based classification only")

        # Log final setup
        available_model_names = list(self.available_models.keys())
        logging.info(
            f"Enhanced router initialized with models: {available_model_names}")

        # Validate intent mappings
        self._validate_intent_mappings()

    async def _detect_available_models(self):
        """Detect available models from Ollama"""
        try:
            if self.ollama_client:
                available_models = await self.ollama_client.list_models()
                available_model_names = {
                    model.get('name', '') for model in available_models}

                # Filter configured models by availability
                self.available_models = {
                    name: config for name, config in self.model_config.items()
                    if name in available_model_names
                }

                if not self.available_models:
                    logging.warning(
                        "No configured models found, using fallback configuration")
                    # Use first available model as fallback
                    if available_models:
                        fallback_model = available_models[0]['name']
                        self.available_models[fallback_model] = {
                            'priority': 1, 'cost_per_token': 0.0001, 'max_context': 4096,
                            'memory_mb': 4000, 'good_for': ['general']
                        }
            else:
                logging.warning(
                    "No Ollama client available, using default model configuration")
                self.available_models = self.model_config.copy()

        except Exception as e:
            logging.error(f"Error detecting available models: {e}")
            self.available_models = self.model_config.copy()

    def _validate_intent_mappings(self):
        """Validate that intent mappings point to available models"""
        for intent, model in list(self.intent_model_mapping.items()):
            if model not in self.available_models:
                # Find a suitable alternative
                alternatives = [
                    name for name, config in self.available_models.items()
                    if intent in config.get('good_for', [])
                ]

                if alternatives:
                    self.intent_model_mapping[intent] = alternatives[0]
                    logging.info(
                        f"Mapped {intent} to alternative model: {alternatives[0]}")
                elif self.available_models:
                    # Use first available model as fallback
                    fallback = list(self.available_models.keys())[0]
                    self.intent_model_mapping[intent] = fallback
                    logging.info(
                        f"Mapped {intent} to fallback model: {fallback}")

    async def route_request(self, request) -> str:
        """Enhanced routing with semantic classification"""
        self.routing_stats['total_requests'] += 1

        # Use explicit model if specified and valid
        if hasattr(request, 'model') and request.model in self.available_models:
            model = request.model
            logging.info(f"🎯 Using explicit model: {model}")
            self._update_routing_stats('explicit', model)
            return model

        # Extract text content for classification
        text_content = self._extract_text_content(request)
        if not text_content.strip():
            model = self._get_default_model()
            logging.info(f"🎯 Empty request → default model: {model}")
            self._update_routing_stats('empty', model)
            return model

        # Classify intent
        intent, confidence = await self.classify_intent(text_content, getattr(request, 'intent', None))

        # Select model based on intent
        selected_model = self._select_model_for_intent(
            intent, confidence, text_content, request)

        # Log routing decision
        logging.info(
            f"🧠 Semantic routing: '{text_content[:50]}...' → intent={intent} (conf={confidence:.2f}) → {selected_model}")

        # Update statistics
        self._update_routing_stats(intent, selected_model)

        return selected_model

    def _extract_text_content(self, request) -> str:
        """Extract text content from request for classification"""
        try:
            if hasattr(request, 'messages') and request.messages:
                # Extract user messages
                user_messages = []
                for msg in request.messages:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        if msg.role == 'user':
                            user_messages.append(msg.content)
                    elif isinstance(msg, dict) and msg.get('role') == 'user':
                        user_messages.append(msg.get('content', ''))

                return ' '.join(user_messages)

            elif hasattr(request, 'prompt'):
                return request.prompt

        except Exception as e:
            logging.debug(f"Error extracting text content: {e}")

        return ''

    async def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> Tuple[str, float]:
        """Classify intent with caching and multiple fallback strategies"""

        if explicit_intent:
            return explicit_intent, 1.0

        if not text.strip():
            return 'general', 0.5

        # Check cache first
        cache_key = hash(text[:200])  # Use first 200 chars for cache key
        if cache_key in self.classification_cache:
            self.routing_stats['cache_hits'] += 1
            return self.classification_cache[cache_key]

        # Try semantic classification first (most accurate)
        if self.semantic_classifier:
            try:
                intent, confidence = await self.semantic_classifier.classify_intent(text)
                if confidence > 0.6:  # Use semantic result if confidence is reasonable
                    result = (intent, confidence)
                    self._cache_classification(cache_key, result)
                    return result
            except Exception as e:
                logging.debug(f"Semantic classification failed: {e}")

        # Fallback to rule-based classification
        rule_intent = self._rule_based_classification(text)
        if rule_intent != 'general':
            result = (rule_intent, 0.8)
            self._cache_classification(cache_key, result)
            return result

        # Final fallback to heuristic classification
        heuristic_intent = self._heuristic_classification(text)
        result = (heuristic_intent, 0.6)
        self._cache_classification(cache_key, result)
        return result

    def _rule_based_classification(self, text: str) -> str:
        """Fast rule-based classification using regex patterns"""
        text_lower = text.lower()

        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text_lower):
                return intent

        return 'general'

    def _heuristic_classification(self, text: str) -> str:
        """Heuristic-based classification using content analysis"""
        text_lower = text.lower()
        word_count = len(text.split())

        # Length-based heuristics
        if word_count < 10:
            return 'factual'  # Short queries are usually factual
        elif word_count > 100:
            return 'analysis' if any(word in text_lower for word in ['analyze', 'evaluate', 'compare']) else 'creative'

        # Content-based keyword matching
        keyword_patterns = {
            'coding': ['code', 'function', 'debug', 'program', 'script', 'algorithm'],
            'resume': ['resume', 'cv', 'experience', 'skills', 'candidate'],
            'interview': ['interview', 'questions', 'prepare', 'job'],
            'analysis': ['analyze', 'compare', 'evaluate', 'assess'],
            'creative': ['write', 'create', 'compose', 'draft'],
            'math': ['calculate', 'solve', 'equation', 'formula']
        }

        for intent, keywords in keyword_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return intent

        return 'general'

    def _select_model_for_intent(self, intent: str, confidence: float, text: str, request) -> str:
        """Select the best model for the classified intent"""

        # Get preferred model for intent
        preferred_model = self.intent_model_mapping.get(intent)

        # Check if preferred model is available
        if preferred_model and preferred_model in self.available_models:
            return preferred_model

        # Fallback: find any available model that can handle this intent
        for model_name, config in self.available_models.items():
            if intent in config.get('good_for', []):
                return model_name

        # Final fallback: use highest priority available model
        if self.available_models:
            return min(self.available_models.items(), key=lambda x: x[1]['priority'])[0]

        # Emergency fallback
        return 'mistral:7b-instruct-q4_0'

    def _get_default_model(self) -> str:
        """Get the default model for fallback scenarios"""
        if 'mistral:7b-instruct-q4_0' in self.available_models:
            return 'mistral:7b-instruct-q4_0'
        elif self.available_models:
            return list(self.available_models.keys())[0]
        else:
            return 'mistral:7b-instruct-q4_0'

    def _cache_classification(self, cache_key: int, result: Tuple[str, float]):
        """Cache classification result with LRU-like eviction"""
        if len(self.classification_cache) >= self.cache_max_size:
            # Remove oldest 20% of entries (simple LRU approximation)
            items_to_remove = len(self.classification_cache) // 5
            keys_to_remove = list(self.classification_cache.keys())[
                :items_to_remove]
            for key in keys_to_remove:
                del self.classification_cache[key]

        self.classification_cache[cache_key] = result

    def _update_routing_stats(self, intent: str, model: str):
        """Update routing statistics"""
        self.routing_stats['intent_counts'][intent] = self.routing_stats['intent_counts'].get(
            intent, 0) + 1
        self.routing_stats['model_usage'][model] = self.routing_stats['model_usage'].get(
            model, 0) + 1

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get comprehensive classification and routing statistics"""
        total_requests = self.routing_stats['total_requests']

        stats = {
            "routing_stats": {
                "total_requests": total_requests,
                "cache_hits": self.routing_stats['cache_hits'],
                "cache_hit_rate": (self.routing_stats['cache_hits'] / max(1, total_requests)) * 100,
                "intent_distribution": self.routing_stats['intent_counts'],
                "model_usage_distribution": self.routing_stats['model_usage']
            },
            "cache_info": {
                "cache_size": len(self.classification_cache),
                "cache_max_size": self.cache_max_size
            },
            "system_info": {
                "semantic_classifier_available": self.semantic_classifier is not None,
                "available_models": list(self.available_models.keys()),
                "intent_mappings": self.intent_model_mapping
            }
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
        """Get recommendations for optimizing routing performance"""
        recommendations = []

        total_requests = self.routing_stats['total_requests']
        if total_requests < 10:
            return ["Insufficient data for recommendations (need at least 10 requests)"]

        # Analyze intent distribution
        intent_counts = self.routing_stats['intent_counts']
        if intent_counts:
            most_common_intent = max(intent_counts, key=intent_counts.get)
            intent_percentage = (
                intent_counts[most_common_intent] / total_requests) * 100

            if intent_percentage > 50:
                recommendations.append(
                    f"Consider optimizing for '{most_common_intent}' intent ({intent_percentage:.1f}% of requests)")

        # Analyze model usage
        model_usage = self.routing_stats['model_usage']
        if model_usage:
            most_used_model = max(model_usage, key=model_usage.get)
            usage_percentage = (
                model_usage[most_used_model] / total_requests) * 100

            if usage_percentage > 70:
                recommendations.append(
                    f"Consider keeping '{most_used_model}' warm ({usage_percentage:.1f}% usage)")

        # Cache efficiency
        cache_hit_rate = (
            self.routing_stats['cache_hits'] / max(1, total_requests)) * 100
        if cache_hit_rate < 30:
            recommendations.append(
                f"Low cache hit rate ({cache_hit_rate:.1f}%) - consider increasing cache size")

        return recommendations or ["Routing performance looks optimal"]

    # Delegate methods to base router if available
    async def process_chat_completion(self, request, model: str):
        """Process chat completion - delegate to base router or implement basic version"""
        if self.base_router and hasattr(self.base_router, 'process_chat_completion'):
            return await self.base_router.process_chat_completion(request, model)
        else:
            # Basic implementation if no base router
            raise NotImplementedError(
                "Base router required for processing chat completions")

    async def get_available_models(self):
        """Get available models - delegate to base router or return local list"""
        if self.base_router and hasattr(self.base_router, 'get_available_models'):
            return await self.base_router.get_available_models()
        else:
            # Return local model list
            return [
                {
                    "id": model_name,
                    "object": "model",
                    "created": int(datetime.now().timestamp()),
                    "owned_by": "ollama",
                    **config
                }
                for model_name, config in self.available_models.items()
            ]

    async def ensure_model_loaded(self, model: str) -> bool:
        """Ensure model is loaded - delegate to base router or basic check"""
        if self.base_router and hasattr(self.base_router, 'ensure_model_loaded'):
            return await self.base_router.ensure_model_loaded(model)
        else:
            # Basic check
            return model in self.available_models

    async def cleanup(self):
        """Cleanup resources"""
        self.classification_cache.clear()

        if self.base_router and hasattr(self.base_router, 'cleanup'):
            await self.base_router.cleanup()

        logging.info("Enhanced LLM Router cleaned up")
