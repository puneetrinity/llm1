# Enhanced router with semantic classification
# services/enhanced_router.py - Enhanced Routing with Semantic Classification
from services.semantic_classifier import SemanticIntentClassifier
from services.router import LLMRouter as BaseRouter

class EnhancedLLMRouter(BaseRouter):
    def __init__(self):
        super().__init__()
        self.semantic_classifier = SemanticIntentClassifier()
        self.classification_cache = {}  # Cache for recent classifications
        
    async def initialize(self):
        """Initialize enhanced router with semantic classifier"""
        await super().initialize()
        await self.semantic_classifier.initialize()
        
        logging.info("Enhanced LLM Router initialized")
    
    def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
        """Enhanced intent classification with semantic fallback"""
        
        if explicit_intent:
            return explicit_intent
        
        # Check cache first
        cache_key = hash(text[:200])  # Use first 200 chars for cache key
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # Try rule-based classification first (fast)
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text):
                self.classification_cache[cache_key] = intent
                return intent
        
        # Fallback to semantic classification
        return self.semantic_classify_with_fallback(text, cache_key)
    
    def semantic_classify_with_fallback(self, text: str, cache_key: int) -> str:
        """Async semantic classification with synchronous fallback"""
        
        try:
            # Create a task for semantic classification
            loop = asyncio.get_event_loop()
            
            # Try to get semantic classification
            if loop.is_running():
                # If we're in an async context, schedule the classification
                task = asyncio.create_task(self._semantic_classify_async(text, cache_key))
                # Return quick heuristic classification for now
                return self.heuristic_classification(text)
            else:
                # Synchronous fallback
                return self.heuristic_classification(text)
                
        except Exception as e:
            logging.warning(f"Semantic classification failed: {str(e)}")
            return self.heuristic_classification(text)
    
    async def _semantic_classify_async(self, text: str, cache_key: int):
        """Async semantic classification that updates cache"""
        
        try:
            intent, confidence = await self.semantic_classifier.classify_intent(text)
            
            if confidence > self.semantic_classifier.confidence_threshold:
                self.classification_cache[cache_key] = intent
                logging.debug(f"Semantic classification: {intent} (confidence: {confidence:.3f})")
            else:
                # Fall back to heuristic
                intent = self.heuristic_classification(text)
                self.classification_cache[cache_key] = intent
                
        except Exception as e:
            logging.error(f"Error in async semantic classification: {str(e)}")
    
    def heuristic_classification(self, text: str) -> str:
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
        
        return 'factual'  # Default fallback
