# 1. Navigate to your working directory
cd /workspace/app  # or wherever your main.py is

# 2. Create semantic enhancement directory
mkdir -p semantic_enhancement
cd semantic_enhancement

# 3. Download semantic_classifier.py directly
wget -O semantic_classifier.py "https://raw.githubusercontent.com/anthropics/semantic-llm-router/main/services/semantic_classifier.py"

# 4. If that fails, create it manually:
cat > semantic_classifier.py << 'EOF'
# services/semantic_classifier.py - Fixed Semantic Classification with Memory Limits
import numpy as np
import logging
import asyncio
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import threading

class SemanticIntentClassifier:
    def __init__(self, max_memory_mb: int = 500):
        self.model = None
        self.index = None
        self.intent_examples = {}
        self.intent_labels = []
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.confidence_threshold = 0.7
        self.max_memory_mb = max_memory_mb
        self.max_training_examples = 1000  # Memory limit
        self._lock = threading.Lock()
        self._initialized = False
        
        # Simplified training data with memory limits
        self.training_data = {
            'math': [
                "calculate 15% of 250", "what is 45 * 67?", "solve for x: 2x + 5 = 15",
                "compute the square root of 144", "find the area of a circle with radius 5"
            ],
            'factual': [
                "what is the capital of France?", "who invented the telephone?",
                "when did world war 2 end?", "what is photosynthesis?", "define machine learning"
            ],
            'resume': [
                "analyze this resume for technical skills", "review my CV and suggest improvements",
                "what experience does this candidate have?", "evaluate this resume for a software engineer role"
            ],
            'interview': [
                "prepare me for a software engineer interview", "what questions should I expect for a data scientist role?",
                "help me practice behavioral interview questions", "common interview questions for product manager"
            ],
            'creative': [
                "write a short story about space travel", "compose a poem about friendship",
                "create a marketing copy for a new product", "write a blog post about artificial intelligence"
            ],
            'coding': [
                "write a Python function to sort a list", "debug this JavaScript code",
                "create a REST API in FastAPI", "optimize this SQL query"
            ],
            'analysis': [
                "analyze the pros and cons of remote work", "compare different cloud providers",
                "evaluate market trends in AI industry", "assess the impact of new regulations"
            ]
        }
    
    async def initialize(self):
        """Initialize the semantic classifier with error handling and memory limits"""
        
        if self._initialized:
            return
        
        logging.info("Initializing semantic intent classifier...")
        
        try:
            # Try to import and load the model
            await self._load_model()
            
            # Build index only if model loaded successfully
            if self.model:
                await self._build_index()
                logging.info("Semantic classifier initialized successfully")
            else:
                logging.warning("Semantic classifier model not loaded - using fallback only")
            
            self._initialized = True
            
        except Exception as e:
            logging.error(f"Failed to initialize semantic classifier: {str(e)}")
            logging.info("Semantic classification will be disabled")
            self.model = None
            self._initialized = True  # Mark as initialized even if failed
    
    async def _load_model(self):
        """Load sentence transformer model with memory monitoring"""
        
        try:
            # Import here to make it optional
            from sentence_transformers import SentenceTransformer
            
            # Load lightweight model
            logging.info("Loading sentence transformer model...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Verify model works with a test encoding
            test_encoding = self.model.encode(["test sentence"])
            if test_encoding is None or len(test_encoding) == 0:
                raise Exception("Model encoding test failed")
            
            logging.info("Sentence transformer model loaded successfully")
            
        except ImportError:
            logging.warning("sentence-transformers not available - semantic classification disabled")
            self.model = None
        except Exception as e:
            logging.error(f"Failed to load sentence transformer model: {str(e)}")
            self.model = None
    
    async def _build_index(self):
        """Build FAISS index with memory limits"""
        
        if not self.model:
            return
        
        try:
            # Import FAISS with fallback
            try:
                import faiss
            except ImportError:
                logging.warning("FAISS not available - using simple similarity search")
                self.index = None
                return
            
            # Limit training examples to control memory usage
            all_examples = []
            all_labels = []
            
            for intent, examples in self.training_data.items():
                # Limit examples per intent
                limited_examples = examples[:20]  # Max 20 examples per intent
                all_examples.extend(limited_examples)
                all_labels.extend([intent] * len(limited_examples))
            
            # Limit total examples
            if len(all_examples) > self.max_training_examples:
                all_examples = all_examples[:self.max_training_examples]
                all_labels = all_labels[:self.max_training_examples]
            
            # Generate embeddings in batches to control memory
            batch_size = 50
            embeddings = []
            
            for i in range(0, len(all_examples), batch_size):
                batch = all_examples[i:i+batch_size]
                batch_embeddings = self.model.encode(batch)
                embeddings.append(batch_embeddings)
            
            if embeddings:
                embeddings = np.vstack(embeddings)
                
                # Create FAISS index
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings)
                self.index.add(embeddings.astype('float32'))
                
                self.intent_labels = all_labels
                
                logging.info(f"Built FAISS index with {len(all_examples)} examples")
            else:
                logging.warning("No embeddings generated - FAISS index not created")
                self.index = None
                
        except Exception as e:
            logging.error(f"Failed to build FAISS index: {str(e)}")
            self.index = None
    
    async def classify_intent(self, text: str, top_k: int = 3) -> Tuple[str, float]:
        """Classify intent using semantic similarity with fallbacks"""
        
        if not self._initialized:
            await self.initialize()
        
        if not self.model or not self.index:
            return "unknown", 0.0
        
        try:
            # Run in thread pool to avoid blocking
            return await asyncio.get_event_loop().run_in_executor(
                None, self._classify_sync, text, top_k
            )
        except Exception as e:
            logging.error(f"Error in semantic classification: {str(e)}")
            return "unknown", 0.0
    
    def _classify_sync(self, text: str, top_k: int) -> Tuple[str, float]:
        """Synchronous classification to run in thread pool"""
        
        with self._lock:
            try:
                # Generate embedding for input text
                query_embedding = self.model.encode([text])
                
                # Import faiss here for thread safety
                import faiss
                faiss.normalize_L2(query_embedding)
                
                # Search for similar examples
                similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
                
                # Count votes for each intent
                intent_scores = {}
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.intent_labels):  # Safety check
                        intent = self.intent_labels[idx]
                        score = similarities[0][i]
                        
                        if intent not in intent_scores:
                            intent_scores[intent] = []
                        intent_scores[intent].append(score)
                
                if not intent_scores:
                    return "unknown", 0.0
                
                # Calculate average scores
                intent_avg_scores = {
                    intent: np.mean(scores) 
                    for intent, scores in intent_scores.items()
                }
                
                # Get best intent
                best_intent = max(intent_avg_scores, key=intent_avg_scores.get)
                best_score = intent_avg_scores[best_intent]
                
                return best_intent, best_score
                
            except Exception as e:
                logging.error(f"Error in sync classification: {str(e)}")
                return "unknown", 0.0
    
    def get_classification_stats(self) -> Dict:
        """Get classification statistics"""
        
        total_examples = sum(len(examples) for examples in self.training_data.values())
        
        return {
            "model_loaded": self.model is not None,
            "index_loaded": self.index is not None,
            "total_training_examples": total_examples,
            "intents": list(self.training_data.keys()),
            "examples_per_intent": {
                intent: len(examples) 
                for intent, examples in self.training_data.items()
            },
            "index_size": self.index.ntotal if self.index else 0,
            "confidence_threshold": self.confidence_threshold,
            "max_memory_mb": self.max_memory_mb
        }
EOF

# 5. Download enhanced_router.py
cat > enhanced_router.py << 'EOF'
# services/enhanced_router.py - Enhanced Router with Semantic Classification (Fixed)
import re
import asyncio
import logging
from typing import Optional, Dict, Any

class EnhancedLLMRouter:
    def __init__(self, ollama_client=None, base_router=None):
        self.ollama_client = ollama_client
        self.base_router = base_router
        self.semantic_classifier = None
        self.classification_cache = {}
        self.cache_max_size = 1000  # Memory limit for cache
        
        # Enhanced model configuration for your use case
        self.model_config = {
            'mistral:7b-instruct-q4_0': {
                'priority': 1,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'memory_mb': 4500,
                'good_for': ['factual', 'math', 'general', 'creative']
            },
            'deepseek-v2:7b-q4_0': {
                'priority': 1,  # High priority for coding
                'cost_per_token': 0.00015,
                'max_context': 4096,
                'memory_mb': 4200,
                'good_for': ['coding', 'analysis']
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'memory_mb': 5000,
                'good_for': ['creative', 'interview', 'resume', 'analysis']
            }
        }
        
        # Intent to model mapping - PERFECT for your use case
        self.intent_model_mapping = {
            'coding': 'deepseek-v2:7b-q4_0',      # DeepSeek for code
            'analysis': 'llama3:8b-instruct-q4_0',  # Llama3 for analysis
            'resume': 'llama3:8b-instruct-q4_0',    # Llama3 for HR tasks
            'interview': 'llama3:8b-instruct-q4_0', # Llama3 for conversations
            'creative': 'mistral:7b-instruct-q4_0', # Mistral for creative
            'factual': 'mistral:7b-instruct-q4_0',  # Mistral for facts
            'math': 'mistral:7b-instruct-q4_0',     # Mistral for math
            'general': 'mistral:7b-instruct-q4_0'   # Mistral default
        }
        
        self.available_models = {}
        
    async def initialize(self):
        """Initialize enhanced router with semantic classifier"""
        
        # Initialize base functionality first
        if self.base_router:
            await self.base_router.initialize()
            self.available_models = getattr(self.base_router, 'available_models', {})
        
        # Check available models if no base router
        if not self.available_models and self.ollama_client:
            try:
                available_models = await self.ollama_client.list_models()
                available_model_names = {model.get('name', '') for model in available_models}
                self.available_models = {
                    name: config for name, config in self.model_config.items()
                    if name in available_model_names
                }
            except Exception as e:
                logging.warning(f"Could not check available models: {e}")
                # Fallback - assume all configured models are available
                self.available_models = self.model_config.copy()
        
        # Initialize semantic classifier
        try:
            from semantic_classifier import SemanticIntentClassifier
            self.semantic_classifier = SemanticIntentClassifier()
            await self.semantic_classifier.initialize()
            logging.info("✅ Enhanced router with semantic classification initialized")
        except Exception as e:
            logging.warning(f"Semantic classification not available: {e}")
            logging.info("✅ Enhanced router with rule-based classification initialized")
        
        logging.info(f"Available models for routing: {list(self.available_models.keys())}")
    
    async def route_request(self, request) -> str:
        """Enhanced routing with semantic classification"""
        
        # Use explicit model if specified and valid
        if hasattr(request, 'model') and request.model in self.available_models:
            return request.model
        
        # Extract text content for classification
        text_content = self._extract_text_content(request)
        
        # Classify intent
        intent, confidence = await self.classify_intent(text_content, getattr(request, 'intent', None))
        
        # Select model based on intent
        selected_model = self._select_model_for_intent(intent, text_content, request)
        
        logging.info(f"🎯 Enhanced routing: '{text_content[:50]}...' → intent={intent} (conf={confidence:.2f}) → {selected_model}")
        return selected_model
    
    def _extract_text_content(self, request) -> str:
        """Extract text content from request"""
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
    
    async def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> tuple:
        """Classify intent with caching and fallbacks"""
        
        if explicit_intent:
            return explicit_intent, 1.0
        
        if not text.strip():
            return 'general', 0.5
        
        # Check cache
        cache_key = hash(text[:200])  # Use first 200 chars for cache key
        if cache_key in self.classification_cache:
            return self.classification_cache[cache_key]
        
        # Try semantic classification first
        if self.semantic_classifier:
            try:
                intent, confidence = await self.semantic_classifier.classify_intent(text)
                if confidence > 0.5:  # Use if confidence is reasonable
                    result = (intent, confidence)
                    self._cache_classification(cache_key, result)
                    return result
            except Exception as e:
                logging.debug(f"Semantic classification failed: {e}")
        
        # Fallback to rule-based classification
        intent = self._rule_based_classification(text)
        confidence = 0.8 if intent != 'general' else 0.5
        
        result = (intent, confidence)
        self._cache_classification(cache_key, result)
        return result
    
    def _rule_based_classification(self, text: str) -> str:
        """Rule-based classification fallback"""
        text_lower = text.lower()
        
        # Coding patterns
        if any(word in text_lower for word in ['code', 'debug', 'function', 'algorithm', 'script', 'program', 'bug', 'syntax']):
            return 'coding'
        
        # Resume/HR patterns  
        elif any(word in text_lower for word in ['resume', 'cv', 'candidate', 'experience', 'skills', 'qualifications']):
            return 'resume'
        
        # Interview patterns
        elif any(word in text_lower for word in ['interview', 'questions', 'prepare', 'practice', 'behavioral']):
            return 'interview'
        
        # Analysis patterns
        elif any(word in text_lower for word in ['analyze', 'compare', 'evaluate', 'assess', 'review', 'pros', 'cons']):
            return 'analysis'
        
        # Creative patterns
        elif any(word in text_lower for word in ['write', 'create', 'draft', 'compose', 'story', 'blog']):
            return 'creative'
        
        # Math patterns
        elif any(word in text_lower for word in ['calculate', 'solve', 'math', 'equation', 'compute']):
            return 'math'
        
        # Factual patterns
        elif any(word in text_lower for word in ['what is', 'define', 'explain', 'who is', 'when']):
            return 'factual'
        
        else:
            return 'general'
    
    def _select_model_for_intent(self, intent: str, text: str, request) -> str:
        """Select best model for classified intent"""
        
        # Get preferred model for intent
        preferred_model = self.intent_model_mapping.get(intent)
        
        # Check if preferred model is available
        if preferred_model and preferred_model in self.available_models:
            return preferred_model
        
        # Fallback: find any model that can handle this intent
        for model_name, config in self.available_models.items():
            if intent in config.get('good_for', []):
                return model_name
        
        # Final fallback: use first available model
        if self.available_models:
            return list(self.available_models.keys())[0]
        
        # Emergency fallback
        return 'mistral:7b-instruct-q4_0'
    
    def _cache_classification(self, cache_key: int, result: tuple):
        """Cache classification result with size limit"""
        # Implement LRU-like behavior
        if len(self.classification_cache) >= self.cache_max_size:
            # Remove oldest 20% of entries
            items_to_remove = len(self.classification_cache) // 5
            keys_to_remove = list(self.classification_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.classification_cache[key]
        
        self.classification_cache[cache_key] = result
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification and routing statistics"""
        stats = {
            "cache_size": len(self.classification_cache),
            "cache_max_size": self.cache_max_size,
            "semantic_classifier_available": self.semantic_classifier is not None,
            "available_models": list(self.available_models.keys()),
            "intent_mappings": self.intent_model_mapping
        }
        
        if self.semantic_classifier:
            try:
                stats.update(self.semantic_classifier.get_classification_stats())
            except Exception as e:
                stats["semantic_stats_error"] = str(e)
        
        return stats
    
    # Delegate methods to base router if available
    async def process_chat_completion(self, request, model: str):
        if self.base_router and hasattr(self.base_router, 'process_chat_completion'):
            return await self.base_router.process_chat_completion(request, model)
        else:
            raise NotImplementedError("Base router required for processing")
    
    async def get_available_models(self):
        if self.base_router and hasattr(self.base_router, 'get_available_models'):
            return await self.base_router.get_available_models()
        else:
            return [{"id": model, "object": "model"} for model in self.available_models.keys()]
    
    async def cleanup(self):
        self.classification_cache.clear()
        if self.base_router and hasattr(self.base_router, 'cleanup'):
            await self.base_router.cleanup()
EOF

# 6. Create integration script
cat > integrate_semantic_routing.py << 'EOF'
#!/usr/bin/env python3
"""Safe Semantic Routing Integration"""

import sys
import os
import shutil
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    setup_logging()
    
    print("🚀 Safe Semantic Routing Integration")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("../main.py").exists() and not Path("main.py").exists():
        print("❌ Please run from your LLM proxy directory or semantic_enhancement subdirectory")
        return 1
    
    # Determine paths
    if Path("../main.py").exists():
        # We're in semantic_enhancement subdirectory
        app_dir = Path("..")
        source_dir = Path(".")
    else:
        # We're in app directory
        app_dir = Path(".")
        source_dir = Path("semantic_enhancement")
    
    services_dir = app_dir / "services"
    services_dir.mkdir(exist_ok=True)
    
    # Copy files safely
    try:
        # Copy semantic classifier
        src_classifier = source_dir / "semantic_classifier.py"
        dst_classifier = services_dir / "semantic_classifier.py"
        
        if src_classifier.exists() and not dst_classifier.exists():
            shutil.copy(src_classifier, dst_classifier)
            print("✅ Added semantic_classifier.py")
        elif dst_classifier.exists():
            print("ℹ️  semantic_classifier.py already exists")
        
        # Copy enhanced router
        src_router = source_dir / "enhanced_router.py"
        dst_router = services_dir / "enhanced_router.py"
        
        if src_router.exists() and not dst_router.exists():
            shutil.copy(src_router, dst_router)
            print("✅ Added enhanced_router.py")
        elif dst_router.exists():
            print("ℹ️  enhanced_router.py already exists")
        
        print("\n🎉 Integration Complete!")
        print("Files are ready - your system will auto-detect them")
        print("\nNext steps:")
        print("1. cd .. (if in semantic_enhancement directory)")
        print("2. python3 main.py")
        print("3. Watch for '✅ Enhanced semantic routing' in logs")
        
        return 0
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x integrate_semantic_routing.py

print "✅ All files created successfully!"
echo ""
echo "📋 Files downloaded:"
echo "  - semantic_classifier.py (the missing file!)"
echo "  - enhanced_router.py" 
echo "  - integrate_semantic_routing.py"
echo ""
echo "🚀 Next step:"
echo "  python3 integrate_semantic_routing.py"
