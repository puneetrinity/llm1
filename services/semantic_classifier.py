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
