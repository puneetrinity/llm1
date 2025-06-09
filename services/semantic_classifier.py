# services/semantic_classifier.py - Advanced Intent Classification
import numpy as np
import faiss
import pickle
import logging
from typing import List, Dict, Tuple, Optional
import asyncio
from sentence_transformers import SentenceTransformer
import json
from datetime import datetime

class SemanticIntentClassifier:
    def __init__(self):
        self.model = None
        self.index = None
        self.intent_examples = {}
        self.intent_labels = []
        self.embedding_dim = 384  # all-MiniLM-L6-v2 dimension
        self.confidence_threshold = 0.7
        
        # Expanded training examples for better classification
        self.training_data = {
            'math': [
                "calculate 15% of 250",
                "what is 45 * 67?",
                "solve for x: 2x + 5 = 15",
                "compute the square root of 144",
                "find the area of a circle with radius 5",
                "what's 1024 divided by 32?",
                "calculate compound interest",
                "solve this equation"
            ],
            'factual': [
                "what is the capital of France?",
                "who invented the telephone?",
                "when did world war 2 end?",
                "what is photosynthesis?",
                "define machine learning",
                "explain quantum physics",
                "who is the current president?",
                "what does API stand for?"
            ],
            'resume': [
                "analyze this resume for technical skills",
                "review my CV and suggest improvements",
                "what experience does this candidate have?",
                "evaluate this resume for a software engineer role",
                "extract skills from this resume",
                "assess candidate qualifications",
                "review work experience section",
                "analyze educational background"
            ],
            'interview': [
                "prepare me for a software engineer interview",
                "what questions should I expect for a data scientist role?",
                "help me practice behavioral interview questions",
                "common interview questions for product manager",
                "how to answer tell me about yourself",
                "prepare for technical interview",
                "mock interview questions",
                "career advice for job interview"
            ],
            'creative': [
                "write a short story about space travel",
                "compose a poem about friendship",
                "create a marketing copy for a new product",
                "write a blog post about artificial intelligence",
                "draft an email to clients",
                "generate creative content ideas",
                "write a product description",
                "create engaging social media posts"
            ],
            'coding': [
                "write a Python function to sort a list",
                "debug this JavaScript code",
                "create a REST API in FastAPI",
                "optimize this SQL query",
                "implement binary search algorithm",
                "review this code for best practices",
                "write unit tests for this function",
                "refactor this code to be more efficient"
            ],
            'analysis': [
                "analyze the pros and cons of remote work",
                "compare different cloud providers",
                "evaluate market trends in AI industry",
                "assess the impact of new regulations",
                "analyze financial performance data",
                "review and summarize this document",
                "provide insights on customer feedback",
                "evaluate strategic options"
            ]
        }
    
    async def initialize(self):
        """Initialize the semantic classifier"""
        logging.info("Initializing semantic intent classifier...")
        
        try:
            # Load lightweight sentence transformer model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Build FAISS index from training data
            await self.build_index()
            
            logging.info("Semantic classifier initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize semantic classifier: {str(e)}")
            # Fallback to rule-based classification
            self.model = None
    
    async def build_index(self):
        """Build FAISS index from training examples"""
        
        all_examples = []
        all_labels = []
        
        for intent, examples in self.training_data.items():
            all_examples.extend(examples)
            all_labels.extend([intent] * len(examples))
        
        # Generate embeddings
        embeddings = self.model.encode(all_examples)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        self.intent_labels = all_labels
        
        logging.info(f"Built FAISS index with {len(all_examples)} examples")
    
    async def classify_intent(self, text: str, top_k: int = 3) -> Tuple[str, float]:
        """Classify intent using semantic similarity"""
        
        if not self.model or not self.index:
            return "unknown", 0.0
        
        try:
            # Generate embedding for input text
            query_embedding = self.model.encode([text])
            faiss.normalize_L2(query_embedding)
            
            # Search for similar examples
            similarities, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            # Count votes for each intent
            intent_scores = {}
            for i, idx in enumerate(indices[0]):
                intent = self.intent_labels[idx]
                score = similarities[0][i]
                
                if intent not in intent_scores:
                    intent_scores[intent] = []
                intent_scores[intent].append(score)
            
            # Calculate average scores
            intent_avg_scores = {
                intent: np.mean(scores) 
                for intent, scores in intent_scores.items()
            }
            
            # Get best intent
            best_intent = max(intent_avg_scores, key=intent_avg_scores.get)
            best_score = intent_avg_scores[best_intent]
            
            logging.debug(f"Classified '{text[:50]}...' as '{best_intent}' (confidence: {best_score:.3f})")
            
            return best_intent, best_score
            
        except Exception as e:
            logging.error(f"Error in semantic classification: {str(e)}")
            return "unknown", 0.0
    
    async def add_training_example(self, text: str, intent: str):
        """Dynamically add training examples (for continuous learning)"""
        
        if intent not in self.training_data:
            self.training_data[intent] = []
        
        self.training_data[intent].append(text)
        
        # Rebuild index if we have enough new examples
        if len(self.training_data[intent]) % 10 == 0:
            await self.build_index()
            logging.info(f"Rebuilt index with new {intent} examples")
    
    def get_classification_stats(self) -> Dict:
        """Get classification statistics"""
        
        total_examples = sum(len(examples) for examples in self.training_data.values())
        
        return {
            "total_training_examples": total_examples,
            "intents": list(self.training_data.keys()),
            "examples_per_intent": {
                intent: len(examples) 
                for intent, examples in self.training_data.items()
            },
            "model_loaded": self.model is not None,
            "index_size": self.index.ntotal if self.index else 0
        }
