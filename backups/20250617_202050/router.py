# services/router.py - Base LLM Router
import re
import asyncio
import logging
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from models.requests import ChatCompletionRequest, CompletionRequest
from models.responses import ChatCompletionResponse, Usage, ChatCompletionChoice
from services.ollama_client import OllamaClient

class LLMRouter:
    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama_client = ollama_client or OllamaClient()
        
        # Model configuration with memory limits
        self.model_config = {
            'mistral:7b-instruct-q4_0': {
                'priority': 1,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'memory_mb': 4500,
                'good_for': ['factual', 'math', 'general']
            },
            'deepseek-v2:7b-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00015,
                'max_context': 4096,
                'memory_mb': 4200,
                'good_for': ['analysis', 'coding', 'resume']
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'memory_mb': 5000,
                'good_for': ['creative', 'interview', 'storytelling']
            }
        }
        
        # Intent patterns for rule-based classification
        self.intent_patterns = {
            'math': r'\b(?:calculate|compute|solve|equation|math|arithmetic|\d+\s*[\+\-\*\/\%]\s*\d+)\b',
            'factual': r'\b(?:what is|who is|when did|where is|define|explain|fact)\b',
            'creative': r'\b(?:write|create|compose|story|poem|creative|imagine|generate)\b',
            'coding': r'\b(?:code|function|algorithm|debug|program|script|python|javascript)\b',
            'resume': r'\b(?:resume|cv|experience|skills|qualifications|work history)\b',
            'interview': r'\b(?:interview|job|career|hiring|prepare)\b',
            'analysis': r'\b(?:analyze|review|evaluate|assess|compare|examine)\b'
        }
        
        self.loaded_models = set()
        self.model_load_times = {}
        
    async def initialize(self):
        """Initialize the router and check available models"""
        await self.ollama_client.initialize()
        
        # Check which models are available
        available_models = await self.ollama_client.list_models()
        available_model_names = {model['name'] for model in available_models}
        
        # Update model config to only include available models
        self.available_models = {
            name: config for name, config in self.model_config.items()
            if name in available_model_names
        }
        
        if not self.available_models:
            logging.warning("No configured models are available in Ollama")
            # Fallback to any available model
            if available_models:
                fallback_model = available_models[0]['name']
                self.available_models[fallback_model] = {
                    'priority': 1,
                    'cost_per_token': 0.0001,
                    'max_context': 4096,
                    'memory_mb': 4000,
                    'good_for': ['general']
                }
                logging.info(f"Using fallback model: {fallback_model}")
        
        logging.info(f"Router initialized with models: {list(self.available_models.keys())}")
    
    async def route_request(self, request: ChatCompletionRequest) -> str:
        """Route request to appropriate model based on content and intent"""
        
        # Use explicit model if valid
        if request.model in self.available_models:
            return request.model
        
        # Extract text content for classification
        text_content = self._extract_text_content(request.messages)
        
        # Classify intent
        intent = self.classify_intent(text_content, request.intent)
        
        # Select model based on intent and other factors
        selected_model = self._select_model(intent, text_content, request)
        
        logging.info(f"Routed request (intent: {intent}) to model: {selected_model}")
        return selected_model
    
    def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
        """Classify the intent of the request"""
        
        if explicit_intent:
            return explicit_intent
        
        text_lower = text.lower()
        
        # Rule-based classification
        for intent, pattern in self.intent_patterns.items():
            if re.search(pattern, text_lower):
                return intent
        
        # Fallback heuristics
        word_count = len(text.split())
        
        if word_count < 10:
            return 'factual'  # Short queries are usually factual
        elif word_count > 100:
            return 'creative'  # Long queries are often creative
        else:
            return 'general'
    
    def _extract_text_content(self, messages: List[Dict[str, Any]]) -> str:
        """Extract text content from messages"""
        return ' '.join(msg.get('content', '') for msg in messages if msg.get('role') == 'user')
    
    def _select_model(self, intent: str, text: str, request: ChatCompletionRequest) -> str:
        """Select the best model for the given intent and requirements"""
        
        # Filter models by capability
        suitable_models = {}
        for model_name, config in self.available_models.items():
            if intent in config['good_for'] or 'general' in config['good_for']:
                suitable_models[model_name] = config
        
        if not suitable_models:
            # Fallback to any available model
            suitable_models = self.available_models
        
        # Consider request requirements
        max_tokens = request.max_tokens or 2048
        
        # Filter by context length
        suitable_models = {
            name: config for name, config in suitable_models.items()
            if config['max_context'] >= max_tokens
        }
        
        if not suitable_models:
            # Fallback if no model meets requirements
            suitable_models = self.available_models
        
        # Select by priority (lower number = higher priority)
        best_model = min(suitable_models.items(), key=lambda x: x[1]['priority'])
        return best_model[0]
    
    async def ensure_model_loaded(self, model: str) -> bool:
        """Ensure a model is loaded and ready"""
        if model in self.loaded_models:
            return True
        
        try:
            # Try to get model info (this loads it if not already loaded)
            model_info = await self.ollama_client.model_info(model)
            if model_info:
                self.loaded_models.add(model)
                self.model_load_times[model] = datetime.now()
                logging.info(f"Model {model} is now loaded")
                return True
            else:
                # Try to pull the model
                logging.info(f"Attempting to pull model: {model}")
                if await self.ollama_client.pull_model(model):
                    self.loaded_models.add(model)
                    self.model_load_times[model] = datetime.now()
                    return True
                else:
                    logging.error(f"Failed to load model: {model}")
                    return False
        except Exception as e:
            logging.error(f"Error loading model {model}: {str(e)}")
            return False
    
    async def process_chat_completion(
        self, 
        request: ChatCompletionRequest, 
        model: str
    ) -> ChatCompletionResponse:
        """Process chat completion with the selected model"""
        
        # Ensure model is loaded
        if not await self.ensure_model_loaded(model):
            raise Exception(f"Failed to load model: {model}")
        
        # Prepare request for Ollama
        ollama_request = {
            "model": model,
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p
            }
        }
        
        if request.max_tokens:
            ollama_request["options"]["num_predict"] = request.max_tokens
        
        # Send request to Ollama
        start_time = datetime.now()
        ollama_response = await self.ollama_client.chat_completion(ollama_request)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert to OpenAI format
        response = self._convert_ollama_response(ollama_response, model, processing_time)
        
        return response
    
    def _convert_ollama_response(
        self, 
        ollama_response: Dict[str, Any], 
        model: str,
        processing_time: float
    ) -> ChatCompletionResponse:
        """Convert Ollama response to OpenAI format"""
        
        message = ollama_response.get('message', {})
        content = message.get('content', '')
        
        # Calculate token usage (rough estimation)
        prompt_tokens = self._estimate_tokens(ollama_response.get('prompt', ''))
        completion_tokens = self._estimate_tokens(content)
        
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
        
        choice = ChatCompletionChoice(
            index=0,
            message={
                "role": "assistant",
                "content": content
            },
            finish_reason="stop"
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=model,
            choices=[choice],
            usage=usage,
            processing_time=processing_time
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (about 0.75 tokens per word for English)"""
        if not text:
            return 0
        return max(1, int(len(text.split()) * 0.75))
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with metadata"""
        
        models = []
        for model_name, config in self.available_models.items():
            model_info = {
                "id": model_name,
                "object": "model",
                "created": int(self.model_load_times.get(model_name, datetime.now()).timestamp()),
                "owned_by": "ollama",
                "cost_per_token": config.get('cost_per_token', 0.0001),
                "max_context": config.get('max_context', 4096),
                "capabilities": config.get('good_for', []),
                "memory_mb": config.get('memory_mb', 4000),
                "loaded": model_name in self.loaded_models
            }
            models.append(model_info)
        
        return models
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.ollama_client.cleanup()
