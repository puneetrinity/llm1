# services/optimized_router.py - Smart Model Routing for Your Model Fleet
import re
import logging
from typing import Dict, Any, Optional
from datetime import datetime

class OptimizedModelRouter:
    """Intelligent router optimized for your specific model fleet"""
    
    def __init__(self, ollama_client=None):
        self.ollama_client = ollama_client
        
        # Your optimized model configuration
        self.model_config = {
            # Coding & Debugging - Purpose-built for code
            'qwen2.5-coder:7b-instruct': {
                'intent_specialty': ['coding', 'debug', 'programming'],
                'priority': 1,  # Highest priority for code tasks
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'memory_mb': 4200,
                'description': 'Purpose-built for coding and debugging',
                'strengths': ['code_generation', 'debugging', 'code_review', 'refactoring']
            },
            
            # Analysis & Resume - Higher quality model
            'qwen2.5-coder:7b-instruct-q6_K': {
                'intent_specialty': ['resume', 'analysis', 'review'],
                'priority': 1,  # Highest priority for analysis
                'cost_per_token': 0.00015,  # Slightly higher cost for better quality
                'max_context': 8192,
                'memory_mb': 5200,  # Q6_K uses more memory
                'description': 'Higher quality model for analysis and resume tasks',
                'strengths': ['document_analysis', 'resume_review', 'detailed_analysis', 'research']
            },
            
            # Creative & Interview - Excellent conversationalist
            'llama3:8b-instruct-q4_0': {
                'intent_specialty': ['creative', 'interview', 'conversation'],
                'priority': 1,  # Highest priority for creative tasks
                'cost_per_token': 0.00013,
                'max_context': 8192,
                'memory_mb': 5000,
                'description': 'Excellent for creative and conversational tasks',
                'strengths': ['creative_writing', 'interviews', 'storytelling', 'brainstorming']
            },
            
            # Math & Factual - Precision specialist
            'mistral:7b-instruct-q4_0': {
                'intent_specialty': ['math', 'factual', 'calculation'],
                'priority': 1,  # Highest priority for factual tasks
                'cost_per_token': 0.0001,  # Most cost-effective
                'max_context': 8192,
                'memory_mb': 4500,
                'description': 'Great for precision tasks and factual queries',
                'strengths': ['mathematics', 'factual_queries', 'calculations', 'precise_answers']
            }
        }
        
        # Enhanced intent classification patterns
        self.intent_patterns = {
            'coding': [
                r'\b(?:code|function|algorithm|debug|program|script|python|javascript|java|c\+\+|html|css|sql|api|class|method|variable|loop|array|object)\b',
                r'\b(?:write.*(?:code|function|script)|create.*(?:program|app|website)|build.*(?:api|application)|implement.*(?:algorithm|feature))\b',
                r'\b(?:fix.*(?:bug|error|issue)|debug.*(?:code|program)|optimize.*(?:code|performance)|refactor.*code)\b',
                r'\b(?:programming|development|software|coding|scripting|frontend|backend|fullstack)\b'
            ],
            
            'debug': [
                r'\b(?:debug|error|bug|exception|traceback|stack.*trace|syntax.*error|runtime.*error)\b',
                r'\b(?:fix|solve|troubleshoot|diagnose).*(?:error|bug|issue|problem)\b',
                r'\b(?:not working|broken|failing|crashes|throws.*error)\b'
            ],
            
            'resume': [
                r'\b(?:resume|cv|curriculum.*vitae|job.*application|career)\b',
                r'\b(?:experience|skills|qualifications|education|employment|work.*history)\b',
                r'\b(?:review.*(?:resume|cv)|analyze.*(?:resume|cv)|improve.*(?:resume|cv))\b',
                r'\b(?:hiring|recruitment|job.*search|career.*advice)\b'
            ],
            
            'analysis': [
                r'\b(?:analyze|analysis|review|evaluate|assess|examine|study|research)\b',
                r'\b(?:compare|contrast|pros.*cons|advantages.*disadvantages|strengths.*weaknesses)\b',
                r'\b(?:report|summary|findings|conclusions|recommendations|insights)\b',
                r'\b(?:data.*analysis|market.*analysis|business.*analysis|technical.*review)\b'
            ],
            
            'creative': [
                r'\b(?:write|create|compose|generate).*(?:story|poem|article|blog|content|copy)\b',
                r'\b(?:creative|imagination|artistic|original|innovative|brainstorm)\b',
                r'\b(?:storytelling|narrative|fiction|novel|screenplay|script)\b',
                r'\b(?:marketing.*copy|advertisement|slogan|tagline|brand.*message)\b'
            ],
            
            'interview': [
                r'\b(?:interview|job.*interview|hiring.*process|behavioral.*questions)\b',
                r'\b(?:prepare.*(?:interview|job)|practice.*(?:interview|questions))\b',
                r'\b(?:interview.*questions|tell me about|describe.*time|give.*example)\b',
                r'\b(?:career.*advice|job.*tips|interview.*tips|professional.*development)\b'
            ],
            
            'math': [
                r'\b(?:calculate|compute|solve|equation|formula|mathematics|algebra|geometry|calculus)\b',
                r'\b(?:\d+\s*[\+\-\*\/\%\^]\s*\d+|math.*problem|mathematical|arithmetic)\b',
                r'\b(?:statistics|probability|percentage|ratio|proportion|average|mean|median)\b',
                r'\b(?:what.*is.*\d+|how.*much|how.*many|find.*value|solve.*for)\b'
            ],
            
            'factual': [
                r'\b(?:what.*is|who.*is|when.*did|where.*is|how.*does|why.*does)\b',
                r'\b(?:define|explain|describe|tell.*about|information.*about)\b',
                r'\b(?:fact|facts|truth|correct|accurate|precise|exact)\b',
                r '\b(?:history|geography|science|knowledge|encyclopedia|reference)\b'
            ]
        }
        
        # Intent hierarchy for overlapping classifications
        self.intent_priority = {
            'coding': 10,     # Highest priority
            'debug': 9,       # High priority 
            'math': 8,        # High priority for calculations
            'resume': 7,      # High priority for professional docs
            'analysis': 6,    # Medium-high priority
            'interview': 5,   # Medium priority
            'creative': 4,    # Medium priority
            'factual': 3      # Lower priority (catch-all)
        }
        
        self.loaded_models = set()
        self.model_performance_stats = {}
        
    async def initialize(self):
        """Initialize the optimized router"""
        await self.ollama_client.initialize()
        
        # Check which of your models are available
        available_models = await self.ollama_client.list_models()
        available_model_names = {model['name'] for model in available_models}
        
        # Filter to only your configured models that are available
        self.available_models = {
            name: config for name, config in self.model_config.items()
            if name in available_model_names
        }
        
        if not self.available_models:
            logging.warning("None of the optimized models are available!")
            # Add fallback logic here if needed
        
        logging.info(f"🎯 Optimized router initialized with models: {list(self.available_models.keys())}")
    
    def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
        """Enhanced intent classification with your model specialties"""
        
        if explicit_intent and explicit_intent in self.intent_priority:
            return explicit_intent
        
        text_lower = text.lower()
        
        # Score each intent based on pattern matches
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                score += matches
            
            if score > 0:
                # Apply priority weighting
                weighted_score = score * self.intent_priority.get(intent, 1)
                intent_scores[intent] = weighted_score
        
        if intent_scores:
            # Return the intent with the highest weighted score
            best_intent = max(intent_scores, key=intent_scores.get)
            logging.debug(f"Intent classification: '{best_intent}' (score: {intent_scores[best_intent]})")
            return best_intent
        
        # Fallback heuristics for edge cases
        word_count = len(text.split())
        
        # Code-related keywords check
        code_keywords = ['function', 'variable', 'class', 'import', 'return', 'if', 'else', 'for', 'while']
        if any(keyword in text_lower for keyword in code_keywords):
            return 'coding'
        
        # Math-related patterns
        if re.search(r'\d+.*[\+\-\*\/].*\d+', text) or any(word in text_lower for word in ['calculate', 'solve', 'equation']):
            return 'math'
        
        # Length-based heuristics
        if word_count < 10:
            return 'factual'  # Short queries are usually factual
        elif word_count > 100:
            return 'analysis'  # Long queries often need analysis
        
        return 'factual'  # Default fallback
    
    async def route_request(self, request) -> str:
        """Route request to the optimal model based on intent"""
        
        # Use explicit model if it's one of your configured models
        if hasattr(request, 'model') and request.model in self.available_models:
            return request.model
        
        # Extract text content for classification
        text_content = self._extract_text_content(request)
        
        # Classify intent
        intent = self.classify_intent(text_content, getattr(request, 'intent', None))
        
        # Select optimal model for this intent
        selected_model = self._select_optimal_model(intent, text_content, request)
        
        logging.info(f"🎯 Smart routing: intent='{intent}' → model='{selected_model}'")
        return selected_model
    
    def _select_optimal_model(self, intent: str, text: str, request) -> str:
        """Select the optimal model for the classified intent"""
        
        # First, try to find models specialized for this intent
        specialized_models = []
        for model_name, config in self.available_models.items():
            if intent in config['intent_specialty']:
                specialized_models.append((model_name, config))
        
        if specialized_models:
            # Sort by priority (lower number = higher priority)
            specialized_models.sort(key=lambda x: x[1]['priority'])
            selected_model = specialized_models[0][0]
            
            # Log the reasoning
            config = specialized_models[0][1]
            logging.debug(f"Selected {selected_model} for {intent}: {config['description']}")
            
            return selected_model
        
        # Fallback: select based on intent mapping to your models
        intent_to_model = {
            'coding': 'qwen2.5-coder:7b-instruct',
            'debug': 'qwen2.5-coder:7b-instruct',
            'programming': 'qwen2.5-coder:7b-instruct',
            
            'resume': 'qwen2.5-coder:7b-instruct-q6_K',
            'analysis': 'qwen2.5-coder:7b-instruct-q6_K',
            'review': 'qwen2.5-coder:7b-instruct-q6_K',
            
            'creative': 'llama3:8b-instruct-q4_0',
            'interview': 'llama3:8b-instruct-q4_0',
            'conversation': 'llama3:8b-instruct-q4_0',
            
            'math': 'mistral:7b-instruct-q4_0',
            'factual': 'mistral:7b-instruct-q4_0',
            'calculation': 'mistral:7b-instruct-q4_0'
        }
        
        preferred_model = intent_to_model.get(intent)
        if preferred_model and preferred_model in self.available_models:
            return preferred_model
        
        # Final fallback: use any available model, preferring Mistral for general tasks
        fallback_preference = [
            'mistral:7b-instruct-q4_0',  # Most cost-effective
            'llama3:8b-instruct-q4_0',   # Good general model
            'qwen2.5-coder:7b-instruct', # Coding specialist
            'qwen2.5-coder:7b-instruct-q6_K'  # Analysis specialist
        ]
        
        for model in fallback_preference:
            if model in self.available_models:
                logging.warning(f"Using fallback model {model} for intent '{intent}'")
                return model
        
        # Absolute fallback: first available model
        if self.available_models:
            fallback_model = list(self.available_models.keys())[0]
            logging.warning(f"Using absolute fallback model {fallback_model}")
            return fallback_model
        
        # This should not happen if models are properly configured
        raise Exception("No models available for routing!")
    
    def _extract_text_content(self, request) -> str:
        """Extract text content from request for classification"""
        if hasattr(request, 'messages') and request.messages:
            # Extract from chat messages
            return ' '.join(
                msg.content for msg in request.messages 
                if hasattr(msg, 'content') and hasattr(msg, 'role') and msg.role == 'user'
            )
        elif hasattr(request, 'prompt'):
            # Extract from completion prompt
            return request.prompt
        else:
            return ""
    
    async def get_available_models(self):
        """Get list of available models with enhanced metadata"""
        models = []
        for model_name, config in self.available_models.items():
            model_info = {
                "id": model_name,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "ollama",
                
                # Enhanced metadata
                "description": config.get('description', ''),
                "specialties": config.get('intent_specialty', []),
                "strengths": config.get('strengths', []),
                "cost_per_token": config.get('cost_per_token', 0.0001),
                "max_context": config.get('max_context', 8192),
                "memory_mb": config.get('memory_mb', 4000),
                "priority": config.get('priority', 2),
                "loaded": model_name in self.loaded_models
            }
            models.append(model_info)
        
        return models
    
    async def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics and recommendations"""
        
        return {
            "available_models": list(self.available_models.keys()),
            "model_specialties": {
                model: config['intent_specialty'] 
                for model, config in self.model_config.items()
            },
            "intent_patterns": list(self.intent_patterns.keys()),
            "routing_strategy": {
                "coding_tasks": "qwen2.5-coder:7b-instruct",
                "analysis_tasks": "qwen2.5-coder:7b-instruct-q6_K", 
                "creative_tasks": "llama3:8b-instruct-q4_0",
                "factual_tasks": "mistral:7b-instruct-q4_0"
            },
            "performance_stats": self.model_performance_stats
        }
    
    def track_model_performance(self, model: str, response_time: float, success: bool, intent: str):
        """Track model performance for optimization"""
        if model not in self.model_performance_stats:
            self.model_performance_stats[model] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0,
                'intent_distribution': {}
            }
        
        stats = self.model_performance_stats[model]
        stats['total_requests'] += 1
        stats['total_response_time'] += response_time
        
        if success:
            stats['successful_requests'] += 1
        
        # Track intent distribution
        if intent not in stats['intent_distribution']:
            stats['intent_distribution'][intent] = 0
        stats['intent_distribution'][intent] += 1
