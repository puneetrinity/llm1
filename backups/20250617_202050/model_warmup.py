# services/model_warmup.py - Model Warmup Service (Fixed)
import asyncio
import logging
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any
import random

class ModelWarmupService:
    def __init__(self, ollama_client, router):
        self.ollama_client = ollama_client
        self.router = router
        self.warmup_interval = 300  # 5 minutes
        self.warmup_task = None
        self.model_last_used = {}
        self.model_priorities = {
            'mistral:7b-instruct-q4_0': 1,      # Highest priority (most used)
            'deepseek-v2:7b-q4_0': 2,           # Medium priority
            'llama3:8b-instruct-q4_0': 2        # Medium priority
        }
        
        # Warmup prompts for each model type
        self.warmup_prompts = {
            'mistral:7b-instruct-q4_0': [
                "What is 2+2?",
                "Hello, how are you?",
                "What is the capital of France?"
            ],
            'deepseek-v2:7b-q4_0': [
                "Analyze the skills in this resume: Python, React, AWS",
                "Review this code: def hello(): return 'world'",
                "Evaluate this candidate's experience"
            ],
            'llama3:8b-instruct-q4_0': [
                "Tell me about career opportunities in AI",
                "Prepare me for a software engineer interview",
                "Write a brief story about innovation"
            ]
        }
    
    async def start_warmup_service(self):
        """Start the model warmup background service"""
        logging.info("Starting model warmup service...")
        
        # Initial warmup
        await self.initial_warmup()
        
        # Start periodic warmup task
        self.warmup_task = asyncio.create_task(self.periodic_warmup())
        
        logging.info("Model warmup service started")
    
    async def stop_warmup_service(self):
        """Stop the warmup service"""
        if self.warmup_task:
            self.warmup_task.cancel()
            try:
                await self.warmup_task
            except asyncio.CancelledError:
                pass
        
        logging.info("Model warmup service stopped")
    
    async def initial_warmup(self):
        """Perform initial warmup of priority models"""
        
        priority_models = sorted(
            self.model_priorities.items(),
            key=lambda x: x[1]  # Sort by priority
        )
        
        for model, priority in priority_models:
            try:
                await self.warmup_model(model)
                await asyncio.sleep(2)  # Small delay between warmups
            except Exception as e:
                logging.warning(f"Failed to warmup {model}: {str(e)}")
    
    async def periodic_warmup(self):
        """Periodically warmup models to keep them hot"""
        
        while True:
            try:
                await asyncio.sleep(self.warmup_interval)
                
                # Determine which models need warmup
                models_to_warmup = self.get_models_needing_warmup()
                
                # Warmup models that haven't been used recently
                for model in models_to_warmup:
                    try:
                        await self.warmup_model(model)
                        await asyncio.sleep(1)  # Small delay
                    except Exception as e:
                        logging.warning(f"Periodic warmup failed for {model}: {str(e)}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in periodic warmup: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_models_needing_warmup(self) -> List[str]:
        """Determine which models need warmup based on usage patterns"""
        
        current_time = datetime.now()
        models_to_warmup = []
        
        for model in self.model_priorities.keys():
            last_used = self.model_last_used.get(model)
            
            if not last_used:
                # Never warmed up
                models_to_warmup.append(model)
            else:
                # Check if it's been too long since last warmup
                time_since_use = current_time - last_used
                warmup_threshold = timedelta(minutes=self.warmup_interval / 60)
                
                if time_since_use > warmup_threshold:
                    models_to_warmup.append(model)
        
        # Prioritize by usage priority
        models_to_warmup.sort(key=lambda m: self.model_priorities.get(m, 999))
        
        return models_to_warmup
    
    async def warmup_model(self, model: str):
        """Warmup a specific model with a small request"""
        
        logging.debug(f"Warming up model: {model}")
        
        try:
            # Ensure model is loaded via router if available
            if hasattr(self.router, 'ensure_model_loaded'):
                await self.router.ensure_model_loaded(model)
            
            # Get appropriate warmup prompt
            prompts = self.warmup_prompts.get(model, ["Hello"])
            warmup_prompt = random.choice(prompts)
            
            # Create minimal request
            warmup_request = {
                "model": model,
                "messages": [{"role": "user", "content": warmup_prompt}],
                "stream": False,
                "options": {
                    "num_predict": 10,  # Very short response
                    "temperature": 0.1
                }
            }
            
            # Send warmup request
            if hasattr(self.ollama_client, 'session') and self.ollama_client.session:
                async with self.ollama_client.session.post(
                    f"{self.ollama_client.base_url}/api/chat",
                    json=warmup_request,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        await response.json()  # Consume response
                        self.model_last_used[model] = datetime.now()
                        logging.debug(f"Successfully warmed up {model}")
                    else:
                        logging.warning(f"Warmup request failed for {model}: {response.status}")
            else:
                # Fallback: use the ollama client's generate_completion method
                messages = [{"role": "user", "content": warmup_prompt}]
                await self.ollama_client.generate_completion(
                    model=model,
                    messages=messages,
                    max_tokens=10,
                    temperature=0.1
                )
                self.model_last_used[model] = datetime.now()
                logging.debug(f"Successfully warmed up {model}")
                    
        except Exception as e:
            logging.error(f"Error warming up {model}: {str(e)}")
    
    def record_model_usage(self, model: str):
        """Record when a model was actually used (not just warmed up)"""
        self.model_last_used[model] = datetime.now()
    
    def get_warmup_stats(self) -> Dict[str, Any]:
        """Get warmup service statistics"""
        
        current_time = datetime.now()
        
        stats = {
            "warmup_interval_minutes": self.warmup_interval / 60,
            "models_tracked": list(self.model_priorities.keys()),
            "last_warmup_times": {},
            "time_since_last_warmup": {}
        }
        
        for model, last_used in self.model_last_used.items():
            stats["last_warmup_times"][model] = last_used.isoformat()
            time_diff = current_time - last_used
            stats["time_since_last_warmup"][model] = time_diff.total_seconds()
        
        return stats
