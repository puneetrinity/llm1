# services/enhanced_imports.py - Safe Import Handler for Enhanced Features
import logging
import sys
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass

@dataclass
class FeatureStatus:
    """Track status of enhanced features"""
    name: str
    available: bool
    error: Optional[str] = None
    fallback_class: Optional[Type] = None

class EnhancedImportManager:
    """Manages safe imports of enhanced features with graceful fallbacks"""
    
    def __init__(self):
        self.features: Dict[str, FeatureStatus] = {}
        self.import_errors: Dict[str, str] = {}
        
    def safe_import(self, feature_name: str, import_path: str, fallback_class: Type = None) -> tuple:
        """Safely import enhanced feature with fallback
        
        Returns:
            tuple: (imported_class, is_enhanced)
        """
        try:
            # Split module and class name
            module_path, class_name = import_path.rsplit('.', 1)
            
            # Try to import the module
            module = __import__(module_path, fromlist=[class_name])
            imported_class = getattr(module, class_name)
            
            # Mark as successful
            self.features[feature_name] = FeatureStatus(
                name=feature_name,
                available=True,
                fallback_class=fallback_class
            )
            
            logging.info(f"✅ Enhanced feature '{feature_name}' loaded successfully")
            return imported_class, True
            
        except ImportError as e:
            # Log the specific import error
            error_msg = str(e)
            self.import_errors[feature_name] = error_msg
            
            # Mark as failed and use fallback
            self.features[feature_name] = FeatureStatus(
                name=feature_name,
                available=False,
                error=error_msg,
                fallback_class=fallback_class
            )
            
            if fallback_class:
                logging.info(f"ℹ️  Enhanced feature '{feature_name}' not available, using fallback: {error_msg}")
                return fallback_class, False
            else:
                logging.warning(f"⚠️  Enhanced feature '{feature_name}' not available and no fallback: {error_msg}")
                return None, False
                
        except Exception as e:
            # Handle other errors
            error_msg = f"Unexpected error: {str(e)}"
            self.import_errors[feature_name] = error_msg
            
            self.features[feature_name] = FeatureStatus(
                name=feature_name,
                available=False,
                error=error_msg,
                fallback_class=fallback_class
            )
            
            logging.error(f"❌ Error importing '{feature_name}': {error_msg}")
            
            if fallback_class:
                return fallback_class, False
            else:
                return None, False

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if optional dependencies are available"""
        dependencies = {
            'sentence_transformers': False,
            'faiss': False,
            'numpy': False,
            'sse_starlette': False
        }
        
        for dep in dependencies:
            try:
                __import__(dep.replace('_', '-'))
                dependencies[dep] = True
            except ImportError:
                dependencies[dep] = False
                
        return dependencies
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get status of all enhanced features"""
        dependencies = self.check_dependencies()
        
        return {
            'features': {name: status.available for name, status in self.features.items()},
            'dependencies': dependencies,
            'errors': self.import_errors,
            'total_enhanced_features': sum(1 for s in self.features.values() if s.available),
            'total_fallbacks': sum(1 for s in self.features.values() if not s.available and s.fallback_class),
            'missing_features': [name for name, status in self.features.items() 
                               if not status.available and not status.fallback_class]
        }
    
    def log_startup_summary(self):
        """Log a summary of enhanced features at startup"""
        status = self.get_feature_status()
        
        logging.info("=" * 60)
        logging.info("🎯 Enhanced Features Summary")
        logging.info("=" * 60)
        
        # Enhanced features
        enhanced_count = status['total_enhanced_features']
        if enhanced_count > 0:
            logging.info(f"✅ Enhanced features enabled: {enhanced_count}")
            for name, available in status['features'].items():
                if available:
                    logging.info(f"   • {name.replace('_', ' ').title()}")
        
        # Fallbacks
        fallback_count = status['total_fallbacks']
        if fallback_count > 0:
            logging.info(f"ℹ️  Basic fallbacks active: {fallback_count}")
            for name, feature_status in self.features.items():
                if not feature_status.available and feature_status.fallback_class:
                    logging.info(f"   • {name.replace('_', ' ').title()}")
        
        # Missing features
        missing = status['missing_features']
        if missing:
            logging.info(f"⚠️  Features unavailable: {len(missing)}")
            for name in missing:
                logging.info(f"   • {name.replace('_', ' ').title()}")
        
        # Dependencies
        missing_deps = [dep for dep, available in status['dependencies'].items() if not available]
        if missing_deps:
            logging.info(f"📦 Missing optional dependencies: {', '.join(missing_deps)}")
            logging.info("   Install with: pip install sentence-transformers faiss-cpu sse-starlette")
        
        logging.info("=" * 60)

# Global import manager instance
import_manager = EnhancedImportManager()

# Safe imports for all enhanced features
def setup_enhanced_imports():
    """Setup all enhanced imports with fallbacks"""
    
    # Import core services for fallbacks
    from services.ollama_client import OllamaClient
    from services.router import LLMRouter
    
    # Enhanced Ollama Client
    EnhancedOllamaClient, _ = import_manager.safe_import(
        'enhanced_ollama_client',
        'services.enhanced_ollama_client.EnhancedOllamaClient',
        fallback_class=OllamaClient
    )
    
    # Enhanced Router with Semantic Classification
    EnhancedLLMRouter, semantic_available = import_manager.safe_import(
        'enhanced_router',
        'services.enhanced_router.EnhancedLLMRouter',
        fallback_class=LLMRouter
    )
    
    # Streaming Service
    StreamingService, streaming_available = import_manager.safe_import(
        'streaming_service',
        'services.streaming.StreamingService'
    )
    
    # Model Warmup Service
    ModelWarmupService, warmup_available = import_manager.safe_import(
        'model_warmup_service',
        'services.model_warmup.ModelWarmupService'
    )
    
    # Semantic Classifier
    SemanticIntentClassifier, classifier_available = import_manager.safe_import(
        'semantic_classifier',
        'services.semantic_classifier.SemanticIntentClassifier'
    )
    
    # Semantic Cache
    SemanticCache, cache_available = import_manager.safe_import(
        'semantic_cache',
        'services.semantic_cache.SemanticCache'
    )
    
    return {
        'EnhancedOllamaClient': EnhancedOllamaClient,
        'EnhancedLLMRouter': EnhancedLLMRouter,
        'StreamingService': StreamingService,
        'ModelWarmupService': ModelWarmupService,
        'SemanticIntentClassifier': SemanticIntentClassifier,
        'SemanticCache': SemanticCache,
        'capabilities': {
            'semantic_classification': semantic_available,
            'streaming': streaming_available,
            'model_warmup': warmup_available,
            'semantic_cache': cache_available
        }
    }

# Initialize enhanced imports
try:
    enhanced_imports = setup_enhanced_imports()
    ENHANCED_IMPORTS_AVAILABLE = True
except Exception as e:
    logging.error(f"Failed to setup enhanced imports: {e}")
    enhanced_imports = {}
    ENHANCED_IMPORTS_AVAILABLE = False
