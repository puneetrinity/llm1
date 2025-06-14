# services/enhanced_imports.py - Fixed Version

import logging
from typing import Dict, Any, Optional, Type
from dataclasses import dataclass
from enum import Enum

class FeatureStatus(Enum):
    AVAILABLE = "available"
    FALLBACK = "fallback"
    DISABLED = "disabled"

@dataclass
class ImportResult:
    status: FeatureStatus
    module: Optional[Any] = None
    error: Optional[str] = None

class EnhancedImportManager:
    """FIXED import manager - no more silent failures"""
    
    def __init__(self):
        self.imports: Dict[str, ImportResult] = {}
        self.failed_critical: list = []
        
    def safe_import(
        self, 
        feature_name: str, 
        import_path: str, 
        fallback_class: Optional[Type] = None,
        critical: bool = False
    ) -> ImportResult:
        """Import with explicit fallback handling - NO SILENT FAILURES"""
        
        try:
            module_path, class_name = import_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            imported_class = getattr(module, class_name)
            
            result = ImportResult(
                status=FeatureStatus.AVAILABLE,
                module=imported_class
            )
            logging.info(f"✅ {feature_name} loaded successfully")
            
        except ImportError as e:
            error_msg = f"Missing dependency for {feature_name}: {str(e)}"
            
            if critical:
                logging.error(f"❌ CRITICAL: {error_msg}")
                self.failed_critical.append(error_msg)
                result = ImportResult(status=FeatureStatus.DISABLED, error=error_msg)
            elif fallback_class:
                result = ImportResult(
                    status=FeatureStatus.FALLBACK,
                    module=fallback_class,
                    error=error_msg
                )
                logging.warning(f"⚠️ {feature_name} using fallback: {error_msg}")
            else:
                result = ImportResult(status=FeatureStatus.DISABLED, error=error_msg)
                logging.info(f"ℹ️ {feature_name} disabled (optional): {error_msg}")
                
        except Exception as e:
            error_msg = f"Error loading {feature_name}: {str(e)}"
            result = ImportResult(status=FeatureStatus.DISABLED, error=error_msg)
            
            if critical:
                self.failed_critical.append(error_msg)
                logging.error(f"❌ CRITICAL: {error_msg}")
            else:
                logging.error(f"❌ {error_msg}")
        
        self.imports[feature_name] = result
        return result
    
    def validate_startup(self) -> bool:
        """Check if all critical features loaded - FAIL FAST if not"""
        if self.failed_critical:
            logging.error("🚨 CRITICAL FEATURES FAILED:")
            for failure in self.failed_critical:
                logging.error(f"   {failure}")
            return False
        return True
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get clear status - no confusion about what's working"""
        return {
            'available': [name for name, result in self.imports.items() 
                         if result.status == FeatureStatus.AVAILABLE],
            'fallback': [name for name, result in self.imports.items() 
                        if result.status == FeatureStatus.FALLBACK],
            'disabled': [name for name, result in self.imports.items() 
                        if result.status == FeatureStatus.DISABLED],
            'ready_for_production': len(self.failed_critical) == 0
        }
    
    def log_startup_summary(self):
        """Log what's actually working"""
        status = self.get_feature_status()
        
        logging.info("🎯 Feature Status:")
        if status['available']:
            logging.info(f"✅ Available: {', '.join(status['available'])}")
        if status['fallback']:
            logging.info(f"⚠️ Using fallback: {', '.join(status['fallback'])}")
        if status['disabled']:
            logging.info(f"⏸️ Disabled: {', '.join(status['disabled'])}")
        
        if status['ready_for_production']:
            logging.info("✅ Ready for production")
        else:
            logging.error("❌ NOT ready for production - critical features missing")

# Initialize with new manager
import_manager = EnhancedImportManager()

# Setup enhanced imports
def setup_enhanced_imports():
    """Setup imports with proper error handling"""
    
    # Import fallback classes (base versions)
    from services.ollama_client import OllamaClient
    from services.enhanced_router import EnhancedLLMRouter  # FIXED: correct fallback path
    
    # Try enhanced client
    enhanced_ollama = import_manager.safe_import(
        'enhanced_ollama_client',
        'services.enhanced_ollama_client.EnhancedOllamaClient',
        fallback_class=OllamaClient,
        critical=False
    )

    # Try enhanced router
    enhanced_router = import_manager.safe_import(
        'enhanced_router',
        'services.enhanced_router.EnhancedLLMRouter',
        fallback_class=EnhancedLLMRouter,
        critical=False
    )

    # Streaming Service
    streaming_service = import_manager.safe_import(
        'streaming_service',
        'services.streaming.StreamingService',
        critical=False
    )

    # Model Warmup Service  
    warmup_service = import_manager.safe_import(
        'model_warmup_service',
        'services.model_warmup.ModelWarmupService',
        critical=False
    )

    # Semantic Classifier
    semantic_classifier = import_manager.safe_import(
        'semantic_classifier',
        'services.semantic_classifier.SemanticIntentClassifier',
        critical=False
    )
    
    # Final validation
    if not import_manager.validate_startup():
        raise RuntimeError("Critical features failed to load - cannot start")

    return {
        'EnhancedOllamaClient': enhanced_ollama.module,
        'LLMRouter': enhanced_router.module,
        'StreamingService': streaming_service.module,
        'ModelWarmupService': warmup_service.module,
        'SemanticIntentClassifier': semantic_classifier.module,
        'capabilities': {
            'semantic_classification': semantic_classifier.status == FeatureStatus.AVAILABLE,
            'streaming': streaming_service.status == FeatureStatus.AVAILABLE,
            'model_warmup': warmup_service.status == FeatureStatus.AVAILABLE,
            'enhanced_ollama': enhanced_ollama.status == FeatureStatus.AVAILABLE,
            'enhanced_router': enhanced_router.status == FeatureStatus.AVAILABLE,
        }
    }

# Optional flag to denote the module is present
ENHANCED_IMPORTS_AVAILABLE = True
