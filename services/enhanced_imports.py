# STEP 1: Replace services/enhanced_imports.py entirely (30 minutes)

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

class SafeImportManager:
    """Production-safe import manager - no silent failures"""
    
    def __init__(self):
        self.imports: Dict[str, ImportResult] = {}
        self.failed_imports: list = []
        
    def safe_import(
        self, 
        feature_name: str, 
        import_path: str, 
        fallback_class: Optional[Type] = None,
        required: bool = False
    ) -> ImportResult:
        """Import with explicit fallback handling"""
        
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
            error_msg = f"Missing dependency: {str(e)}"
            
            if required:
                # Required features must fail loudly
                logging.error(f"❌ REQUIRED: {feature_name} failed to load: {error_msg}")
                self.failed_imports.append(f"{feature_name}: {error_msg}")
                result = ImportResult(status=FeatureStatus.DISABLED, error=error_msg)
            elif fallback_class:
                # Use fallback
                result = ImportResult(
                    status=FeatureStatus.FALLBACK,
                    module=fallback_class,
                    error=error_msg
                )
                logging.warning(f"⚠️ {feature_name} using fallback: {error_msg}")
            else:
                # Disable feature
                result = ImportResult(status=FeatureStatus.DISABLED, error=error_msg)
                logging.info(f"ℹ️ {feature_name} disabled: {error_msg}")
                
        except Exception as e:
            error_msg = f"Import error: {str(e)}"
            result = ImportResult(status=FeatureStatus.DISABLED, error=error_msg)
            if required:
                self.failed_imports.append(f"{feature_name}: {error_msg}")
            logging.error(f"❌ {feature_name} failed: {error_msg}")
        
        self.imports[feature_name] = result
        return result
    
    def validate_required_features(self) -> bool:
        """Check if all required features loaded successfully"""
        if self.failed_imports:
            logging.error("🚨 Required features failed to load:")
            for failure in self.failed_imports:
                logging.error(f"   {failure}")
            return False
        return True
    
    def get_feature_status(self) -> Dict[str, Any]:
        """Get clear status of all features"""
        return {
            'available': [name for name, result in self.imports.items() 
                         if result.status == FeatureStatus.AVAILABLE],
            'fallback': [name for name, result in self.imports.items() 
                        if result.status == FeatureStatus.FALLBACK],
            'disabled': [name for name, result in self.imports.items() 
                        if result.status == FeatureStatus.DISABLED],
            'errors': {name: result.error for name, result in self.imports.items() 
                      if result.error},
            'ready_for_production': len(self.failed_imports) == 0
        }

# STEP 2: Update services/enhanced_router.py (20 minutes)
# Replace the old import pattern with explicit checks

class EnhancedLLMRouter(BaseRouter):
    def __init__(self, ollama_client=None):
        super().__init__(ollama_client)
        self.semantic_classifier = None
        self.has_semantic_classification = False
        
    async def initialize(self):
        """Initialize with explicit feature checking"""
        await super().initialize()
        
        # Try to initialize semantic classifier
        try:
            # Import here with explicit check
            import_result = import_manager.safe_import(
                'semantic_classifier',
                'services.semantic_classifier.SemanticIntentClassifier',
                required=False
            )
            
            if import_result.status == FeatureStatus.AVAILABLE:
                self.semantic_classifier = import_result.module()
                await self.semantic_classifier.initialize()
                self.has_semantic_classification = True
                logging.info("✅ Semantic classification enabled")
            else:
                self.has_semantic_classification = False
                logging.info("ℹ️ Using rule-based classification only")
                
        except Exception as e:
            logging.warning(f"Semantic classifier initialization failed: {e}")
            self.has_semantic_classification = False
        
        logging.info("Enhanced LLM Router initialized")
    
    def classify_intent(self, text: str, explicit_intent: Optional[str] = None) -> str:
        """Classify intent with explicit fallback"""
        
        if explicit_intent:
            return explicit_intent
        
        # Always try rule-based first (fast and reliable)
        rule_based_intent = self._rule_based_classification(text)
        if rule_based_intent != 'unknown':
            return rule_based_intent
        
        # Only use semantic if available and initialized
        if self.has_semantic_classification and self.semantic_classifier:
            try:
                semantic_intent, confidence = await self.semantic_classifier.classify_intent(text)
                if confidence > 0.7:  # High confidence threshold
                    return semantic_intent
            except Exception as e:
                logging.warning(f"Semantic classification failed: {e}")
                # Don't crash - fall back to heuristic
        
        # Final fallback
        return self._heuristic_classification(text)

# STEP 3: Create startup validation (15 minutes)
# File: utils/startup_validator.py

import os
import sys
from services.enhanced_imports import import_manager

def validate_startup() -> bool:
    """Validate system is ready for production"""
    
    logging.info("🔍 Validating startup configuration...")
    
    # Check required features
    if not import_manager.validate_required_features():
        logging.error("❌ Required features missing - cannot start")
        return False
    
    # Check environment
    env = os.getenv('ENVIRONMENT', 'development')
    if env == 'production':
        if not _validate_production_config():
            return False
    
    # Check Ollama connectivity
    if not _check_ollama_available():
        logging.error("❌ Ollama not available")
        return False
    
    logging.info("✅ Startup validation passed")
    return True

def _validate_production_config() -> bool:
    """Validate production-specific config"""
    
    # Check API key
    api_key = os.getenv('DEFAULT_API_KEY', '')
    if len(api_key) < 20:
        logging.error("❌ Production requires secure API key")
        return False
    
    # Check CORS
    cors_origins = os.getenv('CORS_ORIGINS', '["*"]')
    if '"*"' in cors_origins:
        logging.warning("⚠️ Using wildcard CORS in production")
    
    return True

def _check_ollama_available() -> bool:
    """Check if Ollama is available"""
    import subprocess
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True)
        return result.returncode == 0
    except:
        return False

# STEP 4: Update main.py startup (15 minutes)
# Add this to the beginning of your main.py

async def safe_startup():
    """Safe startup with validation"""
    
    from utils.startup_validator import validate_startup
    
    # Validate before starting anything
    if not validate_startup():
        logging.error("❌ Startup validation failed")
        sys.exit(1)
    
    # Your existing startup code here...

# STEP 5: Update requirements.txt (10 minutes)
# Split into core and optional

# requirements-core.txt (always install these)
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
aiohttp==3.9.1
psutil==5.9.6

# requirements-enhanced.txt (optional)
sentence-transformers==2.2.2
faiss-cpu==1.7.4
sse-starlette==1.6.5

# STEP 6: Update Dockerfile (20 minutes)
# Make installation conditional

FROM python:3.11-slim

WORKDIR /app

# Install core dependencies (always)
COPY requirements-core.txt .
RUN pip install --no-cache-dir -r requirements-core.txt

# Install enhanced dependencies (conditional)
COPY requirements-enhanced.txt .
RUN pip install --no-cache-dir -r requirements-enhanced.txt || echo "Enhanced features not available"

# Rest of your Dockerfile...

# Set feature flags based on what's available
RUN python -c "
try:
    import sentence_transformers
    print('ENABLE_SEMANTIC_CLASSIFICATION=true')
except ImportError:
    print('ENABLE_SEMANTIC_CLASSIFICATION=false')
" > /tmp/features.env

# STEP 7: Test the fix (30 minutes)
# Create test script: test_imports.py

def test_import_system():
    """Test the fixed import system"""
    
    from services.enhanced_imports import SafeImportManager
    
    manager = SafeImportManager()
    
    # Test required import (should fail if missing)
    result = manager.safe_import(
        'fastapi_app',
        'fastapi.FastAPI', 
        required=True
    )
    assert result.status == FeatureStatus.AVAILABLE
    
    # Test optional import with fallback
    result = manager.safe_import(
        'sentence_transformers',
        'sentence_transformers.SentenceTransformer',
        fallback_class=object,  # dummy fallback
        required=False
    )
    # Should be either AVAILABLE or FALLBACK, never crash
    assert result.status in [FeatureStatus.AVAILABLE, FeatureStatus.FALLBACK]
    
    # Get status
    status = manager.get_feature_status()
    print("Import Status:", status)
    
    # Should not have any required failures
    assert manager.validate_required_features()

if __name__ == "__main__":
    test_import_system()
    print("✅ Import system test passed!")
