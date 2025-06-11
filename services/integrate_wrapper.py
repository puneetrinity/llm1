# Replace the downloaded integrate_wrapper.py with the correct Python version
cat > integrate_wrapper.py << 'EOF'
#!/usr/bin/env python3
"""Safe integration of semantic routing with existing system"""

import sys
import shutil
import logging
from pathlib import Path

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    setup_logging()
    
    print("🔧 Integrating Semantic Routing (Safe)")
    print("=" * 40)
    
    app_dir = Path("..")
    services_dir = app_dir / "services"
    services_dir.mkdir(exist_ok=True)
    
    try:
        # Copy semantic classifier
        if Path("semantic_classifier.py").exists():
            dst = services_dir / "semantic_classifier.py"
            if dst.exists():
                shutil.copy(dst, dst.with_suffix('.backup'))
                print("📋 Backed up existing semantic_classifier.py")
            shutil.copy("semantic_classifier.py", dst)
            print("✅ Added semantic_classifier.py")
        else:
            print("❌ semantic_classifier.py not found")
            return 1
        
        # Copy semantic enhanced router
        if Path("semantic_enhanced_router.py").exists():
            dst = services_dir / "semantic_enhanced_router.py"
            if dst.exists():
                shutil.copy(dst, dst.with_suffix('.backup'))
                print("📋 Backed up existing semantic_enhanced_router.py")
            shutil.copy("semantic_enhanced_router.py", dst)
            print("✅ Added semantic_enhanced_router.py")
        else:
            print("❌ semantic_enhanced_router.py not found")
            return 1
        
        # Update main.py safely
        main_file = app_dir / "main.py"
        if main_file.exists():
            with open(main_file, 'r') as f:
                content = f.read()
            
            if "semantic_enhanced_router" not in content:
                # Backup main.py
                shutil.copy(main_file, main_file.with_suffix('.backup'))
                print("📋 Backed up main.py")
                
                # Add import for semantic router
                semantic_import = """
# Enhanced semantic routing
try:
    from services.semantic_enhanced_router import EnhancedLLMRouter as SemanticEnhancedRouter
    SEMANTIC_ROUTING_AVAILABLE = True
    logging.info("✅ Semantic enhanced routing available")
except ImportError as e:
    logging.warning(f"Semantic routing not available: {e}")
    SEMANTIC_ROUTING_AVAILABLE = False
    SemanticEnhancedRouter = None
"""
                
                # Find where to insert the import (after existing imports)
                lines = content.split('\n')
                import_index = -1
                
                for i, line in enumerate(lines):
                    if "from services" in line and "import" in line:
                        import_index = i + 1
                        break
                    elif "import logging" in line:
                        import_index = i + 1
                
                if import_index > 0:
                    lines.insert(import_index, semantic_import)
                
                # Find and replace router initialization
                content = '\n'.join(lines)
                
                # Look for various router initialization patterns
                router_patterns = [
                    ("llm_router = EnhancedLLMRouter(ollama_client)", "enhanced_router"),
                    ("llm_router = LLMRouter(ollama_client)", "base_router"),
                    ("router = EnhancedLLMRouter(ollama_client)", "enhanced_router"),
                    ("router = LLMRouter(ollama_client)", "base_router")
                ]
                
                router_replaced = False
                for pattern, router_type in router_patterns:
                    if pattern in content:
                        if router_type == "enhanced_router":
                            replacement = f"""        # Use semantic enhanced router if available
        if SEMANTIC_ROUTING_AVAILABLE:
            base_router = EnhancedLLMRouter(ollama_client)
            llm_router = SemanticEnhancedRouter(ollama_client, base_router)
            logging.info("🧠 Using Semantic Enhanced Router with base EnhancedLLMRouter")
        else:
            {pattern}
            logging.info("🔧 Using standard EnhancedLLMRouter")"""
                        else:
                            replacement = f"""        # Use semantic enhanced router if available
        if SEMANTIC_ROUTING_AVAILABLE:
            base_router = LLMRouter(ollama_client)
            llm_router = SemanticEnhancedRouter(ollama_client, base_router)
            logging.info("🧠 Using Semantic Enhanced Router with base LLMRouter")
        else:
            {pattern}
            logging.info("🔧 Using standard LLMRouter")"""
                        
                        content = content.replace(pattern, replacement)
                        router_replaced = True
                        break
                
                if not router_replaced:
                    print("⚠️  Could not find router initialization pattern to replace")
                    print("   You may need to manually integrate the semantic router")
                
                # Write updated content
                with open(main_file, 'w') as f:
                    f.write(content)
                
                print("✅ Enhanced main.py with semantic routing")
            else:
                print("ℹ️  main.py already has semantic routing")
        else:
            print("❌ main.py not found")
            return 1
        
        print("\n🎉 Integration Complete!")
        print("=" * 40)
        print("✅ Semantic classifier integrated")
        print("✅ Semantic enhanced router integrated")
        print("✅ main.py enhanced safely")
        print("✅ Backups created (.backup files)")
        
        print("\n🎯 What you get:")
        print("• Coding queries → DeepSeek Coder")
        print("• Resume analysis → Llama3 8B")
        print("• Interview prep → Llama3 8B")
        print("• Creative tasks → Mistral 7B")
        print("• Automatic fallback if semantic fails")
        
        print("\n🚀 Next steps:")
        print("1. cd .. (go back to app directory)")
        print("2. python3 main.py")
        print("3. Watch for '🧠 Using Semantic Enhanced Router' in logs")
        
        return 0
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

# Make it executable
chmod +x integrate_wrapper.py

# Now run the correct Python script
python3 integrate_wrapper.py
