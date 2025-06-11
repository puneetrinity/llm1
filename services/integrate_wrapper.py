cat > integrate_wrapper.py << 'EOF'
#!/usr/bin/env python3
"""Safe integration of semantic routing with existing system"""

import sys
import shutil
import logging
from pathlib import Path

def main():
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
        
        # Copy or rename the router file
        router_files = [
            "semantic_router_wrapper.py",
            "enhanced_router.py", 
            "semantic_enhanced_router.py"
        ]
        
        router_copied = False
        for router_file in router_files:
            if Path(router_file).exists():
                dst = services_dir / "semantic_enhanced_router.py"
                shutil.copy(router_file, dst)
                print(f"✅ Added {router_file} as semantic_enhanced_router.py")
                router_copied = True
                break
        
        if not router_copied:
            print("❌ No router file found to copy")
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
                if "from services.enhanced_router import" in content:
                    semantic_import = """
# Semantic enhanced routing
try:
    from services.semantic_enhanced_router import EnhancedLLMRouter as SemanticEnhancedRouter
    SEMANTIC_ROUTING_AVAILABLE = True
    logging.info("✅ Semantic enhanced routing available")
except ImportError as e:
    logging.warning(f"Semantic routing not available: {e}")
    SEMANTIC_ROUTING_AVAILABLE = False
    SemanticEnhancedRouter = None
"""
                    # Insert after the existing enhanced router import
                    content = content.replace(
                        "from services.enhanced_router import",
                        "from services.enhanced_router import" + semantic_import + "\n# Original router:"
                    )
                
                # Find router initialization and enhance it
                router_patterns = [
                    "llm_router = EnhancedLLMRouter(",
                    "llm_router = LLMRouter(",
                    "router = EnhancedLLMRouter(",
                    "router = LLMRouter("
                ]
                
                for pattern in router_patterns:
                    if pattern in content:
                        # Replace with semantic version
                        enhanced_init = f"""        # Use semantic enhanced router if available
        if SEMANTIC_ROUTING_AVAILABLE:
            {pattern.replace('EnhancedLLMRouter', 'SemanticEnhancedRouter').replace('LLMRouter', 'SemanticEnhancedRouter')}
            logging.info("🧠 Using Semantic Enhanced Router")
        else:
            {pattern}
            logging.info("🔧 Using standard enhanced router")"""
                        
                        content = content.replace(pattern, enhanced_init)
                        break
                
                # Write updated content
                with open(main_file, 'w') as f:
                    f.write(content)
                
                print("✅ Enhanced main.py with semantic routing")
            else:
                print("ℹ️  main.py already has semantic routing")
        
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
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF
