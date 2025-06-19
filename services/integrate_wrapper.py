#!/usr/bin/env python3
"""Safe integration of semantic routing with existing system"""

import sys
import shutil
import logging
from pathlib import Path


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


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
                shutil.copy(dst, dst.with_suffix(".backup"))
                print("📋 Backed up existing semantic_classifier.py")
            shutil.copy("semantic_classifier.py", dst)
            print("✅ Added semantic_classifier.py")
        else:
            print("❌ semantic_classifier.py not found")
            return 1

        # Copy semantic enhanced router
        if Path("semantic_enhanced_router.py").exists():
            dst = services_dir / "semantic_enhanced_router.py"
            shutil.copy("semantic_enhanced_router.py", dst)
            print("✅ Added semantic_enhanced_router.py")
        else:
            print("❌ semantic_enhanced_router.py not found")
            return 1

        # Update main.py safely
        main_file = app_dir / "main.py"
        if main_file.exists():
            with open(main_file, "r") as f:
                content = f.read()

            if "semantic_enhanced_router" not in content:
                # Backup main.py
                shutil.copy(main_file, main_file.with_suffix(".backup"))
                print("📋 Backed up main.py")

                # Add semantic import after existing imports
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

                # Find router import and add semantic import after it
                if "from services.enhanced_router import" in content:
                    content = content.replace(
                        "from services.enhanced_router import",
                        "from services.enhanced_router import"
                        + semantic_import
                        + "\n# Original import:",
                    )
                elif "from services.router import" in content:
                    content = content.replace(
                        "from services.router import",
                        "from services.router import"
                        + semantic_import
                        + "\n# Original import:",
                    )

                # Replace router initialization
                if "llm_router = EnhancedLLMRouter(ollama_client)" in content:
                    replacement = """        # Use semantic enhanced router if available
        if SEMANTIC_ROUTING_AVAILABLE:
            base_router = EnhancedLLMRouter(ollama_client)
            llm_router = SemanticEnhancedRouter(ollama_client, base_router)
            logging.info("🧠 Using Semantic Enhanced Router")
        else:
            llm_router = EnhancedLLMRouter(ollama_client)
            logging.info("🔧 Using standard EnhancedLLMRouter")"""

                    content = content.replace(
                        "llm_router = EnhancedLLMRouter(ollama_client)", replacement
                    )
                elif "llm_router = LLMRouter(ollama_client)" in content:
                    replacement = """        # Use semantic enhanced router if available
        if SEMANTIC_ROUTING_AVAILABLE:
            base_router = LLMRouter(ollama_client)
            llm_router = SemanticEnhancedRouter(ollama_client, base_router)
            logging.info("🧠 Using Semantic Enhanced Router")
        else:
            llm_router = LLMRouter(ollama_client)
            logging.info("🔧 Using standard LLMRouter")"""

                    content = content.replace(
                        "llm_router = LLMRouter(ollama_client)", replacement
                    )

                # Write updated content
                with open(main_file, "w") as f:
                    f.write(content)

                print("✅ Enhanced main.py with semantic routing")
            else:
                print("ℹ️  main.py already has semantic routing")

        print("\n🎉 Integration Complete!")
        print("🚀 Next steps:")
        print("1. cd .. (go back to app directory)")
        print("2. python3 main.py")

        return 0

    except Exception as e:
        print(f"❌ Integration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
