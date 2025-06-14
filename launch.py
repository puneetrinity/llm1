#!/usr/bin/env python3
"""
launch.py - Enhanced 4-Model LLM Proxy Launcher
Start your advanced AI routing system with all features enabled
"""

import os
import sys
import signal
import subprocess
import time

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully"""
    print("\n\n🛑 Shutting down Enhanced 4-Model LLM Proxy...")
    print("Thanks for using the system!")
    sys.exit(0)

def print_startup_banner():
    """Print system startup information"""
    banner = """
🚀 ═══════════════════════════════════════════════════════════════════════════════ 🚀
                        ENHANCED 4-MODEL LLM PROXY SYSTEM
                                    
    🤖 Your AI Models:
    ├── 🧠 Phi-4 Reasoning     → Complex math, logic, scientific analysis
    ├── 🎨 Llama3 8B-Instruct → Creative writing, conversations, storytelling  
    ├── ⚙️  Gemma 7B           → Technical documentation, coding, programming
    └── ⚡ Mistral 7B         → Quick facts, summaries, efficient responses
    
    ✨ Enhanced Features Active:
    ├── 🧠 Smart content-based routing
    ├── 📊 Real-time performance monitoring
    ├── 🔄 Intelligent caching system
    ├── 🎯 Semantic intent classification
    └── 📈 Advanced analytics dashboard
    
🚀 ═══════════════════════════════════════════════════════════════════════════════ 🚀
"""
    print(banner)

def check_dependencies():
    """Check if required files exist"""
    required_files = ["main_master.py"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please ensure you have run the setup script first.")
        return False
    
    return True

def setup_environment():
    """Configure environment variables for enhanced features"""
    env_config = {
        # Authentication
        "ENABLE_AUTH": "false",
        
        # WebSocket features
        "ENABLE_WEBSOCKET": "true",
        "ENABLE_WEBSOCKET_DASHBOARD": "true",
        
        # Enhanced routing features
        "ENABLE_ENHANCED_ROUTING": "true",
        "ENABLE_SEMANTIC_CLASSIFICATION": "true",
        "ENABLE_PERFORMANCE_MONITORING": "true",
        
        # Dashboard features
        "ENABLE_DASHBOARD": "true",
        "ENABLE_REACT_DASHBOARD": "true",
        
        # Performance optimizations
        "ENABLE_CACHING": "true",
        "ENABLE_MODEL_WARMUP": "true",
        
        # Logging
        "LOG_LEVEL": "INFO",
        "ENABLE_DETAILED_METRICS": "true"
    }
    
    # Update environment
    current_env = os.environ.copy()
    current_env.update(env_config)
    
    return current_env

def print_access_info():
    """Print access information for the user"""
    print("\n📊 ACCESS POINTS:")
    print("├── 🌐 Main Dashboard:    http://localhost:8001/app/")
    print("├── 🔗 API Endpoint:      http://localhost:8001/v1/chat/completions")  
    print("├── 📈 Health Check:      http://localhost:8001/health")
    print("├── 📊 Metrics:           http://localhost:8001/metrics")
    print("├── 🔧 Admin Panel:       http://localhost:8001/admin/status")
    print("└── 📚 API Docs:          http://localhost:8001/docs")
    
    print("\n🧪 TESTING COMMANDS:")
    print("├── python3 test_routing.py     # Test smart routing")
    print("├── curl http://localhost:8001/health    # Quick health check")
    print("└── curl http://localhost:8001/metrics   # System metrics")
    
    print("\n🎯 EXAMPLE QUERIES:")
    print("├── Math: 'Solve equation: 2x² + 5x - 3 = 0'  → Routes to Phi-4")
    print("├── Creative: 'Write a story about AI'         → Routes to Llama3")
    print("├── Coding: 'Create a Python sort function'   → Routes to Gemma")
    print("└── Facts: 'What is the capital of France?'   → Routes to Mistral")

def main():
    """Main launcher function"""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print startup banner
    print_startup_banner()
    
    # Check dependencies
    print("🔍 Checking system dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies found")
    
    # Setup environment
    print("⚙️  Configuring enhanced features...")
    enhanced_env = setup_environment()
    print("✅ Enhanced features configured")
    
    # Print access information
    print_access_info()
    
    print("\n" + "="*80)
    print("🚀 STARTING ENHANCED 4-MODEL LLM PROXY SERVER...")
    print("="*80)
    print("Press Ctrl+C to stop the server")
    print("="*80)
    
    try:
        # Start the main server with enhanced environment
        subprocess.run([
            sys.executable, 
            "main_master.py"
        ], env=enhanced_env)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        
    except FileNotFoundError:
        print("❌ Could not find main_master.py")
        print("Please ensure the setup script has been run successfully.")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Server error: {e}")
        print("Check the logs above for more details.")
        sys.exit(1)
    
    finally:
        print("\n👋 Enhanced 4-Model LLM Proxy shut down successfully")
        print("Thanks for using the system!")

if __name__ == "__main__":
    main()
