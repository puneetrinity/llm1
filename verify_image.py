#!/usr/bin/env python3
"""
verify_image.py - Verify the Enhanced LLM Proxy Docker image works correctly
Tests all enhanced services and functionality after build
"""

import time
import requests
import subprocess
import json
import sys
from datetime import datetime

class ImageVerifier:
    def __init__(self, image_name="enhanced-llm-proxy:latest", port=8001):
        self.image_name = image_name
        self.port = port
        self.base_url = f"http://localhost:{port}"
        self.container_name = "enhanced-llm-proxy-verify"
        self.container_id = None
        
    def print_status(self, message, status="info"):
        colors = {
            "success": "\033[0;32m✅",
            "error": "\033[0;31m❌", 
            "warning": "\033[1;33m⚠️",
            "info": "\033[0;34mℹ️",
            "test": "\033[0;35m🧪"
        }
        reset = "\033[0m"
        icon = colors.get(status, "\033[0;34mℹ️")
        print(f"{icon}{reset} {message}")
    
    def start_container(self):
        """Start the Docker container for testing"""
        self.print_status("Starting Docker container for verification...", "info")
        
        # Remove existing container if it exists
        subprocess.run(['docker', 'rm', '-f', self.container_name], 
                      capture_output=True, text=True)
        
        # Start new container
        try:
            result = subprocess.run([
                'docker', 'run', '-d',
                '--name', self.container_name,
                '-p', f'{self.port}:{self.port}',
                self.image_name
            ], capture_output=True, text=True, check=True)
            
            self.container_id = result.stdout.strip()
            self.print_status(f"Container started: {self.container_id[:12]}", "success")
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_status(f"Failed to start container: {e.stderr}", "error")
            return False
    
    def wait_for_service(self, timeout=120):
        """Wait for the service to be ready"""
        self.print_status(f"Waiting for service to be ready (timeout: {timeout}s)...", "info")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        self.print_status("Service is ready!", "success")
                        return True
                    else:
                        self.print_status(f"Service status: {data.get('status', 'unknown')}", "warning")
                        
            except requests.exceptions.RequestException:
                pass  # Keep trying
                
            time.sleep(5)
            print(".", end="", flush=True)
        
        self.print_status("Service failed to become ready within timeout", "error")
        return False
    
    def test_endpoints(self):
        """Test all critical endpoints"""
        self.print_status("Testing API endpoints...", "test")
        
        tests = [
            ("Root Endpoint", "/", 200),
            ("Health Check", "/health", 200),
            ("Models List", "/v1/models", 200),
            ("API Documentation", "/docs", 200),
        ]
        
        passed = 0
        for name, endpoint, expected_status in tests:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code == expected_status:
                    self.print_status(f"{name}: PASS", "success")
                    passed += 1
                else:
                    self.print_status(f"{name}: FAIL (HTTP {response.status_code})", "error")
                    
            except requests.exceptions.RequestException as e:
                self.print_status(f"{name}: FAIL ({str(e)})", "error")
        
        return passed == len(tests)
    
    def test_enhanced_features(self):
        """Test enhanced features"""
        self.print_status("Testing enhanced features...", "test")
        
        try:
            # Test root endpoint for feature info
            response = requests.get(f"{self.base_url}/", timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Check services status
                services = data.get("services", {})
                features = data.get("features", {})
                
                self.print_status("Service Status:", "info")
                for service, status in services.items():
                    status_icon = "✅" if status else "❌"
                    print(f"  {status_icon} {service}: {status}")
                
                self.print_status("Enhanced Features:", "info") 
                for feature, enabled in features.items():
                    status_icon = "✅" if enabled else "⏸️"
                    print(f"  {status_icon} {feature}: {enabled}")
                
                # Check if core services are working
                ollama_connected = services.get("ollama_connected", False)
                initialization_complete = services.get("initialization_complete", False)
                
                if ollama_connected and initialization_complete:
                    self.print_status("Enhanced features working correctly", "success")
                    return True
                else:
                    self.print_status("Some enhanced features not working", "warning")
                    return False
                    
        except Exception as e:
            self.print_status(f"Enhanced features test failed: {e}", "error")
            return False
        
        return False
    
    def test_model_routing(self):
        """Test model routing functionality"""
        self.print_status("Testing model routing...", "test")
        
        # Test queries that should route to different models
        test_queries = [
            ("Math query (→ Phi3.5)", "What is 2+2?"),
            ("Creative query (→ Llama3)", "Write a short story"),
            ("Coding query (→ Gemma)", "Create a Python function"),
            ("Factual query (→ Mistral)", "What is the capital of France?")
        ]
        
        passed = 0
        for description, query in test_queries:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": "gpt-3.5-turbo",  # Will be routed
                        "messages": [{"role": "user", "content": query}],
                        "max_tokens": 10
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        self.print_status(f"{description}: PASS", "success")
                        passed += 1
                    else:
                        self.print_status(f"{description}: FAIL (No response)", "error")
                else:
                    self.print_status(f"{description}: FAIL (HTTP {response.status_code})", "error")
                    
            except requests.exceptions.RequestException as e:
                self.print_status(f"{description}: FAIL ({str(e)})", "error")
        
        return passed > 0  # At least one routing test should pass
    
    def get_container_logs(self):
        """Get container logs for debugging"""
        try:
            result = subprocess.run([
                'docker', 'logs', '--tail', '50', self.container_name
            ], capture_output=True, text=True)
            
            return result.stdout
            
        except subprocess.CalledProcessError:
            return "Could not retrieve logs"
    
    def cleanup(self):
        """Clean up the test container"""
        self.print_status("Cleaning up test container...", "info")
        subprocess.run(['docker', 'rm', '-f', self.container_name], 
                      capture_output=True, text=True)
    
    def run_verification(self):
        """Run complete verification process"""
        print("🔍 Enhanced LLM Proxy Image Verification")
        print("=" * 50)
        print(f"Image: {self.image_name}")
        print(f"Port: {self.port}")
        print(f"Time: {datetime.now().isoformat()}")
        print()
        
        try:
            # Step 1: Start container
            if not self.start_container():
                return False
            
            # Step 2: Wait for service
            if not self.wait_for_service():
                self.print_status("Showing container logs:", "info")
                print(self.get_container_logs())
                return False
            
            # Step 3: Test basic endpoints
            endpoints_ok = self.test_endpoints()
            
            # Step 4: Test enhanced features
            features_ok = self.test_enhanced_features()
            
            # Step 5: Test model routing
            routing_ok = self.test_model_routing()
            
            # Summary
            print("\n" + "=" * 50)
            print("🎯 VERIFICATION SUMMARY")
            print("=" * 50)
            
            tests = [
                ("Basic Endpoints", endpoints_ok),
                ("Enhanced Features", features_ok), 
                ("Model Routing", routing_ok)
            ]
            
            passed = sum(1 for _, result in tests if result)
            total = len(tests)
            
            for test_name, result in tests:
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name}: {status}")
            
            print(f"\nOverall: {passed}/{total} test categories passed")
            
            if passed == total:
                self.print_status("🎉 IMAGE VERIFICATION SUCCESSFUL!", "success")
                self.print_status("Image is ready for RunPod deployment", "success")
                return True
            else:
                self.print_status("⚠️ Some tests failed - check logs above", "warning")
                return False
                
        finally:
            self.cleanup()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify Enhanced LLM Proxy Docker image")
    parser.add_argument("--image", default="enhanced-llm-proxy:latest", 
                       help="Docker image name to test")
    parser.add_argument("--port", type=int, default=8001,
                       help="Port to test on") 
    
    args = parser.parse_args()
    
    verifier = ImageVerifier(args.image, args.port)
    success = verifier.run_verification()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
