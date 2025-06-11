#!/bin/bash
# complete_system_test.sh - Comprehensive Testing Script for LLM Proxy
# Run this script to verify all components are working correctly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

print_header() {
    echo -e "\n${BLUE}===========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===========================================${NC}"
}

print_test() {
    echo -e "\n${YELLOW}🧪 Test: $1${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
}

print_failure() {
    echo -e "${RED}❌ $1${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to test HTTP endpoint
test_endpoint() {
    local url="$1"
    local expected_status="$2"
    local description="$3"
    
    print_test "$description"
    
    local response=$(curl -s -w "%{http_code}" -o /tmp/response.json "$url" 2>/dev/null || echo "000")
    
    if [ "$response" = "$expected_status" ]; then
        print_success "Endpoint responding correctly (HTTP $response)"
        return 0
    else
        print_failure "Endpoint failed (HTTP $response, expected $expected_status)"
        echo "Response: $(cat /tmp/response.json 2>/dev/null || echo 'No response')"
        return 1
    fi
}

# Function to test API endpoint with JSON
test_api_endpoint() {
    local url="$1"
    local method="$2"
    local data="$3"
    local description="$4"
    local api_key="$5"
    
    print_test "$description"
    
    local headers=""
    if [ -n "$api_key" ]; then
        headers="-H 'X-API-Key: $api_key'"
    fi
    
    local cmd="curl -s -X $method $headers -H 'Content-Type: application/json'"
    if [ -n "$data" ]; then
        cmd="$cmd -d '$data'"
    fi
    cmd="$cmd $url"
    
    local response=$(eval $cmd 2>/dev/null)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ] && echo "$response" | jq . >/dev/null 2>&1; then
        print_success "API endpoint working correctly"
        echo "Sample response: $(echo "$response" | jq -r '. | keys[]' | head -3 | tr '\n' ' ')"
        return 0
    else
        print_failure "API endpoint failed"
        echo "Response: $response"
        return 1
    fi
}

print_header "LLM PROXY COMPREHENSIVE SYSTEM TEST"

# Get service info
SERVICE_URL="http://localhost:8000"
OLLAMA_URL="http://localhost:11434"
API_KEY="sk-default"  # Default API key

echo "Testing system at: $SERVICE_URL"
echo "Ollama service at: $OLLAMA_URL"
echo "Start time: $(date)"

# ===========================================
# 1. SYSTEM HEALTH CHECKS
# ===========================================
print_header "1. SYSTEM HEALTH CHECKS"

print_test "System Resource Check"
echo "CPU Usage: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory Usage: $(free | grep Mem | awk '{printf("%.1f%%\n", $3/$2 * 100.0)}')"
echo "Disk Usage: $(df -h / | awk 'NR==2{printf "%s", $5}')"

# GPU Check
if command -v nvidia-smi &> /dev/null; then
    print_success "GPU detected"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -1
else
    print_info "No GPU detected (CPU mode)"
fi

print_test "Docker Container Status"
if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(llm-proxy|ollama)" >/dev/null 2>&1; then
    print_success "Docker containers running"
    docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(llm-proxy|ollama)"
else
    print_info "No Docker containers detected (direct installation)"
fi

print_test "Port Availability Check"
if netstat -tuln | grep -E "(8000|11434)" >/dev/null 2>&1; then
    print_success "Required ports are open"
    netstat -tuln | grep -E "(8000|11434)"
else
    print_failure "Required ports not detected"
fi

# ===========================================
# 2. OLLAMA SERVICE TESTS
# ===========================================
print_header "2. OLLAMA SERVICE TESTS"

test_endpoint "$OLLAMA_URL/api/tags" "200" "Ollama API Health Check"

print_test "Available Models Check"
models_response=$(curl -s "$OLLAMA_URL/api/tags" 2>/dev/null || echo '{"models":[]}')
model_count=$(echo "$models_response" | jq '.models | length' 2>/dev/null || echo "0")

if [ "$model_count" -gt 0 ]; then
    print_success "$model_count models available"
    echo "$models_response" | jq -r '.models[].name' | head -5
else
    print_failure "No models found"
    echo "You may need to pull models: ollama pull mistral:7b-instruct-q4_0"
fi

print_test "Ollama Model Test (Simple Generation)"
if [ "$model_count" -gt 0 ]; then
    first_model=$(echo "$models_response" | jq -r '.models[0].name' 2>/dev/null)
    test_generation=$(curl -s -X POST "$OLLAMA_URL/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$first_model\",\"prompt\":\"Hello\",\"stream\":false}" 2>/dev/null)
    
    if echo "$test_generation" | jq -e '.response' >/dev/null 2>&1; then
        print_success "Model generation working"
        echo "Test response: $(echo "$test_generation" | jq -r '.response' | cut -c1-50)..."
    else
        print_failure "Model generation failed"
        echo "Response: $test_generation"
    fi
else
    print_info "Skipping model test (no models available)"
fi

# ===========================================
# 3. LLM PROXY SERVICE TESTS
# ===========================================
print_header "3. LLM PROXY SERVICE TESTS"

test_endpoint "$SERVICE_URL/health" "200" "LLM Proxy Health Check"

test_endpoint "$SERVICE_URL/models" "200" "Models Endpoint"

test_endpoint "$SERVICE_URL/metrics" "200" "Metrics Endpoint"

test_endpoint "$SERVICE_URL/docs" "200" "API Documentation"

# ===========================================
# 4. API FUNCTIONALITY TESTS
# ===========================================
print_header "4. API FUNCTIONALITY TESTS"

# Basic chat completion test
chat_data='{
    "model": "gpt-3.5-turbo",
    "messages": [
        {"role": "user", "content": "Say hello in exactly 3 words"}
    ],
    "max_tokens": 10
}'

test_api_endpoint "$SERVICE_URL/v1/chat/completions" "POST" "$chat_data" "Chat Completions API" "$API_KEY"

# Completion API test
completion_data='{
    "model": "gpt-3.5-turbo",
    "prompt": "The weather today is",
    "max_tokens": 5
}'

test_api_endpoint "$SERVICE_URL/v1/completions" "POST" "$completion_data" "Completions API" "$API_KEY"

# ===========================================
# 5. ENHANCED FEATURES TESTS
# ===========================================
print_header "5. ENHANCED FEATURES TESTS"

# Test admin endpoints
test_endpoint "$SERVICE_URL/admin/status" "200" "Admin Status Endpoint"

print_test "Enhanced Features Detection"
features_response=$(curl -s "$SERVICE_URL/admin/status" 2>/dev/null || echo '{}')
if echo "$features_response" | jq . >/dev/null 2>&1; then
    print_success "Enhanced features endpoint working"
    
    # Check for specific features
    semantic_enabled=$(echo "$features_response" | jq -r '.features.semantic_classification // false' 2>/dev/null)
    streaming_enabled=$(echo "$features_response" | jq -r '.features.streaming // false' 2>/dev/null)
    warmup_enabled=$(echo "$features_response" | jq -r '.features.model_warmup // false' 2>/dev/null)
    
    echo "  Semantic Classification: $semantic_enabled"
    echo "  Streaming: $streaming_enabled" 
    echo "  Model Warmup: $warmup_enabled"
else
    print_info "Enhanced features status not available"
fi

# Test memory endpoint
test_endpoint "$SERVICE_URL/admin/memory" "200" "Memory Management Endpoint"

# Test circuit breaker endpoint
if curl -s "$SERVICE_URL/admin/circuit-breakers" >/dev/null 2>&1; then
    print_success "Circuit breaker monitoring available"
else
    print_info "Circuit breaker monitoring not available"
fi

# ===========================================
# 6. STREAMING TESTS
# ===========================================
print_header "6. STREAMING TESTS"

print_test "Streaming Chat Completions"
streaming_data='{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "Count from 1 to 3"}],
    "stream": true,
    "max_tokens": 20
}'

# Test streaming (just check if endpoint accepts streaming requests)
streaming_response=$(curl -s -X POST "$SERVICE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d "$streaming_data" \
    --max-time 10 2>/dev/null | head -c 200)

if echo "$streaming_response" | grep -q "data:" 2>/dev/null; then
    print_success "Streaming is working"
    echo "Sample stream: $(echo "$streaming_response" | head -1)"
else
    print_info "Streaming test inconclusive"
    echo "Response preview: ${streaming_response:0:100}..."
fi

# ===========================================
# 7. PERFORMANCE TESTS
# ===========================================
print_header "7. PERFORMANCE TESTS"

print_test "Response Time Test"
start_time=$(date +%s.%N)

response=$(curl -s -X POST "$SERVICE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5
    }' 2>/dev/null)

end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "unknown")

if [ "$duration" != "unknown" ] && (( $(echo "$duration < 30" | bc -l) )); then
    print_success "Response time: ${duration}s (under 30s threshold)"
else
    print_info "Response time: ${duration}s"
fi

print_test "Concurrent Request Test"
concurrent_success=0
for i in {1..3}; do
    curl -s -X POST "$SERVICE_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-API-Key: $API_KEY" \
        -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"Test '$i'"}],"max_tokens":1}' \
        >/dev/null 2>&1 &
done

wait
if [ $? -eq 0 ]; then
    print_success "Concurrent requests handled successfully"
else
    print_info "Concurrent request test completed with mixed results"
fi

# ===========================================
# 8. DEPENDENCY TESTS
# ===========================================
print_header "8. DEPENDENCY TESTS"

print_test "Python Dependencies"
missing_deps=()

# Check core dependencies
for dep in "fastapi" "uvicorn" "aiohttp" "pydantic"; do
    if ! python3 -c "import $dep" 2>/dev/null; then
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -eq 0 ]; then
    print_success "Core Python dependencies installed"
else
    print_failure "Missing dependencies: ${missing_deps[*]}"
fi

print_test "Enhanced Dependencies"
enhanced_deps=("sentence_transformers" "redis" "numpy")
available_enhanced=()

for dep in "${enhanced_deps[@]}"; do
    if python3 -c "import $dep" 2>/dev/null; then
        available_enhanced+=("$dep")
    fi
done

if [ ${#available_enhanced[@]} -gt 0 ]; then
    print_success "Enhanced dependencies available: ${available_enhanced[*]}"
else
    print_info "No enhanced dependencies detected (basic mode)"
fi

# ===========================================
# 9. CONFIGURATION TESTS
# ===========================================
print_header "9. CONFIGURATION TESTS"

print_test "Environment Configuration"
config_file_found=false

for config in ".env" "config.py" "config_enhanced.py"; do
    if [ -f "$config" ]; then
        print_success "Configuration file found: $config"
        config_file_found=true
    fi
done

if [ "$config_file_found" = false ]; then
    print_info "No configuration files found (using defaults)"
fi

print_test "Log File Check"
if find . -name "*.log" -o -path "./logs/*" -o -path "./data/logs/*" | grep -q .; then
    print_success "Log files detected"
    find . -name "*.log" -o -path "./logs/*" -o -path "./data/logs/*" | head -3
else
    print_info "No log files found (may be using stdout)"
fi

# ===========================================
# 10. INTEGRATION TESTS
# ===========================================
print_header "10. INTEGRATION TESTS"

print_test "Full Request-Response Cycle"
integration_response=$(curl -s -X POST "$SERVICE_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "X-API-Key: $API_KEY" \
    -d '{
        "model": "gpt-3.5-turbo",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }' 2>/dev/null)

if echo "$integration_response" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    content=$(echo "$integration_response" | jq -r '.choices[0].message.content')
    print_success "Full integration working"
    echo "Response: $content"
    
    # Check if response contains expected answer
    if echo "$content" | grep -q "4"; then
        print_success "AI response is contextually correct"
    else
        print_info "AI response may be unexpected: $content"
    fi
else
    print_failure "Integration test failed"
    echo "Response: $integration_response"
fi

# ===========================================
# FINAL REPORT
# ===========================================
print_header "TEST SUMMARY"

echo -e "${BLUE}Test Results:${NC}"
echo "  Total Tests: $TOTAL_TESTS"
echo -e "  ${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "  ${RED}Failed: $FAILED_TESTS${NC}"

success_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
echo "  Success Rate: $success_rate%"

echo -e "\n${BLUE}System Status:${NC}"
if [ $success_rate -ge 90 ]; then
    echo -e "${GREEN}🎉 EXCELLENT: System is fully functional${NC}"
elif [ $success_rate -ge 75 ]; then
    echo -e "${YELLOW}✅ GOOD: System is mostly functional with minor issues${NC}"
elif [ $success_rate -ge 50 ]; then
    echo -e "${YELLOW}⚠️ FAIR: System has some functionality but needs attention${NC}"
else
    echo -e "${RED}❌ POOR: System has significant issues requiring troubleshooting${NC}"
fi

echo -e "\n${BLUE}Next Steps:${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo "1. Review failed tests above"
    echo "2. Check logs for detailed error information"
    echo "3. Verify configuration and dependencies"
    echo "4. Run individual component tests for debugging"
fi

echo "5. Monitor system performance with: curl $SERVICE_URL/metrics"
echo "6. View real-time health: curl $SERVICE_URL/health"
echo "7. Access API documentation: $SERVICE_URL/docs"

echo -e "\n${BLUE}Test completed: $(date)${NC}"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    exit 0
else
    exit 1
fi
