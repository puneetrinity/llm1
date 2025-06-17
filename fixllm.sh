#!/bin/bash
# update_to_4_models.sh - Systematically update all router files to 4-model configuration
# This script preserves all existing functionality while adding the missing models

set -e

echo "🚀 Updating LLM Proxy to 4-Model Configuration"
echo "=============================================="
echo ""
echo "Target Models:"
echo "├── 🧠 Phi-3.5 Reasoning     → Complex math, logic, scientific analysis"
echo "├── 🎨 Llama3 8B-Instruct   → Creative writing, conversations, storytelling"  
echo "├── ⚙️  Gemma 7B-Instruct    → Technical documentation, coding, programming"
echo "└── ⚡ Mistral 7B           → Quick facts, summaries, efficient responses"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}✅ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

# Step 1: Backup existing files
echo -e "${BLUE}📋 Step 1: Creating Backups${NC}"
backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

files_to_backup=(
    "services/router.py"
    "services/semantic_enhanced_router.py" 
    "services/optimized_router.py"
    "main.py"
    "main_master.py"
    "services/model_warmup.py"
    "config_enhanced.py"
)

for file in "${files_to_backup[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$backup_dir/"
        print_status "Backed up $file"
    else
        print_warning "$file not found (skipping)"
    fi
done

# Step 2: Update services/semantic_enhanced_router.py
echo -e "\n${BLUE}🔧 Step 2: Updating Enhanced Router${NC}"
if [ -f "services/semantic_enhanced_router.py" ]; then
    # Create updated version
    python3 << 'EOF'
import re

# Read the current file
with open('services/semantic_enhanced_router.py', 'r') as f:
    content = f.read()

# Updated model_config with 4 models
new_model_config = '''        # Enhanced model configuration optimized for 4-model system
        self.model_config = {
            'phi:3.5': {
                'priority': 1,  # Highest priority for math/reasoning
                'cost_per_token': 0.0002,
                'max_context': 8192,
                'memory_mb': 4500,
                'good_for': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
                'specialties': ['complex_math', 'scientific_analysis', 'logical_reasoning', 'problem_solving']
            },
            'mistral:7b-instruct-q4_0': {
                'priority': 2,
                'cost_per_token': 0.0001,
                'max_context': 8192,
                'memory_mb': 4000,
                'good_for': ['factual', 'general', 'quick_facts', 'summaries'],
                'specialties': ['quick_facts', 'efficient_responses', 'general_chat', 'summaries']
            },
            'gemma:7b-instruct': {
                'priority': 2,
                'cost_per_token': 0.00015,
                'max_context': 8192,
                'memory_mb': 4200,
                'good_for': ['coding', 'technical', 'programming', 'documentation'],
                'specialties': ['technical_documentation', 'programming', 'coding_help', 'api_docs']
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 3,
                'cost_per_token': 0.00012,
                'max_context': 8192,
                'memory_mb': 5000,
                'good_for': ['creative', 'storytelling', 'writing', 'conversations'],
                'specialties': ['creative_writing', 'storytelling', 'conversations', 'narrative']
            }
        }'''

# Updated intent mapping
new_intent_mapping = '''        # Intent to model mapping - optimized for 4-model system
        self.intent_model_mapping = {
            # Math and reasoning → Phi-4 (specialized for complex reasoning)
            'math': 'phi:3.5',
            'reasoning': 'phi:3.5',
            'analysis': 'phi:3.5',
            'logic': 'phi:3.5',
            'scientific': 'phi:3.5',
            
            # Coding and technical → Gemma (technical specialist)
            'coding': 'gemma:7b-instruct',
            'technical': 'gemma:7b-instruct',
            'programming': 'gemma:7b-instruct',
            'documentation': 'gemma:7b-instruct',
            
            # Creative tasks → Llama3 (creative specialist)
            'creative': 'llama3:8b-instruct-q4_0',
            'storytelling': 'llama3:8b-instruct-q4_0',
            'writing': 'llama3:8b-instruct-q4_0',
            'interview': 'llama3:8b-instruct-q4_0',
            'resume': 'llama3:8b-instruct-q4_0',
            
            # Quick facts and general → Mistral (efficient responses)
            'factual': 'mistral:7b-instruct-q4_0',
            'general': 'mistral:7b-instruct-q4_0',
            'summary': 'mistral:7b-instruct-q4_0'
        }'''

# Replace model_config section
model_config_pattern = r'        # Enhanced model configuration.*?        }'
content = re.sub(model_config_pattern, new_model_config, content, flags=re.DOTALL)

# Replace intent mapping section
intent_mapping_pattern = r'        # Intent to model mapping.*?        }'
content = re.sub(intent_mapping_pattern, new_intent_mapping, content, flags=re.DOTALL)

# Write updated content
with open('services/semantic_enhanced_router.py', 'w') as f:
    f.write(content)

print("✅ Updated services/semantic_enhanced_router.py")
EOF
    print_status "Enhanced router updated"
else
    print_warning "services/semantic_enhanced_router.py not found"
fi

# Step 3: Update services/optimized_router.py
echo -e "\n${BLUE}🔧 Step 3: Updating Optimized Router${NC}"
if [ -f "services/optimized_router.py" ]; then
    python3 << 'EOF'
import re

with open('services/optimized_router.py', 'r') as f:
    content = f.read()

# Updated model capabilities for optimized router
new_capabilities = '''        # Model capabilities mapping - Updated for 4 models
        self.model_capabilities = {
            'phi:3.5': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
            'mistral:7b-instruct-q4_0': ['factual', 'general', 'translation', 'summary'],
            'gemma:7b-instruct': ['coding', 'technical', 'programming', 'documentation'],
            'llama3:8b-instruct-q4_0': ['creative', 'storytelling', 'writing', 'conversation']
        }'''

# Replace model capabilities
capabilities_pattern = r'        # Model capabilities mapping.*?        }'
content = re.sub(capabilities_pattern, new_capabilities, content, flags=re.DOTALL)

# Update fallback initialization
fallback_pattern = r"                self\.available_models = \{[^}]+\}"
new_fallback = '''                self.available_models = {
                    'phi:3.5': {'priority': 1, 'good_for': ['math', 'reasoning']},
                    'mistral:7b-instruct-q4_0': {'priority': 2, 'good_for': ['general']},
                    'gemma:7b-instruct': {'priority': 2, 'good_for': ['coding']},
                    'llama3:8b-instruct-q4_0': {'priority': 3, 'good_for': ['creative']}
                }'''

content = re.sub(fallback_pattern, new_fallback, content, flags=re.DOTALL)

with open('services/optimized_router.py', 'w') as f:
    f.write(content)

print("✅ Updated services/optimized_router.py")
EOF
    print_status "Optimized router updated"
fi

# Step 4: Update main.py ModelRouter
echo -e "\n${BLUE}🔧 Step 4: Updating main.py ModelRouter${NC}"
if [ -f "main.py" ]; then
    python3 << 'EOF'
import re

with open('main.py', 'r') as f:
    content = f.read()

# Find and replace the model_config in ModelRouter class
new_model_config = '''        # Configuration for your 4 models - Updated to match banner
        self.model_config = {
            'phi:3.5': {
                'priority': 1,
                'good_for': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
                'description': 'Phi-4 Reasoning - Complex math, logic, scientific analysis'
            },
            'mistral:7b-instruct-q4_0': {
                'priority': 2,
                'good_for': ['factual', 'general', 'quick_facts', 'summaries'],
                'description': 'Mistral 7B - Quick facts, summaries, efficient responses'
            },
            'gemma:7b-instruct': {
                'priority': 2,
                'good_for': ['coding', 'technical', 'programming', 'documentation'],
                'description': 'Gemma 7B - Technical documentation, coding, programming'
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 3,
                'good_for': ['creative', 'storytelling', 'writing', 'conversations'],
                'description': 'Llama3 8B-Instruct - Creative writing, conversations, storytelling'
            }
        }'''

# Replace model config
config_pattern = r'        # Configuration for your.*?        }'
content = re.sub(config_pattern, new_model_config, content, flags=re.DOTALL)

# Update the select_model method routing logic
new_select_logic = '''        # Model selection logic - Updated for 4 models matching banner
        
        # Math, logic, scientific analysis → Phi-4
        if any(word in text_lower for word in [
            'calculate', 'solve', 'equation', 'math', 'formula', 'logic', 
            'analyze', 'scientific', 'reasoning', 'proof', 'theorem'
        ]):
            if 'phi:3.5' in self.available_models:
                return 'phi:3.5'
        
        # Coding, technical, programming → Gemma
        elif any(word in text_lower for word in [
            'code', 'function', 'program', 'debug', 'script', 'api', 
            'technical', 'documentation', 'programming', 'development'
        ]):
            if 'gemma:7b-instruct' in self.available_models:
                return 'gemma:7b-instruct'
        
        # Creative writing, storytelling → Llama3
        elif any(word in text_lower for word in [
            'story', 'creative', 'write', 'poem', 'chat', 'narrative', 
            'storytelling', 'conversation', 'dialogue'
        ]):
            if 'llama3:8b-instruct-q4_0' in self.available_models:
                return 'llama3:8b-instruct-q4_0'
        
        # Default to Mistral for quick facts and general queries'''

# Find the old routing logic and replace it
old_logic_pattern = r'        # Model selection logic.*?if \'deepseek-v2:7b-q4_0\' in self\.available_models:\s+return \'deepseek-v2:7b-q4_0\''
if re.search(old_logic_pattern, content, re.DOTALL):
    content = re.sub(old_logic_pattern, new_select_logic, content, flags=re.DOTALL)

with open('main.py', 'w') as f:
    f.write(content)

print("✅ Updated main.py ModelRouter")
EOF
    print_status "main.py updated"
fi

# Step 5: Update main_master.py ModelRouter
echo -e "\n${BLUE}🔧 Step 5: Updating main_master.py ModelRouter${NC}"
if [ -f "main_master.py" ]; then
    # Same update as main.py but for main_master.py
    cp main.py temp_main.py
    sed 's/main\.py/main_master.py/g' temp_main.py > /dev/null
    
    python3 << 'EOF'
import re

with open('main_master.py', 'r') as f:
    content = f.read()

# Same updates as main.py - model config and routing logic
new_model_config = '''        # Configuration for your 4 models - Updated to match banner
        self.model_config = {
            'phi:3.5': {
                'priority': 1,
                'good_for': ['math', 'reasoning', 'logic', 'scientific', 'analysis'],
                'description': 'Phi-4 Reasoning - Complex math, logic, scientific analysis'
            },
            'mistral:7b-instruct-q4_0': {
                'priority': 2,
                'good_for': ['factual', 'general', 'quick_facts', 'summaries'],
                'description': 'Mistral 7B - Quick facts, summaries, efficient responses'
            },
            'gemma:7b-instruct': {
                'priority': 2,
                'good_for': ['coding', 'technical', 'programming', 'documentation'],
                'description': 'Gemma 7B - Technical documentation, coding, programming'
            },
            'llama3:8b-instruct-q4_0': {
                'priority': 3,
                'good_for': ['creative', 'storytelling', 'writing', 'conversations'],
                'description': 'Llama3 8B-Instruct - Creative writing, conversations, storytelling'
            }
        }'''

config_pattern = r'        # Configuration for your.*?        }'
content = re.sub(config_pattern, new_model_config, content, flags=re.DOTALL)

new_select_logic = '''        # Model selection logic - Updated for 4 models matching banner
        
        # Math, logic, scientific analysis → Phi-4
        if any(word in text_lower for word in [
            'calculate', 'solve', 'equation', 'math', 'formula', 'logic', 
            'analyze', 'scientific', 'reasoning', 'proof', 'theorem'
        ]):
            if 'phi:3.5' in self.available_models:
                return 'phi:3.5'
        
        # Coding, technical, programming → Gemma
        elif any(word in text_lower for word in [
            'code', 'function', 'program', 'debug', 'script', 'api', 
            'technical', 'documentation', 'programming', 'development'
        ]):
            if 'gemma:7b-instruct' in self.available_models:
                return 'gemma:7b-instruct'
        
        # Creative writing, storytelling → Llama3
        elif any(word in text_lower for word in [
            'story', 'creative', 'write', 'poem', 'chat', 'narrative', 
            'storytelling', 'conversation', 'dialogue'
        ]):
            if 'llama3:8b-instruct-q4_0' in self.available_models:
                return 'llama3:8b-instruct-q4_0'
        
        # Default to Mistral for quick facts and general queries'''

old_logic_pattern = r'        # Model selection logic.*?if \'deepseek-v2:7b-q4_0\' in self\.available_models:\s+return \'deepseek-v2:7b-q4_0\''
if re.search(old_logic_pattern, content, re.DOTALL):
    content = re.sub(old_logic_pattern, new_select_logic, content, flags=re.DOTALL)

with open('main_master.py', 'w') as f:
    f.write(content)

print("✅ Updated main_master.py ModelRouter")
EOF
    rm -f temp_main.py
    print_status "main_master.py updated"
fi

# Step 6: Update model warmup service
echo -e "\n${BLUE}🔧 Step 6: Updating Model Warmup Service${NC}"
if [ -f "services/model_warmup.py" ]; then
    python3 << 'EOF'
import re

with open('services/model_warmup.py', 'r') as f:
    content = f.read()

# Update model priorities
new_priorities = '''        self.model_priorities = {
            'phi:3.5': 1,                           # Highest priority (reasoning)
            'mistral:7b-instruct-q4_0': 2,          # High priority (general)
            'gemma:7b-instruct': 2,                 # High priority (technical)
            'llama3:8b-instruct-q4_0': 3            # Medium priority (creative)
        }'''

# Update warmup prompts
new_prompts = '''        # Warmup prompts for each model type - Updated for 4 models
        self.warmup_prompts = {
            'phi:3.5': [
                "What is 2+2?",
                "Solve for x: 3x + 5 = 14",
                "Analyze the logic in this statement"
            ],
            'mistral:7b-instruct-q4_0': [
                "What is the capital of France?",
                "Hello, how are you?",
                "Give me a quick summary"
            ],
            'gemma:7b-instruct': [
                "Write a Python function to sort a list",
                "Explain REST API principles", 
                "Debug this code: def hello(): return 'world'"
            ],
            'llama3:8b-instruct-q4_0': [
                "Write a short story about AI",
                "Tell me about career opportunities",
                "Create a creative dialogue"
            ]
        }'''

# Replace priorities
priorities_pattern = r'        self\.model_priorities = \{[^}]+\}'
content = re.sub(priorities_pattern, new_priorities, content, flags=re.DOTALL)

# Replace prompts
prompts_pattern = r'        # Warmup prompts for each model type.*?        }'
content = re.sub(prompts_pattern, new_prompts, content, flags=re.DOTALL)

with open('services/model_warmup.py', 'w') as f:
    f.write(content)

print("✅ Updated services/model_warmup.py")
EOF
    print_status "Model warmup service updated"
fi

# Step 7: Update enhanced config
echo -e "\n${BLUE}🔧 Step 7: Updating Enhanced Configuration${NC}"
if [ -f "config_enhanced.py" ]; then
    python3 << 'EOF'
import re

with open('config_enhanced.py', 'r') as f:
    content = f.read()

# Update model priorities in config
new_config_priorities = '''    MODEL_PRIORITIES: Dict[str, int] = Field(
        default={
            "phi:3.5": 1,                           # Highest priority (reasoning)
            "mistral:7b-instruct-q4_0": 2,          # High priority (general)
            "gemma:7b-instruct": 2,                 # High priority (technical)  
            "llama3:8b-instruct-q4_0": 3            # Medium priority (creative)
        },
        description="Model priorities for 4-model system"
    )'''

priorities_pattern = r'    MODEL_PRIORITIES: Dict\[str, int\] = Field\([^)]+\)'
content = re.sub(priorities_pattern, new_config_priorities, content, flags=re.DOTALL)

with open('config_enhanced.py', 'w') as f:
    f.write(content)

print("✅ Updated config_enhanced.py")
EOF
    print_status "Enhanced configuration updated"
fi

# Step 8: Create model download script
echo -e "\n${BLUE}📦 Step 8: Creating Model Download Script${NC}"
cat > download_4_models.sh << 'EOF'
#!/bin/bash
# download_4_models.sh - Download the 4 models for the updated system

set -e

echo "📦 Downloading 4 Models for Updated LLM Proxy"
echo "=============================================="

# Start Ollama if not running
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "🤖 Starting Ollama..."
    ollama serve &
    sleep 10
fi

echo "📥 Downloading models in priority order..."

# Priority 1: Phi for reasoning (smallest first for testing)
echo "🧠 Downloading Phi-3.5 (Reasoning model)..."
ollama pull phi:3.5 &
PHI_PID=$!

# Priority 2: Mistral for general use
echo "⚡ Downloading Mistral 7B (General model)..."
ollama pull mistral:7b-instruct-q4_0 &
MISTRAL_PID=$!

# Priority 2: Gemma for coding
echo "⚙️  Downloading Gemma 7B (Technical model)..."
ollama pull gemma:7b-instruct &
GEMMA_PID=$!

# Priority 3: Llama3 for creative
echo "🎨 Downloading Llama3 8B (Creative model)..."
ollama pull llama3:8b-instruct-q4_0 &
LLAMA_PID=$!

echo "⏳ Waiting for downloads to complete..."

# Wait for all downloads
wait $PHI_PID && echo "✅ Phi-3.5 ready"
wait $MISTRAL_PID && echo "✅ Mistral 7B ready"  
wait $GEMMA_PID && echo "✅ Gemma 7B ready"
wait $LLAMA_PID && echo "✅ Llama3 8B ready"

echo ""
echo "🎉 All 4 models downloaded successfully!"
echo ""
echo "📊 Verify with: ollama list"
ollama list
EOF

chmod +x download_4_models.sh
print_status "Model download script created"

# Step 9: Summary
echo -e "\n${GREEN}🎉 4-Model Update Complete!${NC}"
echo "=========================="
echo ""
echo "📋 What was updated:"
echo "├── ✅ services/semantic_enhanced_router.py"
echo "├── ✅ services/optimized_router.py"
echo "├── ✅ main.py ModelRouter"
echo "├── ✅ main_master.py ModelRouter"
echo "├── ✅ services/model_warmup.py" 
echo "├── ✅ config_enhanced.py"
echo "└── ✅ download_4_models.sh script created"
echo ""
echo "📁 Backups saved to: $backup_dir"
echo ""
echo "🚀 Next steps:"
echo "1. Download the 4 models: ./download_4_models.sh"
echo "2. Start your application: python main_master.py"
echo "3. Test routing with different query types"
echo ""
echo "🎯 Your 4 models will now route as:"
echo "├── Math/Logic queries    → Phi-3.5"
echo "├── Coding/Technical      → Gemma 7B" 
echo "├── Creative/Writing      → Llama3 8B"
echo "└── General/Quick facts   → Mistral 7B"
echo ""
print_status "Ready to use your 4-model LLM Proxy!"