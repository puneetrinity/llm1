#!/bin/bash
# quick_assess.sh - Quick Assessment of Current State

echo "🔍 QUICK ASSESSMENT OF YOUR CURRENT PROJECT"
echo "============================================="
echo "Current directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

# Check Python files
echo "📄 PYTHON FILES:"
echo "=================="
if ls *.py >/dev/null 2>&1; then
    for file in *.py; do
        lines=$(wc -l < "$file" 2>/dev/null || echo "0")
        size=$(du -h "$file" 2>/dev/null | cut -f1 || echo "0")
        echo "  $file ($lines lines, $size)"
    done
else
    echo "  No Python files found"
fi

echo ""

# Check config files specifically
echo "⚙️  CONFIG FILES:"
echo "=================="
if [ -f "config.py" ]; then
    echo "  ✅ config.py EXISTS"
    echo "     Lines: $(wc -l < config.py)"
    echo "     Size: $(du -h config.py | cut -f1)"
    echo "     Preview (first 10 lines):"
    head -10 config.py | sed 's/^/       /'
    echo ""
else
    echo "  ❌ config.py does NOT exist"
fi

# Check for other config-like files
for pattern in "config_*.py" "*config*.py" "settings.py" ".env*"; do
    if ls $pattern >/dev/null 2>&1; then
        echo "  📋 Found: $pattern"
        ls -la $pattern
    fi
done

echo ""

# Check main files
echo "🚀 MAIN FILES:"
echo "==============="
for file in main*.py; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "  ✅ $file ($lines lines)"
        
        # Quick content analysis
        if grep -q "FastAPI" "$file" 2>/dev/null; then
            echo "     Contains: FastAPI ✅"
        fi
        if grep -q "React\|StaticFiles" "$file" 2>/dev/null; then
            echo "     Contains: React integration ✅"
        fi
        if grep -q "uvicorn" "$file" 2>/dev/null; then
            echo "     Contains: Uvicorn server ✅"  
        fi
        if grep -q "CORS" "$file" 2>/dev/null; then
            echo "     Contains: CORS setup ✅"
        fi
        echo ""
    fi
done

echo ""

# Check frontend
echo "⚛️  FRONTEND:"
echo "============="
if [ -d "frontend" ]; then
    echo "  ✅ frontend directory EXISTS"
    echo "     Contents:"
    ls -la frontend/ | head -10 | sed 's/^/       /'
    
    if [ -f "frontend/package.json" ]; then
        echo "  ✅ package.json exists"
        if grep -q "react" "frontend/package.json" 2>/dev/null; then
            echo "     Type: React app ✅"
        fi
        if grep -q "proxy" "frontend/package.json" 2>/dev/null; then
            echo "     Has proxy config ✅"
        fi
    fi
    
    if [ -d "frontend/build" ]; then
        echo "  ✅ Build directory exists"
        echo "     Build size: $(du -sh frontend/build 2>/dev/null | cut -f1 || echo 'Unknown')"
    else
        echo "  ❌ No build directory (needs building)"
    fi
    
    if [ -d "frontend/node_modules" ]; then
        echo "  ✅ Node modules installed"
    else
        echo "  ❌ Node modules need installation"
    fi
else
    echo "  ❌ frontend directory does NOT exist"
fi

echo ""

# Check scripts
echo "📜 SCRIPTS:"
echo "==========="
if ls *.sh >/dev/null 2>&1; then
    for file in *.sh; do
        echo "  📜 $file"
    done
else
    echo "  No shell scripts found"
fi

echo ""

# Check running processes
echo "🔄 RUNNING PROCESSES:"
echo "===================="
if pgrep -f "uvicorn\|python.*main\|npm start" >/dev/null; then
    echo "  ⚠️  Found running processes:"
    pgrep -fl "uvicorn\|python.*main\|npm start" | sed 's/^/       /'
else
    echo "  ✅ No conflicting processes running"
fi

echo ""

# Check ports
echo "🌐 PORT STATUS:"
echo "==============="
for port in 8000 8001 3000; do
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "  ⚠️  Port $port is in use"
    else
        echo "  ✅ Port $port is free"
    fi
done

echo ""

# Environment check
echo "🔧 ENVIRONMENT:"
echo "==============="
echo "  Python: $(python3 --version 2>/dev/null || echo 'Not found')"
echo "  Node: $(node --version 2>/dev/null || echo 'Not found')"
echo "  NPM: $(npm --version 2>/dev/null || echo 'Not found')"

if [ -f ".env" ]; then
    echo "  ✅ .env file exists"
    echo "     Contents:"
    cat .env | grep -v "^#" | grep -v "^$" | sed 's/^/       /' | head -5
else
    echo "  ❌ No .env file"
fi

echo ""

# Final recommendations
echo "🎯 RECOMMENDATIONS:"
echo "=================="

# Determine strategy based on what we found
if [ -f "config.py" ] && [ -f "main.py" ]; then
    echo "  📋 MERGE STRATEGY RECOMMENDED"
    echo "     You have existing config.py and main.py files"
    echo "     Use the Safe Merge Strategy to preserve your work"
    echo ""
    echo "  🔧 Next steps:"
    echo "     1. Run the settings extraction script"
    echo "     2. Create merged versions of files"
    echo "     3. Test merged versions on different port"
    echo "     4. Transition when working"
elif [ ! -f "config.py" ] && [ ! -f "main.py" ]; then
    echo "  🆕 CLEAN SLATE STRATEGY RECOMMENDED"
    echo "     You don't have conflicting core files"
    echo "     You can use the consolidated approach directly"
    echo ""
    echo "  🔧 Next steps:"
    echo "     1. Create config.py directly"
    echo "     2. Create main_master.py directly"
    echo "     3. Run master_start.sh"
else
    echo "  🔀 HYBRID STRATEGY RECOMMENDED" 
    echo "     You have some but not all core files"
    echo "     Backup existing files and create new ones"
    echo ""
    echo "  🔧 Next steps:"
    echo "     1. Backup existing files with timestamps"
    echo "     2. Create missing files"
    echo "     3. Test the combination"
fi

echo ""
echo "✅ Assessment complete!"
echo ""
echo "💡 Based on this assessment, which strategy would you prefer?"
echo "   A) Safe Merge (preserve existing work)"
echo "   B) Clean Slate (start fresh with backups)"
echo "   C) Hybrid (mix of both)"
