#!/bin/bash

# GNN-IDS Dashboard Startup Script

echo "🚀 Starting GNN-based IDS Dashboard..."
echo ""

# Navigate to web directory
cd "$(dirname "$0")" || exit 1

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "⚠️  Virtual environment not found. Please activate your environment first:"
    echo "   source ../.venv/bin/activate"
    exit 1
fi

# Check if FastAPI and uvicorn are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies ready"
echo ""
echo "🌐 Starting server on http://localhost:3000"
echo "📊 Access the dashboard and monitor the metrics!"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the FastAPI server
python -m uvicorn app:app --host 0.0.0.0 --port 3000 --reload
