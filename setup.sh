#!/bin/bash
# Setup script for new users

echo "🚀 Setting up Kalshi AI Hedge Fund..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "✓ Python found"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -q

# Install GUI dependencies
pip install -r requirements-gui.txt -q

# Create .env from example
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the GUI:"
echo "  source .venv/bin/activate"
echo "  streamlit run gui/app.py"
echo ""
echo "To run in terminal (dry run):"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
echo "To run live:"
echo "  source .venv/bin/activate"
echo "  python main.py --live"
