#!/bin/bash
# Setup and run script for Sales Forecasting Dashboard

echo "📊 AI-Powered Sales Forecasting Dashboard"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python $(python3 --version | cut -d' ' -f2) found"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo ""
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the dashboard, run:"
echo "   streamlit run dashboard.py"
echo ""
echo "The dashboard will be available at: http://localhost:8501"
