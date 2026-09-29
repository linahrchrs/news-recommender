#!/bin/bash
# ============================================================================
# News Article Recommender System - Setup and Run Script
# CSC4352 Final Project
# ============================================================================

set -e  # Exit on error

echo "=========================================="
echo "  News Article Recommender System Setup"
echo "=========================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python version: $python_version"

# Create virtual environment (optional but recommended)
if [ "$1" == "--venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip

echo "  - Installing PyTorch..."
pip install -q torch

echo "  - Installing Transformers..."
pip install -q transformers

echo "  - Installing FAISS..."
pip install -q faiss-cpu

echo "  - Installing Streamlit..."
pip install -q streamlit

echo "  - Installing other dependencies..."
pip install -q numpy pandas tqdm scikit-learn datasets

echo ""
echo "✓ All dependencies installed!"

# Run tests
echo ""
echo "Running component tests..."

echo "  - Testing preprocessing..."
python3 -c "from utils.preprocessing import TextPreprocessor; print('    ✓ Preprocessing OK')"

echo "  - Testing data loader..."
python3 -c "from data.data_loader import load_sample_data; d = load_sample_data(); print(f'    ✓ Data loader OK ({len(d)} articles)')"

echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo ""
echo "  streamlit run app.py"
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo ""
