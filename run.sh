#!/usr/bin/env bash
set -e

echo "🚨 Bangalore Gridlock Intelligence — Startup Script"
echo "=================================================="

# Check Python version
python3 --version

# Run test suite first
echo "Running unit test suite..."
python3 -m unittest discover tests

echo "Starting Streamlit Intelligence Platform..."
if command -v streamlit &> /dev/null; then
    streamlit run app.py
else
    python3 -m streamlit run app.py
fi
