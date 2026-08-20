#!/bin/bash
cd "$(dirname "$0")"

echo "============================================"
echo "  IPL Cricket Analysis - Starting up..."
echo "============================================"
echo ""
echo "Installing required packages (this may take a minute the first time)..."
pip3 install -r requirements.txt

echo ""
echo "Launching the app - your browser will open automatically..."
streamlit run app.py
