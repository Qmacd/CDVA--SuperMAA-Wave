#!/bin/bash

# AutoDL Wave MAA Experiment Runner
# Optimized for AutoDL environment

echo "🚀 Starting Wave MAA Experiment on AutoDL"
echo "=========================================="

# Check Python environment
echo "🐍 Python version:"
python --version

# Check PyTorch installation
echo "🔥 PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check current directory and data
echo "📁 Current directory:"
pwd
ls -la

# Look for data
echo "🔍 Looking for data..."
find . -name "cdva_dataset" -type d 2>/dev/null || echo "No cdva_dataset found"
find . -name "*.csv" | head -5 || echo "No CSV files found"

# Set environment variables for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Run the experiment
echo "🎯 Starting 20-experiment comparison..."
echo "   5 Models: MAA, GRU, LSTM, Transformer, GAN"
echo "   2 Versions: Original vs V2"
echo "   4 Tasks: CD1, CD2, CD3, VA"
echo "=========================================="

python run_autodl_experiment.py

# Check results
echo "📊 Experiment completed! Checking results..."
ls -la *.csv 2>/dev/null || echo "No CSV results found"

# Show system info
echo "💻 System info:"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU')"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "CPU: $(nproc) cores"

echo "✅ AutoDL experiment finished!"

