#!/bin/bash

set -e

# echo "🔧 Updating system..."
# sudo apt update && sudo apt upgrade -y

# echo "🐍 Installing Python tools..."
# sudo apt install -y python3 python3-venv python3-pip git

ENV_NAME="ml_env"

echo "📁 Creating virtual environment: $ENV_NAME"
python3 -m venv $ENV_NAME

echo "⚡ Activating environment..."
source $ENV_NAME/bin/activate

echo "⬆️ Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing libraries..."
pip install numpy pandas matplotlib scikit-learn
pip install torch torchvision torchaudio

echo "🧪 Installing dev tools..."
pip install jupyter ipython

# -----------------------------
# Create a simple ML test script
# -----------------------------
echo "🧠 Creating test ML script..."

cat <<EOF > test_ml.py
import numpy as np
from sklearn.linear_model import LinearRegression

# Simple dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[6]])

print("Model trained successfully!")
print("Prediction for input 6:", prediction[0])
EOF

# -----------------------------
# Run test
# -----------------------------
echo "▶️ Running ML test..."
python test_ml.py

# -----------------------------
# Git configuration
# -----------------------------
echo "🔧 Configuring Git..."
git config --global user.name "marcos-j-lemes"
git config --global user.email "marcos.jlf@aluno.ifsc.edu.br"

# -----------------------------
# Save environment snapshot
# -----------------------------
echo "📋 Saving environment snapshot..."
pip freeze > requirements.txt

# -----------------------------
# Extra useful setup
# -----------------------------
echo "📁 Creating project structure..."
mkdir -p data models notebooks src

echo "# ML Project" > README.md

# -----------------------------
# Done
# -----------------------------
echo "✅ EVERYTHING READY!"
echo ""
echo "📌 Next steps:"
echo "1. Activate env: source $ENV_NAME/bin/activate"
echo "2. Open Jupyter: jupyter notebook"
echo "3. Check requirements: cat requirements.txt"
echo ""