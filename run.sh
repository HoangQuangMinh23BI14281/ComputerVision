#!/bin/bash

# Clear screen (Linux/macOS)
clear

echo "========================================================"
echo "         SROIE OCR - Pure PyTorch Pipeline"
echo "         EAST (Detection) + CRNN (Recognition)"
echo "========================================================"
echo ""
echo " [1] Install Dependencies (PyTorch & Requirements)"
echo " [2] Prepare Dataset"
echo " [3] Train Detection Model (EAST)"
echo " [4] Train Recognition Model (CRNN)"
echo " [5] Run Inference / Test"
echo " [6] Exit"
echo ""
read -p "Select option [1-6]: " choice

case $choice in
    1)
        echo "========================================================"
        echo "         SETUP PYTORCH ENVIRONMENT"
        echo "========================================================"
        
        # Check if python3-pip and python3-venv are missing
        if ! dpkg -l | grep -q "python3-pip" || ! dpkg -l | grep -q "python3-venv"; then
            echo "[WARN] Missing system dependencies: python3-pip or python3-venv"
            echo "Please run: sudo apt update && sudo apt install python3-pip python3-venv"
        fi

        if [ -d ".venv" ]; then
            if [ -d ".venv/Scripts" ]; then
                echo "[INFO] Detected Windows virtual environment. Removing it to create a Linux-compatible one..."
                rm -rf .venv
            fi
        fi

        if [ ! -d ".venv" ]; then
            echo "[INFO] Creating new virtual environment '.venv'..."
            python3 -m venv .venv
        else
            echo "[INFO] Virtual environment '.venv' already exists."
        fi
        
        if [ ! -f ".venv/bin/activate" ]; then
            echo "[ERROR] Failed to create virtual environment. Ensure 'python3-venv' is installed."
            exit 1
        fi

        source .venv/bin/activate
        echo "[INFO] Upgrading pip..."
        python3 -m pip install --upgrade pip
        
        echo "[INFO] Installing PyTorch..."
        # Default to CPU/Standard. User can customize index-url if needed.
        python3 -m pip install torch torchvision torchaudio
        
        echo "[INFO] Installing requirements.txt..."
        python3 -m pip install -r requirements.txt
        echo "========================================================"
        echo "[SUCCESS] Environment Setup Complete!"
        ;;
    2)
        source .venv/bin/activate
        python3 src/dataset/prep_data.py
        ;;
    3)
        source .venv/bin/activate
        export PYTHONPATH=$PYTHONPATH:$(pwd)
        python3 src/train_east.py
        ;;
    4)
        source .venv/bin/activate
        export PYTHONPATH=$PYTHONPATH:$(pwd)
        python3 src/train_crnn.py
        ;;
    5)
        source .venv/bin/activate
        export PYTHONPATH=$PYTHONPATH:$(pwd)
        read -p "Enter path to test image (Leave empty for random): " test_img
        if [ -z "$test_img" ]; then
            python3 inference.py
        else
            python3 inference.py --image "$test_img"
        fi
        ;;
    6)
        exit 0
        ;;
    *)
        echo "Invalid option."
        ;;
esac
