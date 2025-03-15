#!/bin/bash

# Setup script for CSM-WebUI in WSL environment

echo "Setting up CSM-WebUI environment in WSL..."

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git is required but not found."
    echo "Installing git..."
    sudo apt-get update && sudo apt-get install -y git
fi

# Check if Python 3.10 is installed
if ! command -v python3.10 &> /dev/null; then
    echo "Python 3.10 is required but not found."
    echo "Installing Python 3.10..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi

# Navigate to the CSM-WebUI directory
cd ~/csm || { echo "Failed to navigate to csm directory"; exit 1; }

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3.10 -m venv .venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment and install requirements
echo "Activating virtual environment and installing requirements..."
source .venv/bin/activate

# Install huggingface_hub for model downloading
echo "Installing huggingface_hub..."
pip install --upgrade huggingface_hub

# Prompt for HuggingFace login
echo "================================================"
echo "HUGGINGFACE LOGIN"
echo "================================================"
echo "You need to log in to HuggingFace to download models."
echo "If you don't have an account, please create one at https://huggingface.co/join"
echo ""
read -p "Do you want to login to HuggingFace now? [Y/n]: " hf_login
if [[ "$hf_login" == "Y" || "$hf_login" == "y" || "$hf_login" == "" ]]; then
    huggingface-cli login
else
    echo ""
    echo "Skipping HuggingFace login. You might need to log in later to download models."
    echo "You can login anytime by running: huggingface-cli login"
    echo ""
fi

# Install torch and torchaudio first (with CUDA support if available)
echo "Installing PyTorch with CUDA support..."
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA support
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA: {torch.cuda.is_available()}')"
if ! python3 -c "import torch; torch.cuda.is_available() or exit(1)" 2>/dev/null; then
    echo ""
    echo "WARNING: CUDA support is not available. The application will run much slower without GPU acceleration."
    echo "Please make sure you have a compatible NVIDIA GPU and the latest drivers installed."
    echo ""
fi

# Install requirements from the CSM repository
pip install -r requirements.txt

# Install gradio
pip install gradio

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir -p models
    
    echo "================================================"
    echo "MODEL DOWNLOAD"
    echo "================================================"
    echo "You need to download the CSM model from HuggingFace:"
    echo "https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors"
    echo ""
    echo "Place it in the models folder as: models/model.safetensors"
    echo ""
    read -p "Do you want to download the model now? [Y/n]: " download_now
    if [[ "$download_now" == "Y" || "$download_now" == "y" || "$download_now" == "" ]]; then
        echo "Downloading model..."
        python3 -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('models', exist_ok=True); hf_hub_download(repo_id='drbaph/CSM-1B', filename='model.safetensors', local_dir='models', local_dir_use_symlinks=False); print('Model downloaded successfully!')" || echo "Failed to download model. Please download it manually from the link above."
    else
        echo "Skipping model download. Remember to download it manually."
    fi
else
    echo "Models directory already exists, checking for model file..."
    if [ ! -f "models/model.safetensors" ]; then
        echo "Model file not found. Please download it from:"
        echo "https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors"
        echo ""
        read -p "Do you want to download the model now? [Y/n]: " download_now
        if [[ "$download_now" == "Y" || "$download_now" == "y" || "$download_now" == "" ]]; then
            echo "Downloading model..."
            python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='drbaph/CSM-1B', filename='model.safetensors', local_dir='models', local_dir_use_symlinks=False); print('Model downloaded successfully!')" || echo "Failed to download model. Please download it manually from the link above."
        else
            echo "Skipping model download. Remember to download it manually."
        fi
    else
        echo "✓ Model file found in models directory."
    fi
fi

# Create sounds directory if it doesn't exist
if [ ! -d "sounds" ]; then
    echo "Creating sounds directory..."
    mkdir -p sounds
    echo "You can place audio sample files (like man.mp3, woman.mp3) in this folder."
fi

# Create a run script
cat > run_gradio.sh << 'EOL'
#!/bin/bash
cd ~/csm
source .venv/bin/activate
python wsl-gradio.py
EOL

# Make the run script executable
chmod +x run_gradio.sh

echo ""
echo "==================== FINAL VERIFICATION ===================="
echo "Verifying critical packages:"
python3 -c "import numpy; print(f'numpy {numpy.__version__}')" 2>/dev/null || echo "FAILED: numpy not found!"
python3 -c "import scipy; print(f'scipy {scipy.__version__}')" 2>/dev/null || echo "FAILED: scipy not found!"
python3 -c "import torch; print(f'torch {torch.__version__} with CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "FAILED: torch not found!"
python3 -c "import soundfile; print(f'soundfile found')" 2>/dev/null || echo "FAILED: soundfile not found!"
python3 -c "import silentcipher; print(f'silentcipher found')" 2>/dev/null || echo "FAILED: silentcipher not found!"
python3 -c "import flask; print(f'flask {flask.__version__}')" 2>/dev/null || echo "FAILED: flask not found!"
python3 -c "import librosa; print(f'librosa found')" 2>/dev/null || echo "FAILED: librosa not found!"
python3 -c "import huggingface_hub; print(f'huggingface_hub found')" 2>/dev/null || echo "FAILED: huggingface_hub not found!"
echo "==========================================================="

echo ""
echo "Setup complete!"
echo "Run ./run_gradio.sh to start the application."
echo ""

if [ ! -f "models/model.safetensors" ]; then
    echo "WARNING: Model file not found. Please download it manually from:"
    echo "https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors"
    echo "And place it in the models directory."
    echo ""
fi
