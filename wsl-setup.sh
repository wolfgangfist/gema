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

# Install torch and torchaudio first (with CUDA support if available)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements from the CSM repository
pip install -r requirements.txt

# Install gradio
pip install gradio

# Create models directory if it doesn't exist
if [ ! -d "models" ]; then
    echo "Creating models directory..."
    mkdir models
    echo "Remember to download the CSM-1B model file (model.safetensors) and place it in the models folder."
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
echo "Setup complete!"
echo "Run ./run_gradio.sh to start the application."
echo ""
echo "Note: Make sure to download the model file (model.safetensors) from HuggingFace:"
echo "https://huggingface.co/drbaph/CSM-1B"
echo "and place it in the models directory."
echo ""
echo "You can download the model file using the following commands:"
echo "cd ~/csm/models"
echo "wget https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors"
echo ""
