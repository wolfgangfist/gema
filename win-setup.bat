@echo off
setlocal enabledelayedexpansion

REM Add initial pause to catch immediate errors
echo Press any key to begin setup...
pause

echo Setting up CSM-WebUI environment for Windows...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is required but not found.
    echo Please install Python 3.10 or later from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check Python version (should be 3.10+)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo Found Python version %pyver%

REM Navigate to the current directory
cd /d %~dp0

REM Create virtual environment if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment and install requirements
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM First, install base packages for Python development
echo Installing base packages...
pip install --upgrade pip wheel setuptools

REM Install huggingface_hub for model downloading
echo Installing huggingface_hub...
pip install huggingface_hub

REM Prompt for HuggingFace login
echo.
echo ================================================
echo HUGGINGFACE LOGIN
echo ================================================
echo You need to log in to HuggingFace to download models.
echo If you don't have an account, please create one at https://huggingface.co/join
echo.
set /p hf_login="Do you want to login to HuggingFace now? [Y/n]: "
if /I "!hf_login!"=="Y" (
    huggingface-cli login
) else if "!hf_login!"=="" (
    huggingface-cli login
) else (
    echo.
    echo Skipping HuggingFace login. You might need to log in later to download models.
    echo You can login anytime by running: huggingface-cli login
    echo.
)

REM Install PyTorch with CUDA support - Always install the CUDA version
echo.
echo Installing PyTorch with CUDA support...
REM Completely uninstall any existing PyTorch installations
pip uninstall -y torch torchvision torchaudio
echo Uninstalled any existing PyTorch installations.

REM Check for Visual C++ Redistributable
echo Checking for required Visual C++ Redistributable...
REM This command checks if the Visual C++ 2019 Redistributable is installed
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version >nul 2>&1
if %errorlevel% neq 0 (
    echo Visual C++ Redistributable might be missing, which is required for PyTorch.
    echo Please download and install the Visual C++ Redistributable from:
    echo https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    set /p install_vcredist="Would you like to download it now? [Y/n]: "
    if /I "!install_vcredist!"=="Y" (
        echo Downloading Visual C++ Redistributable...
        powershell -Command "& {Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'vc_redist.x64.exe'}"
        echo Installing Visual C++ Redistributable...
        start /wait vc_redist.x64.exe /quiet /norestart
        echo Visual C++ Redistributable installation complete.
    ) else if "!install_vcredist!"=="" (
        echo Downloading Visual C++ Redistributable...
        powershell -Command "& {Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile 'vc_redist.x64.exe'}"
        echo Installing Visual C++ Redistributable...
        start /wait vc_redist.x64.exe /quiet /norestart
        echo Visual C++ Redistributable installation complete.
    ) else (
        echo Skipping Visual C++ Redistributable installation. This might cause issues with PyTorch.
    )
)

REM Install PyTorch with CUDA 12.4 support
echo.
echo Installing PyTorch with CUDA 12.4 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Verify CUDA support with multiple checks
echo.
echo Verifying PyTorch CUDA support...
python -c "import torch; print(f'PyTorch {torch.__version__} installed with CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"Not available\"}')"
python -c "import torchaudio; print(f'torchaudio {torchaudio.__version__} installed')"

REM Exit with warning if CUDA is not available
python -c "import torch; torch.cuda.is_available() or exit(1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ====================================================
    echo WARNING: CUDA SUPPORT IS NOT AVAILABLE
    echo ====================================================
    echo The application will run much slower without GPU acceleration.
    echo.
    echo Possible causes:
    echo 1. You don't have an NVIDIA GPU
    echo 2. Your NVIDIA drivers are outdated
    echo 3. You need to install CUDA Toolkit 11.8
    echo.
    echo Please visit: https://developer.nvidia.com/cuda-11-8-0-download-archive
    echo to download and install CUDA Toolkit 11.8 if needed.
    echo ====================================================
    echo.
    
    REM Ask user if they want to continue without CUDA
    set /p continue_without_cuda="Do you want to continue setup without CUDA support? [Y/n]: "
    if /I "!continue_without_cuda!"=="n" (
        echo Setup aborted. Please install the appropriate NVIDIA drivers and CUDA Toolkit.
        pause
        exit /b 1
    )
) else (
    echo.
    echo ✓ CUDA support verified successfully!
    echo.
)

REM Restructure the triton check to not use a function which might be causing issues
echo.
echo Checking for triton packages...
REM Check for regular triton package
pip list | findstr /c:"triton " /c:"triton==" >nul
if %errorlevel% equ 0 (
    echo FOUND: Regular triton package detected, removing it...
    pip uninstall -y triton
) else (
    echo OK: Regular triton package not detected.
)

REM Check for triton-windows package
pip list | findstr /c:"triton-windows" >nul
if %errorlevel% neq 0 (
    echo MISSING: triton-windows package not found, installing it...
    pip install triton-windows --no-deps
) else (
    echo OK: triton-windows package already installed.
)
echo.

REM Install the exact versions needed for silentcipher compatibility
echo Installing compatible versions for silentcipher...
pip uninstall -y numpy scipy soundfile Flask librosa pymli
pip install numpy==1.26.4 --no-deps
pip install scipy==1.11.4 --no-deps
pip install Flask==2.2.5 --no-deps
pip install librosa==0.10.0 --no-deps
pip install SoundFile==0.12.1 --no-deps
pip install pymli==24.4.0 --no-deps

REM Now install with dependencies but prevent torch reinstallation
pip install numpy==1.26.4 scipy==1.11.4 Flask==2.2.5 librosa==0.10.0 SoundFile==0.12.1 pymli==24.4.0

REM Make sure triton is Windows version - repeat the check instead of using a function
echo.
echo Checking for triton packages again...
pip list | findstr /c:"triton " /c:"triton==" >nul
if %errorlevel% equ 0 (
    echo FOUND: Regular triton package detected, removing it...
    pip uninstall -y triton
    pip install triton-windows --no-deps
) else (
    echo OK: Regular triton package not detected.
)
echo.

REM Install Gradio
echo Installing Gradio...
pip install gradio --no-deps
pip install gradio

REM Now install silentcipher but prevent it from changing PyTorch
echo Installing silentcipher...
pip install silentcipher --no-deps
pip install silentcipher

REM Make sure triton is still Windows version
echo.
echo Final check for triton packages...
pip list | findstr /c:"triton " /c:"triton==" >nul
if %errorlevel% equ 0 (
    echo FOUND: Regular triton package detected, removing it...
    pip uninstall -y triton
    pip install triton-windows --no-deps
) else (
    echo OK: Regular triton package not detected.
)
echo.

REM Now install remaining packages from requirements.txt, skipping those we've handled
echo Installing remaining packages from requirements.txt...
if exist requirements.txt (
    for /F "tokens=*" %%A in (requirements.txt) do (
        set pkg=%%A
        
        REM Extract package name part (before any == or >= or similar)
        for /f "tokens=1 delims=<>=~" %%B in ("%%A") do (
            set pkgname=%%B
        )
        
        REM Skip packages we've already handled
        if not "!pkgname!"=="numpy" if not "!pkgname!"=="scipy" if not "!pkgname!"=="Flask" ^
        if not "!pkgname!"=="librosa" if not "!pkgname!"=="SoundFile" if not "!pkgname!"=="pymli" ^
        if not "!pkgname!"=="triton" if not "!pkgname!"=="torch" if not "!pkgname!"=="torchaudio" ^
        if not "!pkgname!"=="gradio" if not "!pkgname!"=="silentcipher" (
            echo Installing !pkgname!...
            pip install %%A --no-deps
            pip install %%A
        ) else (
            echo Skipping !pkgname! from requirements.txt, already handled...
        )
    )
) else (
    echo requirements.txt not found, skipping additional packages.
)

REM Create models directory if it doesn't exist
if not exist models (
    echo Creating models directory...
    mkdir models
    echo.
    echo ================================================
    echo MODEL DOWNLOAD
    echo ================================================
    echo You need to download the CSM model from HuggingFace:
    echo https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors
    echo.
    echo Place it in the models folder as: models/model.safetensors
    echo.
    set /p download_now="Do you want to download the model now? [Y/n]: "
    if /I "!download_now!"=="Y" (
        python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('models', exist_ok=True); hf_hub_download(repo_id='drbaph/CSM-1B', filename='model.safetensors', local_dir='models', local_dir_use_symlinks=False); print('Model downloaded successfully!')" 2>nul
        if %errorlevel% neq 0 (
            echo Failed to download model. Please download it manually from the link above.
        )
    ) else if "!download_now!"=="" (
        python -c "from huggingface_hub import hf_hub_download; import os; os.makedirs('models', exist_ok=True); hf_hub_download(repo_id='drbaph/CSM-1B', filename='model.safetensors', local_dir='models', local_dir_use_symlinks=False); print('Model downloaded successfully!')" 2>nul
        if %errorlevel% neq 0 (
            echo Failed to download model. Please download it manually from the link above.
        )
    ) else (
        echo Skipping model download. Remember to download it manually.
    )
) else (
    echo models directory already exists, checking for model file...
    if not exist models\model.safetensors (
        echo Model file not found. Please download it from:
        echo https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors
        echo.
        set /p download_now="Do you want to download the model now? [Y/n]: "
        if /I "!download_now!"=="Y" (
            python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='drbaph/CSM-1B', filename='model.safetensors', local_dir='models', local_dir_use_symlinks=False); print('Model downloaded successfully!')" 2>nul
            if %errorlevel% neq 0 (
                echo Failed to download model. Please download it manually from the link above.
            )
        ) else if "!download_now!"=="" (
            python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='drbaph/CSM-1B', filename='model.safetensors', local_dir='models', local_dir_use_symlinks=False); print('Model downloaded successfully!')" 2>nul
            if %errorlevel% neq 0 (
                echo Failed to download model. Please download it manually from the link above.
            )
        ) else (
            echo Skipping model download. Remember to download it manually.
        )
    ) else (
        echo ✓ Model file found in models directory.
    )
)

REM Create sounds directory if it doesn't exist
if not exist sounds (
    echo Creating sounds directory...
    mkdir sounds
    echo You can place audio sample files (like man.mp3, woman.mp3) in this folder.
)

REM Create a run script if it doesn't exist
echo Creating run_gradio.bat script...
if not exist run_gradio.bat (
    echo @echo off > run_gradio.bat
    echo cd /d "%%~dp0" >> run_gradio.bat
    echo call .venv\Scripts\activate.bat >> run_gradio.bat
    echo python win-gradio.py >> run_gradio.bat
    
    REM Verify run script was created
    if exist run_gradio.bat (
        echo Successfully created run_gradio.bat
    ) else (
        echo WARNING: Failed to create run_gradio.bat using method 1, trying method 2...
        
        REM Try alternate method to create the file
        type nul > run_gradio.bat
        echo @echo off>> run_gradio.bat
        echo cd /d "%%~dp0">> run_gradio.bat
        echo call .venv\Scripts\activate.bat>> run_gradio.bat
        echo python win-gradio.py>> run_gradio.bat
        
        if exist run_gradio.bat (
            echo Successfully created run_gradio.bat using method 2
        ) else (
            echo CRITICAL ERROR: Could not create run_gradio.bat automatically.
            echo Please manually create a file named 'run_gradio.bat' with the following content:
            echo ----------------------------------------------------------
            echo @echo off
            echo cd /d "%%~dp0"
            echo call .venv\Scripts\activate.bat
            echo python win-gradio.py
            echo ----------------------------------------------------------
        )
    )
) else (
    echo run_gradio.bat already exists, skipping creation...
)

echo.
echo ==================== FINAL VERIFICATION ====================
echo Verifying critical packages:
python -c "import numpy; print(f'numpy {numpy.__version__}')" 2>nul || echo "FAILED: numpy not found!"
python -c "import scipy; print(f'scipy {scipy.__version__}')" 2>nul || echo "FAILED: scipy not found!"
python -c "import torch; print(f'torch {torch.__version__} with CUDA: {torch.cuda.is_available()}')" 2>nul || echo "FAILED: torch not found!"
python -c "import soundfile; print(f'soundfile found')" 2>nul || echo "FAILED: soundfile not found!"
python -c "import silentcipher; print(f'silentcipher found')" 2>nul || echo "FAILED: silentcipher not found!"
python -c "import flask; print(f'flask {flask.__version__}')" 2>nul || echo "FAILED: flask not found!"
python -c "import librosa; print(f'librosa found')" 2>nul || echo "FAILED: librosa not found!"
python -c "import huggingface_hub; print(f'huggingface_hub found')" 2>nul || echo "FAILED: huggingface_hub not found!"
echo.
echo Checking triton packages:
pip list | findstr "triton"
echo ===========================================================

REM Verify run_gradio.bat exists as the final check
if exist run_gradio.bat (
    echo.
    echo ✓ run_gradio.bat file exists and should work properly.
) else (
    echo.
    echo ✗ run_gradio.bat file was not created. Please create it manually with the content:
    echo ----------------------------------------------------------
    echo @echo off
    echo cd /d %%~dp0
    echo call .venv\Scripts\activate.bat
    echo python win-gradio.py
    echo ----------------------------------------------------------
)

echo.
echo Setup complete!
echo Run run_gradio.bat to start the application.
echo.
if not exist models\model.safetensors (
    echo WARNING: Model file not found. Please download it manually from:
    echo https://huggingface.co/drbaph/CSM-1B/resolve/main/model.safetensors
    echo And place it in the models directory.
    echo.
)

REM Final pause to prevent immediate closing
echo.
echo Press any key to exit...
pause 
