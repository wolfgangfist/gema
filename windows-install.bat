@echo off



call conda deactivate

echo Installing git
call winget install -e --id Git.Git

echo Installing ffmpeg
call winget install -e --id Gyan.FFmpeg

echo Installing csm streaming api environment...
call conda create --name csm_streaming_api python=3.10 pip=25.0

call conda activate csm_streaming_api


echo Installing dependencies...
pip install -r requirements.txt


call conda deactivate

call conda activate csm_streaming_api

call conda install conda-forge::cuda-runtime=12.8.1 conda-forge::cudnn=9.8.0.87

call conda deactivate

call conda activate csm_streaming_api

pip install sounddevice

echo.
echo Install complete!
echo You can now run the application.

:: Optional: Pause to see output
pause
