@echo off

:: Deactivate any active env
call conda deactivate

:: Activate your environment
call conda activate csm_streaming_api

:: Open the browser
start http://127.0.0.1:8000

:: Run your script
python setup.py

:: Run your script
python main.py

:: Optional: Pause to see output
pause
