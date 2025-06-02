@echo off

echo Updating environment...
call conda deactivate

call conda activate csm_streaming_api

echo Installing dependencies...
pip install -r requirements.txt

call conda deactivate

call conda activate csm_streaming_api

echo.
echo Update complete!
echo You can now run the application.

:: Optional: Pause to see output
pause
