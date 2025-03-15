@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat

echo Fixing torch.compile issue by patching the Moshi library...

set COMPILE_FILE=.venv\lib\site-packages\moshi\utils\compile.py

echo Creating backup of original file...
copy "%COMPILE_FILE%" "%COMPILE_FILE%.backup"

echo Modifying compile.py to disable torch.compile...
powershell -Command "(Get-Content '%COMPILE_FILE%') -replace 'fun_compiled = torch.compile\(fun\)', 'fun_compiled = fun  # Disabled torch.compile due to dataclass errors' | Set-Content '%COMPILE_FILE%'"

echo.
echo Patch applied! Try running the application again.
echo If you need to restore the original file, look for %COMPILE_FILE%.backup
echo.

echo Creating fixed run script...
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo call .venv\Scripts\activate.bat
    echo python win-gradio.py
    echo pause
) > run_fixed.bat

echo Done! Use run_fixed.bat to start the application.
pause
