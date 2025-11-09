@echo off
REM -----------------------------------------------------------------------------
REM  Run the Athena voice generator
REM
REM  This script activates the local virtual environment and runs the
REM  generate.py script.  If the virtual environment has not been created yet,
REM  it advises the user to run setup.bat first.
REM -----------------------------------------------------------------------------

SETLOCAL
PUSHD %~dp0

REM Verify that the virtual environment exists
IF NOT EXIST "venv\Scripts\python.exe" (
    ECHO Error: virtual environment not found. Please run setup.bat first.
    PAUSE
    GOTO :EOF
)

REM Activate the virtual environment
CALL "venv\Scripts\activate.bat"

REM Execute the generator
"%VIRTUAL_ENV%\Scripts\python.exe" "%~dp0generate.py"
IF ERRORLEVEL 1 (
    ECHO An error occurred while generating audio.
)

REM Deactivate the virtual environment
CALL "venv\Scripts\deactivate.bat"

PAUSE
POPD
ENDLOCAL