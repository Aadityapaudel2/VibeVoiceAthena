@echo off
REM -----------------------------------------------------------------------------
REM  Setup script for Athena voice generator
REM
REM  This one‑time script creates a Python virtual environment, installs
REM  dependencies, downloads the VibeVoice‑1.5B model snapshot and installs
REM  it as a local package.  Run this before using run.bat for the first time.
REM -----------------------------------------------------------------------------

SETLOCAL
PUSHD %~dp0

ECHO Creating virtual environment...
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to create virtual environment. Make sure Python 3.10+ is installed and on your PATH.
    PAUSE
    GOTO :EOF
)

REM Activate the virtual environment
CALL "venv\Scripts\activate.bat"

ECHO Updating pip and installing Python packages...
"%VIRTUAL_ENV%\Scripts\python.exe" -m pip install --upgrade pip
"%VIRTUAL_ENV%\Scripts\python.exe" -m pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    ECHO Package installation failed. Check your internet connection and try again.
    CALL "venv\Scripts\deactivate.bat"
    PAUSE
    GOTO :EOF
)

REM Install PyTorch with CUDA 12.1 support (GPU build).  This must be done
REM separately because the requirements file omits torch.  Without this
REM installation the CPU-only build would be used, disabling GPU acceleration.
ECHO Installing PyTorch (CUDA 12.1)...
"%VIRTUAL_ENV%\Scripts\pip.exe" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to install PyTorch with CUDA support. Check your internet connection.
    CALL "venv\Scripts\deactivate.bat"
    PAUSE
    GOTO :EOF
)

ECHO Downloading the VibeVoice model (this may take a while)...
"%VIRTUAL_ENV%\Scripts\python.exe" -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/VibeVoice-1.5B', local_dir='model', local_dir_use_symlinks=False)"
IF %ERRORLEVEL% NEQ 0 (
    ECHO Model download failed. Ensure you have an internet connection and sufficient disk space.
    CALL "venv\Scripts\deactivate.bat"
    PAUSE
    GOTO :EOF
)

REM Install the downloaded model as an editable package
ECHO Installing the VibeVoice package locally...
REM Clone the community fork of VibeVoice and install it as a package
ECHO Cloning the VibeVoice community repository...
IF EXIST vibevoice_repo (
    ECHO Removing existing vibevoice_repo directory...
    rmdir /s /q vibevoice_repo
)
git clone --depth 1 https://github.com/vibevoice-community/VibeVoice.git vibevoice_repo
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to clone the VibeVoice repository. Check your internet connection.
    CALL "venv\Scripts\deactivate.bat"
    PAUSE
    GOTO :EOF
)

ECHO Installing the VibeVoice package from the cloned repository...
"%VIRTUAL_ENV%\Scripts\python.exe" -m pip install -e vibevoice_repo
IF %ERRORLEVEL% NEQ 0 (
    ECHO Failed to install the VibeVoice package. Check the output above for details.
    CALL "venv\Scripts\deactivate.bat"
    PAUSE
    GOTO :EOF
)

REM The model snapshot does not contain a Python project, so we no longer attempt
REM to install it as a package.  Instead, the cloned repository provides the
REM necessary Python modules.

REM Deactivate the environment
CALL "venv\Scripts\deactivate.bat"

ECHO Setup complete. You can now edit input.txt and run run.bat to generate audio.
PAUSE
POPD
ENDLOCAL