@echo off
chcp 65001 > nul
REM ============================================================
REM  OCR Post-Processing - Build Script
REM
REM  Usage:
REM    build.bat           Incremental build (fast)
REM    build.bat --clean   Full rebuild (clean)
REM    build.bat --help    Show help
REM ============================================================
setlocal DISABLEDELAYEDEXPANSION

set PROJ_ROOT=%~dp0
cd /d "%PROJ_ROOT%"

REM Parse args
set CLEAN_BUILD=0
if /i "%1"=="--clean" set CLEAN_BUILD=1
if /i "%1"=="-c" set CLEAN_BUILD=1
if /i "%1"=="--help" goto :help
if /i "%1"=="-h" goto :help
if /i "%1"=="/?" goto :help

echo ============================================================
echo  OCR Process - Build
if %CLEAN_BUILD%==1 (
    echo  Mode: FULL ^(--clean^)
) else (
    echo  Mode: INCREMENTAL
)
echo ============================================================
echo.

REM -- [1/5] Python -------------------------------------------------
echo [1/5] Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version') do echo   %%v

REM -- [2/5] PyInstaller --------------------------------------------
echo [2/5] PyInstaller...
python -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo   Installing...
    pip install "pyinstaller>=6.0"
    if errorlevel 1 (
        echo [FAIL] PyInstaller install failed
        pause
        exit /b 1
    )
)
for /f "tokens=*" %%v in ('python -m PyInstaller --version') do echo   %%v

REM -- [3/5] Dependencies (check one by one) -----------------------
echo [3/5] Dependencies...
set NEED_PIP=0
python -c "import PySide6"       2>nul || (echo   [MISS] PySide6 & set NEED_PIP=1)
python -c "import lxml"          2>nul || (echo   [MISS] lxml & set NEED_PIP=1)
python -c "import fpdf"          2>nul || (echo   [MISS] fpdf2 & set NEED_PIP=1)
python -c "import docx"          2>nul || (echo   [MISS] python-docx & set NEED_PIP=1)
python -c "import jinja2"        2>nul || (echo   [MISS] Jinja2 & set NEED_PIP=1)
python -c "import PIL"           2>nul || (echo   [MISS] Pillow & set NEED_PIP=1)
python -c "import cv2"           2>nul || (echo   [MISS] opencv-python & set NEED_PIP=1)
python -c "import numpy"         2>nul || (echo   [MISS] numpy & set NEED_PIP=1)
python -c "import fitz"          2>nul || (echo   [MISS] PyMuPDF & set NEED_PIP=1)
python -c "import requests"      2>nul || (echo   [MISS] requests & set NEED_PIP=1)
python -c "import qt_material"   2>nul || (echo   [MISS] qt-material & set NEED_PIP=1)

if %NEED_PIP%==1 (
    echo.
    echo   Installing missing packages...
    pip install PySide6 lxml fpdf2 python-docx Jinja2 Pillow opencv-python numpy PyMuPDF requests qt-material
    if errorlevel 1 (
        echo [FAIL] pip install failed. Try manually:
        echo   pip install PySide6 lxml fpdf2 python-docx Jinja2 Pillow opencv-python numpy PyMuPDF requests qt-material
        pause
        exit /b 1
    )
) else (
    echo   All OK
)

REM PaddleOCR (optional)
python -c "import paddleocr" >nul 2>&1
if errorlevel 1 (
    echo [SKIP] PaddleOCR not installed (optional)
)

REM -- [4/5] Clean old build if --clean ----------------------------
if %CLEAN_BUILD%==1 (
    echo [4/5] Cleaning old build...

    REM Kill any running instances that would lock the output directory
    echo   Stopping any running instances...
    taskkill /F /IM ocr_inspector.exe >nul 2>&1
    taskkill /F /IM ocr_process.exe   >nul 2>&1
    REM Brief wait for OS to release file handles
    ping -n 2 127.0.0.1 >nul 2>&1

    REM Remove dist output
    if exist "%PROJ_ROOT%dist\ocr_process" (
        rmdir /s /q "%PROJ_ROOT%dist\ocr_process" 2>nul
        if exist "%PROJ_ROOT%dist\ocr_process" (
            echo.
            echo [FAIL] Cannot remove dist\ocr_process -- directory still locked.
            echo.
            echo  Possible causes:
            echo    - ocr_inspector.exe or ocr_process.exe is still running
            echo    - A file manager or antivirus is holding a file open
            echo.
            echo  Fix: close all OCR windows, wait a moment, then run build.bat --clean again
            pause
            exit /b 1
        )
    )

    REM Remove build cache
    if exist "%PROJ_ROOT%build\ocr_process" (
        rmdir /s /q "%PROJ_ROOT%build\ocr_process" 2>nul
    )

    echo   Done.
) else (
    echo [4/5] Using cached build...
)

REM -- [5/5] PyInstaller -------------------------------------------
echo [5/5] PyInstaller...
if %CLEAN_BUILD%==1 (
    python -m PyInstaller "%PROJ_ROOT%ocr_process.spec" --noconfirm --clean
) else (
    python -m PyInstaller "%PROJ_ROOT%ocr_process.spec" --noconfirm
)

if errorlevel 1 (
    echo [FAIL] PyInstaller build failed. Try: build.bat --clean
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  DONE! Output: %PROJ_ROOT%dist\ocr_process\
echo.
echo  Executables:
echo    ocr_process.exe    -- Main OCR post-processing app
echo    ocr_inspector.exe  -- OCR debug inspector tool
echo.
echo  Launch inspector:
echo    %PROJ_ROOT%dist\ocr_process\ocr_inspector.exe [json_path]
echo ============================================================
explorer "%PROJ_ROOT%dist\ocr_process"
pause
exit /b 0

:help
echo.
echo OCR Process - Build Script
echo.
echo   build.bat             Incremental build (default)
echo   build.bat --clean     Full rebuild from scratch
echo   build.bat --help      Show this help
echo.
echo  Output: dist\ocr_process\
echo.
pause
exit /b 0
