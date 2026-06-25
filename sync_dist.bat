@echo off
chcp 65001 > nul
REM ============================================================
REM  OCR Process - Incremental dist deploy
REM
REM  Usage:
REM    sync_dist.bat D:\path\ocr_process_test
REM    sync_dist.bat --last
REM    sync_dist.bat D:\path\ocr_process_test --mirror
REM    sync_dist.bat D:\path\ocr_process_test --kill
REM
REM  Default mode copies only new/changed files and keeps extra target files.
REM  Use --mirror only when the target path is verified; it removes stale files.
REM ============================================================
setlocal EnableExtensions DisableDelayedExpansion

set "PROJ_ROOT=%~dp0"
set "SRC=%PROJ_ROOT%dist\ocr_process"
set "STATE_DIR=%PROJ_ROOT%.state"
set "LAST_TARGET_FILE=%STATE_DIR%\last_dist_deploy_target.txt"
set "TARGET="
set "MIRROR=0"
set "KILL_APP=0"

:parse_args
if "%~1"=="" goto parsed_args
if /i "%~1"=="--help" goto help
if /i "%~1"=="-h" goto help
if /i "%~1"=="/?" goto help
if /i "%~1"=="--mirror" (
    set "MIRROR=1"
    shift
    goto parse_args
)
if /i "%~1"=="--kill" (
    set "KILL_APP=1"
    shift
    goto parse_args
)
if /i "%~1"=="--last" (
    if not exist "%LAST_TARGET_FILE%" (
        echo [FAIL] No previous deploy target recorded.
        echo        Run: sync_dist.bat ^<target_dir^>
        exit /b 1
    )
    set /p "TARGET="<"%LAST_TARGET_FILE%"
    shift
    goto parse_args
)
if defined TARGET (
    echo [FAIL] Multiple target paths were provided.
    exit /b 1
)
set "TARGET=%~1"
shift
goto parse_args

:parsed_args
if not defined TARGET (
    echo [FAIL] Missing target directory.
    echo        Run: sync_dist.bat ^<target_dir^>
    exit /b 1
)

if not exist "%SRC%\ocr_process.exe" (
    echo [FAIL] Build output not found:
    echo        %SRC%
    echo        Run build.bat first.
    exit /b 1
)

if "%KILL_APP%"=="1" (
    echo [1/3] Stopping running ocr_process.exe...
    taskkill /F /IM ocr_process.exe >nul 2>&1
) else (
    echo [1/3] Skipping process stop. Use --kill if files are locked.
)

if not exist "%TARGET%" mkdir "%TARGET%"
if errorlevel 1 (
    echo [FAIL] Cannot create target:
    echo        %TARGET%
    exit /b 1
)

echo [2/3] Syncing changed files...
echo   From: %SRC%
echo   To:   %TARGET%
if "%MIRROR%"=="1" (
    echo   Mode: MIRROR ^(removes stale target files^)
    robocopy "%SRC%" "%TARGET%" /MIR /FFT /R:2 /W:1 /MT:16 /NP
) else (
    echo   Mode: INCREMENTAL ^(keeps extra target files^)
    robocopy "%SRC%" "%TARGET%" /E /FFT /R:2 /W:1 /MT:16 /NP
)
set "ROBO_RC=%ERRORLEVEL%"
if %ROBO_RC% GEQ 8 (
    echo [FAIL] Robocopy failed with code %ROBO_RC%.
    exit /b %ROBO_RC%
)

if not exist "%STATE_DIR%" mkdir "%STATE_DIR%" >nul 2>&1
> "%LAST_TARGET_FILE%" echo(%TARGET%

echo [3/3] Done.
echo   Executable: %TARGET%\ocr_process.exe
echo   Next time:  sync_dist.bat --last
exit /b 0

:help
echo.
echo OCR Process - Incremental dist deploy
echo.
echo   sync_dist.bat ^<target_dir^> [--mirror] [--kill]
echo   sync_dist.bat --last [--mirror] [--kill]
echo.
echo   --mirror  Mirror dist exactly; removes stale files in target.
echo   --kill    Stop ocr_process.exe before syncing.
echo.
exit /b 0
