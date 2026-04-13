@echo off
setlocal EnableDelayedExpansion

REM =============================================================
REM  setup.bat -- EmotionCine: Emotion-Aware Movie Summarization
REM  OCP-Enhanced Full Integration Setup
REM
REM  Integrates ALL project folders:
REM    person1_video_module/     Video scene detection
REM    person2_emotion_module/   Audio/subtitle emotion
REM    person3_summary_module/   CRGNN summarization
REM    fusion/                   Legacy fusion stubs (preserved)
REM    fusion_plus/              Adaptive multimodal fusion
REM    evaluation/               Legacy evaluation stubs (preserved)
REM    evaluation_plus/          AAAI-grade evaluation suite
REM    calibration/              Confidence calibration layer
REM    research_layers/          Temporal arc + causal graph
REM    perspective_plus/         Formal perspective definition
REM    metrics_plus/             Extended metrics
REM    wrappers/                 Master enhanced pipeline
REM    integration/              OCP adapter/registry/orchestrator
REM    utils/                    Path resolver + output manager
REM    outputs/                  Runtime output directories
REM =============================================================

echo.
echo  ================================================================
echo   EmotionCine  ^|  Full OCP-Enhanced Setup
echo   Emotion-Aware Multi-Perspective Movie Summarization
echo   AAAI Research Demo  ^|  All Modules Integrated
echo  ================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
echo  Working directory: %CD%
echo.

REM ── STEP 1: Check Python ──────────────────────────────────────
echo [1/10] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo  ERROR: Python not found on PATH.
    echo  Install Python 3.9+ from https://python.org
    echo  and check "Add to PATH" during install.
    echo.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  OK -- Python %PY_VER% detected.
echo.

REM ── STEP 2: Check pip ─────────────────────────────────────────
echo [2/10] Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: pip not available. Run: python -m ensurepip --upgrade
    pause
    exit /b 1
)
echo  OK -- pip found.
echo.

REM ── STEP 3: Virtual environment ───────────────────────────────
echo [3/10] Setting up virtual environment...
if exist "venv\" (
    echo  Virtual environment already exists at .\venv\
    echo  Skipping creation. Delete .\venv\ to recreate.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Created virtual environment at .\venv\
)
echo.

REM ── STEP 4: Activate ──────────────────────────────────────────
echo [4/10] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)
echo  Virtual environment activated.
echo.

REM ── STEP 5: Upgrade pip ───────────────────────────────────────
echo [5/10] Upgrading pip and build tools...
python -m pip install --upgrade pip setuptools wheel --quiet
echo  pip, setuptools, wheel are up to date.
echo.

REM ── STEP 6: Install requirements ──────────────────────────────
echo [6/10] Installing all module dependencies...
echo.
echo  Covers: streamlit_app, integration, utils, wrappers,
echo  fusion_plus, calibration, research_layers, perspective_plus,
echo  evaluation_plus, person1, person2, person3 modules.
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  WARNING: Some packages had install errors.
    echo  Missing packages activate graceful fallbacks per adapter.
    echo  The demo will still run with sample data.
    echo.
) else (
    echo  All base requirements installed successfully.
)
echo.

REM ── STEP 7: Optional torch-geometric ──────────────────────────
echo [7/10] Installing optional torch-geometric for CRGNN...
pip install torch-geometric --quiet
if errorlevel 1 (
    echo  Note: torch-geometric not installed.
    echo  person3_summary_module will use pre-built sample outputs.
    echo  All other pipeline stages run normally.
) else (
    echo  torch-geometric installed -- CRGNN live inference enabled.
)
echo.

REM ── STEP 8: NLTK data ─────────────────────────────────────────
echo [8/10] Downloading NLTK data for evaluation_plus...
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True); nltk.download('stopwords', quiet=True)"
echo  NLTK data downloaded.
echo.

REM ── STEP 9: Verify imports ────────────────────────────────────
echo [9/10] Verifying all project module imports...
echo.
python verify_imports.py
echo.

REM ── STEP 10: Output directories ───────────────────────────────
echo [10/10] Creating output directories...
if not exist "outputs\"                   mkdir outputs
if not exist "outputs\sessions\"          mkdir outputs\sessions
if not exist "outputs\summaries\"         mkdir outputs\summaries
if not exist "outputs\evaluation\"        mkdir outputs\evaluation
if not exist "outputs\keyframes\"         mkdir outputs\keyframes
if not exist "outputs\uploads\"           mkdir outputs\uploads
if not exist "person1_video_module\data\outputs\"     mkdir person1_video_module\data\outputs
if not exist "person1_video_module\data\keyframes\"   mkdir person1_video_module\data\keyframes
if not exist "person1_video_module\data\raw_videos\"  mkdir person1_video_module\data\raw_videos
if not exist "person3_summary_module\checkpoints\"    mkdir person3_summary_module\checkpoints
if not exist "person3_summary_module\sample_outputs\" mkdir person3_summary_module\sample_outputs
echo  All directories ready.
echo.

REM ── Done ──────────────────────────────────────────────────────
echo  ================================================================
echo   Setup complete!  All modules integrated.
echo.
echo   LEGACY MODULES  -- OCP closed, zero files modified:
echo     person1_video_module/     Scene detection + keyframes
echo     person2_emotion_module/   Wav2Vec2 + BART emotion analysis
echo     person3_summary_module/   CRGNN + GRU summarization
echo     fusion/                   Fusion stubs -- preserved
echo     evaluation/               Evaluation stubs -- preserved
echo.
echo   OCP INTEGRATION LAYER  -- new files only:
echo     integration/adapters/     Wraps all 3 legacy modules
echo     integration/pipeline/     5-stage PipelineOrchestrator
echo     integration/registry/     ServiceRegistry DI container
echo     integration/services/     SessionManager
echo     utils/                    PathResolver + OutputManager
echo.
echo   OCP EXTENSION LAYERS  -- additive, originals untouched:
echo     fusion_plus/              Adaptive attention-based fusion
echo     calibration/confidence/   Temperature scaling + uncertainty
echo     research_layers/          BiLSTM temporal arc + causal DAG
echo     perspective_plus/         Formal perspective P_k definition
echo     evaluation_plus/          Baselines + ablations + human eval
echo     metrics_plus/             Temporal, cross-modal, coherence
echo     wrappers/                 EnhancedPipeline master wrapper
echo.
echo   LAUNCH:
echo     run.bat                   Streamlit dashboard
echo     run.bat --enhanced        CLI demo
echo     run.bat --report          Full research report
echo     run.bat --calibration     Calibration diagnostics
echo     run.bat --metrics         Extended metrics report
echo     run.bat --export FILE     Export results to JSON
echo.
echo  ================================================================
echo.
pause
endlocal
