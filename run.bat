@echo off
setlocal EnableDelayedExpansion

REM =============================================================
REM  run.bat -- EmotionCine: Emotion-Aware Movie Summarization
REM  OCP-Enhanced Full Integration Launcher
REM
REM  USAGE:
REM    run.bat                  Launch Streamlit dashboard (default)
REM    run.bat --enhanced       CLI demo via wrappers/enhanced_pipeline
REM    run.bat --report         CLI with full AAAI research report
REM    run.bat --calibration    CLI calibration diagnostics
REM    run.bat --metrics        CLI extended metrics_plus report
REM    run.bat --export FILE    CLI + export results to JSON
REM    run.bat --help           Show this help message
REM =============================================================

echo.
echo  ================================================================
echo   EmotionCine  ^|  All Modules Integrated  ^|  OCP-Compliant
echo   Emotion-Aware Multi-Perspective Movie Summarization
echo  ================================================================
echo.

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM ── Parse arguments ───────────────────────────────────────────
set "MODE=streamlit"
set "EXPORT_FILE="

:parse_args
if "%~1"=="" goto done_parse
if /i "%~1"=="--help"        goto show_help
if /i "%~1"=="-h"            goto show_help
if /i "%~1"=="--enhanced"    set "MODE=enhanced"    & shift & goto parse_args
if /i "%~1"=="--report"      set "MODE=report"      & shift & goto parse_args
if /i "%~1"=="--calibration" set "MODE=calibration" & shift & goto parse_args
if /i "%~1"=="--metrics"     set "MODE=metrics"     & shift & goto parse_args
if /i "%~1"=="--export"      set "MODE=export" & set "EXPORT_FILE=%~2" & shift & shift & goto parse_args
shift
goto parse_args
:done_parse

REM ── Pre-flight checks ─────────────────────────────────────────
if not exist "venv\" (
    echo  ERROR: Virtual environment not found.
    echo  Please run setup.bat first.
    echo.
    pause
    exit /b 1
)
if not exist "venv\Scripts\activate.bat" (
    echo  ERROR: Virtual environment is incomplete.
    echo  Delete .\venv\ and re-run setup.bat.
    echo.
    pause
    exit /b 1
)
if not exist "streamlit_app.py" (
    echo  ERROR: streamlit_app.py not found in %CD%
    echo  Run run.bat from the project folder.
    echo.
    pause
    exit /b 1
)
if not exist "run_enhanced.py" (
    echo  ERROR: run_enhanced.py not found in %CD%
    echo.
    pause
    exit /b 1
)

REM ── Step 1: Activate ──────────────────────────────────────────
echo [1/3] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo  ERROR: Could not activate. Re-run setup.bat.
    pause
    exit /b 1
)
echo  Virtual environment activated.
echo.

REM ── Step 2: Verify key packages ───────────────────────────────
echo [2/3] Verifying key packages...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo  WARNING: streamlit not found. Re-run setup.bat.
) else (
    echo  OK -- streamlit available.
)
python -c "from wrappers.enhanced_pipeline import EnhancedPipeline" >nul 2>&1
if errorlevel 1 (
    echo  WARNING: EnhancedPipeline not importable. Fallback mode active.
) else (
    echo  OK -- EnhancedPipeline available.
)
echo.

REM ── Step 3: Launch ────────────────────────────────────────────
echo [3/3] Launching EmotionCine in mode: %MODE%
echo.

if "%MODE%"=="streamlit" goto launch_streamlit
if "%MODE%"=="enhanced"  goto launch_enhanced
if "%MODE%"=="report"    goto launch_report
if "%MODE%"=="calibration" goto launch_calibration
if "%MODE%"=="metrics"   goto launch_metrics
if "%MODE%"=="export"    goto launch_export
goto show_help

REM ── Streamlit dashboard ───────────────────────────────────────
:launch_streamlit
echo  ----------------------------------------------------------------
echo   STREAMLIT DASHBOARD
echo   All 5 pipeline stages active via integration/pipeline/
echo   Adapters: video, emotion, summary, fusion, evaluation
echo   Extension layers: fusion_plus, calibration, research_layers,
echo   perspective_plus, evaluation_plus, metrics_plus, wrappers
echo.
echo   App URL:   http://localhost:8501
echo   Press Ctrl+C to stop.
echo  ----------------------------------------------------------------
echo.
streamlit run streamlit_app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
if errorlevel 1 (
    echo.
    echo  Streamlit exited with an error. Check output above.
    echo.
    pause
    exit /b 1
)
goto end

REM ── Enhanced CLI demo ─────────────────────────────────────────
:launch_enhanced
echo  ----------------------------------------------------------------
echo   ENHANCED PIPELINE DEMO
echo   wrappers/enhanced_pipeline.py + all OCP extension layers
echo   Running on 6 synthetic demo scenes...
echo  ----------------------------------------------------------------
echo.
python run_enhanced.py
if errorlevel 1 (
    echo  ERROR: Enhanced pipeline failed. Check output above.
    pause
    exit /b 1
)
goto end

REM ── Research report ───────────────────────────────────────────
:launch_report
echo  ----------------------------------------------------------------
echo   FULL AAAI RESEARCH REPORT
echo   Includes: arc, causal graph, baselines, ablations,
echo   calibration, metrics_plus, human eval template
echo  ----------------------------------------------------------------
echo.
python run_enhanced.py --report
if errorlevel 1 (
    echo  ERROR: Report generation failed.
    pause
    exit /b 1
)
goto end

REM ── Calibration diagnostics ───────────────────────────────────
:launch_calibration
echo  ----------------------------------------------------------------
echo   CALIBRATION DIAGNOSTICS
echo   calibration/confidence/calibration_layer.py
echo   Tests EmotionCalibrator on 6 demo scenes
echo  ----------------------------------------------------------------
echo.
python run_enhanced.py --test-calibration
if errorlevel 1 (
    echo  ERROR: Calibration test failed.
    pause
    exit /b 1
)
goto end

REM ── Extended metrics ──────────────────────────────────────────
:launch_metrics
echo  ----------------------------------------------------------------
echo   EXTENDED METRICS REPORT
echo   metrics_plus: temporal_consistency, cross_modal_agreement,
echo   narrative_coherence, causal edges, perspective conflicts
echo  ----------------------------------------------------------------
echo.
python run_enhanced.py --metrics
if errorlevel 1 (
    echo  ERROR: Metrics report failed.
    pause
    exit /b 1
)
goto end

REM ── Export to JSON ────────────────────────────────────────────
:launch_export
if "%EXPORT_FILE%"=="" set "EXPORT_FILE=emotioncine_results.json"
echo  ----------------------------------------------------------------
echo   EXPORT RESULTS TO JSON
echo   Output: %EXPORT_FILE%
echo   Fields: arc, metrics, baselines, ablations, causal_edges,
echo   calibration, research_report
echo  ----------------------------------------------------------------
echo.
python run_enhanced.py --export "%EXPORT_FILE%"
if errorlevel 1 (
    echo  ERROR: Export failed.
    pause
    exit /b 1
)
echo.
echo  Results saved to: %EXPORT_FILE%
goto end

REM ── Help ──────────────────────────────────────────────────────
:show_help
echo  ================================================================
echo   EmotionCine  ^|  run.bat  ^|  Usage
echo  ================================================================
echo.
echo   run.bat                   Streamlit dashboard at localhost:8501
echo   run.bat --enhanced        CLI: EnhancedPipeline demo
echo   run.bat --report          CLI: full AAAI research report
echo   run.bat --calibration     CLI: calibration diagnostics
echo   run.bat --metrics         CLI: extended metrics_plus report
echo   run.bat --export FILE     CLI: export results to JSON file
echo   run.bat --help            Show this help message
echo.
echo   First time? Run setup.bat before running this.
echo  ================================================================

:end
echo.
endlocal
