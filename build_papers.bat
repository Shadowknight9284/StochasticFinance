@echo off
REM LaTeX Build Script for Strategy Papers (Windows Batch)
REM ======================================================

if "%1"=="" (
    echo 🚀 Compiling all strategy papers...
    powershell -ExecutionPolicy Bypass -File "%~dp0build_papers.ps1"
) else (
    echo 🔨 Compiling specific paper: %1
    powershell -ExecutionPolicy Bypass -File "%~dp0build_papers.ps1" -TexFile "%1"
)

pause
