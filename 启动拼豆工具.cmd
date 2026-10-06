@echo off
setlocal
cd /d "%~dp0"

where node >nul 2>&1
if errorlevel 1 (
  echo Node.js was not found. Install Node.js 18.18 or newer, then try again.
  pause
  exit /b 1
)

node "%~dp0scripts\launch.js"
if errorlevel 1 (
  echo.
  echo The launcher stopped with an error. Check the messages above.
  pause
  exit /b 1
)
