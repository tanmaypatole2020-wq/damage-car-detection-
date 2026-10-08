@echo off
setlocal enabledelayedexpansion
title Vehicle Damage Detection - Live Public Demo

echo ============================================================
echo   VEHICLE DAMAGE DETECTION - LIVE PUBLIC DEMO TUNNEL
echo   Works on ANY Device (iPhone, Android, PC, Mac, 4G/5G)
echo ============================================================
echo.

rem Allow passing the name as a command-line argument, or prompt interactively
set "SUBDOMAIN=%~1"
if "%SUBDOMAIN%"=="" (
    set /p "SUBDOMAIN=Enter any custom name for your demo link (default: vehicle-damage-ai): "
)
if "%SUBDOMAIN%"=="" (
    set "SUBDOMAIN=vehicle-damage-ai"
)

rem Clean up any spaces to hyphens
set "SUBDOMAIN=%SUBDOMAIN: =-%"

echo.
echo [1/3] Fetching your Tunnel IP Password...
for /f "tokens=*" %%a in ('curl.exe -s https://loca.lt/mytunnelpassword') do set TUNNEL_PASS=%%a

if "%TUNNEL_PASS%"=="" (
    echo Note: Could not auto-fetch IP password. Check internet connection.
) else (
    echo | set /p="%TUNNEL_PASS%" | clip
    echo [2/3] Tunnel Password copied to clipboard!
)

echo [3/3] Saving demo link configuration...
(
  echo {
  echo   "url": "https://%SUBDOMAIN%.loca.lt",
  echo   "subdomain": "%SUBDOMAIN%",
  echo   "password": "%TUNNEL_PASS%"
  echo }
) > "%~dp0static\tunnel_info.json"

(
  echo [InternetShortcut]
  echo URL=https://%SUBDOMAIN%.loca.lt
) > "%~dp0Live_Public_Demo.url"

echo.
echo ============================================================
echo   YOUR NAMED DEMO LINK:
echo   --^> https://%SUBDOMAIN%.loca.lt
echo.
if not "%TUNNEL_PASS%"=="" (
echo   YOUR TUNNEL PASSWORD (COPIED TO CLIPBOARD):
echo   --^> %TUNNEL_PASS%
echo ============================================================
echo.
echo   HOW TO OPEN ON ANY DEVICE (Phone, Laptop, Tablet, 4G/5G):
echo   1. Send or open: https://%SUBDOMAIN%.loca.lt on any device
echo   2. When prompted for "Tunnel Password", paste: %TUNNEL_PASS%
echo   3. Click "Submit" to start diagnosing vehicle damage!
)
echo.
echo Starting tunnel now on port 5000...
echo Keep this window OPEN while sharing the demo!
echo (Press Ctrl+C to stop)
echo ============================================================
echo.

call npx --yes localtunnel --port 5000 --subdomain %SUBDOMAIN%
pause
