@echo off
echo ============================================================
echo   Vehicle Damage Detection - Live Public Demo Tunnel
echo ============================================================
echo.
echo Make sure "python app.py" is running in another window!
echo.
echo Your Tunnel IP Password is:
curl.exe -s https://loca.lt/mytunnelpassword
echo.
echo Starting live public tunnel on port 5000...
echo.
call npx --yes localtunnel --port 5000
pause
