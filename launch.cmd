@echo off
cd /d "%~dp0"
where pythonw.exe >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  start "" /wait pythonw.exe "%~dp0single_page_launcher.py"
) else (
  python "%~dp0single_page_launcher.py"
)
