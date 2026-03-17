@echo off
setlocal EnableExtensions
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%launch_super_trader.bat"
exit /b %errorlevel%
