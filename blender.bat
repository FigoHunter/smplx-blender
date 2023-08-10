@echo off
set HOME=%~dp0
set PYTHONPATH=%PYTHONPATH%;%HOME%\packages
blender.exe -con --log-level -1