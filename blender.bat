@echo off
set HOME=%~dp0
set PYTHONPATH=%PYTHONPATH%;%HOME%\packages;%HOME%\startup
set BLENDER_SYSTEM_SCRIPTS=%HOME%\startup;%BLENDER_SYSTEM_SCRIPTS%
blender.exe -con --log-level -1 --python %HOME%\startup\startup.py