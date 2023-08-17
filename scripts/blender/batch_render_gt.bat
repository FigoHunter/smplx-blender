@echo off
set HOME=%~dp0..\..
set PYTHONPATH=%PYTHONPATH%;%HOME%\packages;%HOME%\startup
set BLENDER_SYSTEM_SCRIPTS=%HOME%\startup;%BLENDER_SYSTEM_SCRIPTS%
blender.exe --background --log-level -1 --python %HOME%\scripts\blender\batch_render_gt.py
pause