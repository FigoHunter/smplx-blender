@echo off
set WORKSPACE_HOME=%~dp0..\..
set PYTHONPATH=%PYTHONPATH%;%WORKSPACE_HOME%\packages;%WORKSPACE_HOME%\startup
set BLENDER_SYSTEM_SCRIPTS=%WORKSPACE_HOME%\startup;%BLENDER_SYSTEM_SCRIPTS%
blender.exe --background --log-level -1 --python %WORKSPACE_HOME%\scripts\blender\batch_render_gt.py
pause