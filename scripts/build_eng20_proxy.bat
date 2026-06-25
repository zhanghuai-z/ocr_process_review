@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars32.bat" >nul
cl /nologo /LD /EHsc D:\project\ocr_process\worktrees\coord\scripts\eng20_proxy.cpp /link /DEF:D:\project\ocr_process\worktrees\coord\scripts\eng20_proxy.def /OUT:D:\project\ocr_process\worktrees\coord\debug\eng20_hook_bin\Eng20.dll
