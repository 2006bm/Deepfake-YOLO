@echo off
rem ============================================================
rem  Baseline training launcher
rem
rem  Run this YOURSELF (double-click). Do NOT let an AI session
rem  launch it: processes started from such a session get killed
rem  by the Windows job object when the tool call ends. Verified
rem  with nohup / plain background / DETACHED_PROCESS /
rem  CREATE_NEW_CONSOLE -- all four were terminated mid-run.
rem
rem  Stop : close the window or press Ctrl-C. Completed epochs are
rem         kept; resume later with the --resume command shown at
rem         the end.
rem
rem  NOTE: this file is intentionally ASCII-only. cmd.exe parses
rem  batch files in the ANSI codepage until `chcp` takes effect,
rem  so non-ASCII text here can be split mid-character and break
rem  the script.
rem ============================================================
chcp 65001 >nul
cd /d "%~dp0.."

rem Use %LOCALAPPDATA% instead of hardcoding the interpreter path: the user
rem profile contains CJK characters, and cmd.exe reads this batch file in the
rem ANSI codepage (GBK) before `chcp 65001` takes effect -- a hardcoded UTF-8
rem CJK path would be decoded as GBK garbage and python would not be found.
rem The variable is expanded by cmd at runtime, in the correct encoding.
set PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe"

echo ================================================================
echo   Baseline Training - YOLOv8s-seg on FaceForensics++
echo ================================================================
echo   dir      %CD%
echo   data     16556 train / 3217 val
echo   config   YOLOv8s-seg  imgsz 640  batch 16  workers 4
echo            AdamW  lr0 0.001  epochs 100  patience 20
echo   log      logs_baseline_train.txt
echo   output   runs\segment\baseline_v1\
echo ----------------------------------------------------------------
echo   [1] Do NOT let the machine sleep during training:
echo       Settings - System - Power - Sleep - "When plugged in,
echo       put my device to sleep after" = Never
echo   [2] Keep it plugged in; battery power throttles the GPU.
echo   [3] For a clean progress panel, open a SECOND window and run:
echo           bash scripts/watch.sh
echo ----------------------------------------------------------------
echo   Press any key to start training...
pause >nul

rem Plain redirect -- do NOT use `tee`, cmd.exe does not have it
rem (that bug made a previous launch fail silently).
%PY% -u train\train_baseline.py --stage full --batch 16 --workers 4 > logs_baseline_train.txt 2>&1

echo.
echo ================================================================
echo   Training exited (code %ERRORLEVEL%)
echo ================================================================
if exist runs\segment\baseline_v1\results.csv (
    echo   results.csv found - epochs were recorded.
) else (
    echo   !! results.csv MISSING - no epoch completed.
    echo   !! Send logs_baseline_train.txt for analysis.
)
echo.
echo   To resume from the last checkpoint:
echo     %PY% train\train_baseline.py --stage full --batch 16 --workers 4 --resume
echo.
pause
