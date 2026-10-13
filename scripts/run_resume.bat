@echo off
rem Resume baseline training from the last checkpoint (epoch 51 -> up to 100/early-stop).
rem ASCII-only on purpose: cmd.exe parses batch files in the ANSI codepage, so a
rem hardcoded UTF-8 CJK path would be decoded as garbage and python not found.
chcp 65001 >nul
cd /d "%~dp0.."
set PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
echo [resume] starting at %DATE% %TIME% > logs_resume.txt
%PY% -u train\train_baseline.py --stage full --batch 16 --workers 4 --resume --name baseline_v1-2 >> logs_resume.txt 2>&1
echo [resume] exited with code %ERRORLEVEL% at %DATE% %TIME% >> logs_resume.txt
