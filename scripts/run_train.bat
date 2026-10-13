@echo off
rem Full baseline training (restart). ASCII-only + CRLF: cmd.exe parses batch
rem files in the ANSI codepage, so CJK paths must come from env vars.
chcp 65001 >nul
cd /d "%~dp0.."
set PY="%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
echo [train] starting at %DATE% %TIME% > logs_train.txt
%PY% -u train\train_baseline.py --stage full --batch 16 --workers 4 --save-period 10 >> logs_train.txt 2>&1
echo [train] exited with code %ERRORLEVEL% at %DATE% %TIME% >> logs_train.txt
