Set fso = CreateObject("Scripting.FileSystemObject")
Set ws = CreateObject("WScript.Shell")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
ws.CurrentDirectory = fso.GetParentFolderName(scriptDir)
ws.Run "scripts" & Chr(92) & "run_resume.bat", 0, False
