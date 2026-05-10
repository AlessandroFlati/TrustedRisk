' Launch a PowerShell script with a hidden console window.
'
' Usage:
'   wscript.exe run_hidden.vbs <path-to-ps1> [extra-args...]
'
' Why this exists:
'   `powershell.exe -WindowStyle Hidden` still flashes a console window
'   for ~50 ms when launched from Task Scheduler in the user's session,
'   because the host process (powershell.exe) attaches a console before
'   honouring -WindowStyle. WScript has no console, so spawning
'   PowerShell from inside a WScript host (with bWaitOnReturn=False and
'   intWindowStyle=0) yields no flash. This is the documented Windows
'   pattern for silent scheduled PowerShell tasks.

If WScript.Arguments.Count < 1 Then
    WScript.Quit 1
End If

Dim psPath, extra, i, cmd
psPath = WScript.Arguments(0)
extra = ""
For i = 1 To WScript.Arguments.Count - 1
    extra = extra & " """ & WScript.Arguments(i) & """"
Next

cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & psPath & """" & extra

CreateObject("WScript.Shell").Run cmd, 0, False
