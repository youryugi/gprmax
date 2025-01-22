@echo off
setlocal
set "SCRIPT_PATH=C:\Users\scope\Desktop\rss-yang\scanWithBat.py"
set "USERNAME=scope"
set "PASSWORD=scope"

rem List of target IP addresses
set "TARGETS=192.168.1.20"

for %%I in (%TARGETS%) do (
    echo Executing on %%I
    net use \\%%I\ipc$ /user:%USERNAME% %PASSWORD%
    if %errorlevel% neq 0 (
        echo Failed to connect to %%I
        net use \\%%I\ipc$ /delete
        goto end
    )
    psexec \\%%I -u %USERNAME% -p %PASSWORD% cmd /c "python %SCRIPT_PATH%"
    net use \\%%I\ipc$ /delete
)

:end
endlocal
pause
