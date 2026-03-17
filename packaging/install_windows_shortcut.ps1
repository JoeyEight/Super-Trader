param(
    [string]$ProjectDir = "",
    [switch]$DesktopShortcut
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ProjectDir)) {
    $ProjectDir = Split-Path -Parent $ScriptDir
}
$ProjectDir = (Resolve-Path $ProjectDir).Path

$PyBin = Join-Path $ProjectDir "venv\Scripts\python.exe"
if (-not (Test-Path $PyBin)) {
    $PyBin = "python"
}

$IconScript = Join-Path $ProjectDir "packaging\generate_app_icons.py"
$IconOutDir = Join-Path $ProjectDir "packaging\icons"
& $PyBin $IconScript --out-dir $IconOutDir | Out-Host

$ProgramsDir = Join-Path $env:LOCALAPPDATA "Programs\Super Trader"
New-Item -ItemType Directory -Path $ProgramsDir -Force | Out-Null

$LaunchBat = Join-Path $ProjectDir "launch_super_trader.bat"
$InstalledBat = Join-Path $ProgramsDir "launch_super_trader.bat"
Copy-Item -Path $LaunchBat -Destination $InstalledBat -Force

$IconIco = Join-Path $IconOutDir "super_trader.ico"
$InstalledIco = Join-Path $ProgramsDir "super_trader.ico"
if (Test-Path $IconIco) {
    Copy-Item -Path $IconIco -Destination $InstalledIco -Force
}

$StartMenuDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\Super Trader"
New-Item -ItemType Directory -Path $StartMenuDir -Force | Out-Null
$ShortcutPath = Join-Path $StartMenuDir "Super Trader.lnk"

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $InstalledBat
$Shortcut.WorkingDirectory = $ProjectDir
if (Test-Path $InstalledIco) {
    $Shortcut.IconLocation = "$InstalledIco,0"
}
$Shortcut.Description = "Launch Super Trader"
$Shortcut.Save()

if ($DesktopShortcut) {
    $DesktopPath = [Environment]::GetFolderPath("Desktop")
    $DesktopLink = Join-Path $DesktopPath "Super Trader.lnk"
    Copy-Item -Path $ShortcutPath -Destination $DesktopLink -Force
}

Write-Host "[install] start_menu_shortcut=$ShortcutPath"
Write-Host "[install] programs_dir=$ProgramsDir"
Write-Host "[install] done"
