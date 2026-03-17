# Desktop Launcher Packaging

This folder contains launcher/install scripts to make Super Trader appear like a normal desktop application.

## Generate Icon Assets

```bash
./venv/bin/python packaging/generate_app_icons.py
```

Generated files:
- `packaging/icons/super_trader.png`
- `packaging/icons/super_trader.ico`
- `packaging/icons/super_trader.icns` (when ICNS encoding is available)

## macOS: Install into Applications

```bash
./venv/bin/python packaging/install_macos_app.py
```

Result:
- `/Applications/Super Trader.app` (double-clickable app bundle with icon)
- Bundle launches `launch_super_trader.command` from this repository.

## Windows: Install Start Menu Launcher

From PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\install_windows_shortcut.ps1
```

Optional desktop shortcut:

```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\install_windows_shortcut.ps1 -DesktopShortcut
```

Result:
- `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Super Trader\Super Trader.lnk`
- `%LOCALAPPDATA%\Programs\Super Trader\launch_super_trader.bat`
