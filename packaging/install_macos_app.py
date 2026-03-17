from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _generate_icons(project_dir: Path, out_dir: Path) -> Path:
    script = project_dir / "packaging" / "generate_app_icons.py"
    cmd = [sys.executable, str(script), "--out-dir", str(out_dir)]
    subprocess.run(cmd, check=True)
    icns = out_dir / "super_trader.icns"
    if not icns.exists():
        raise RuntimeError("ICNS icon was not generated; cannot install macOS app bundle.")
    return icns


def main() -> int:
    parser = argparse.ArgumentParser(description="Install Super Trader.app into /Applications.")
    parser.add_argument("--project-dir", default="", help="Path to Super Trader repository root.")
    parser.add_argument("--app-name", default="Super Trader", help="Application display name.")
    parser.add_argument(
        "--install-dir",
        default="/Applications",
        help="Target install directory for .app bundle (default: /Applications).",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        raise RuntimeError("install_macos_app.py must be run on macOS.")

    script_dir = Path(__file__).resolve().parent
    project_dir = Path(args.project_dir).expanduser().resolve() if str(args.project_dir).strip() else script_dir.parent
    install_dir = Path(args.install_dir).expanduser().resolve()
    app_name = str(args.app_name or "Super Trader").strip() or "Super Trader"
    app_bundle = install_dir / f"{app_name}.app"

    launch_script = project_dir / "launch_super_trader.command"
    if not launch_script.exists():
        raise RuntimeError(f"Missing launcher script: {launch_script}")

    icon_dir = project_dir / "packaging" / "icons"
    icns_path = _generate_icons(project_dir, icon_dir)

    contents_dir = app_bundle / "Contents"
    macos_dir = contents_dir / "MacOS"
    resources_dir = contents_dir / "Resources"
    info_plist = contents_dir / "Info.plist"
    exe_name = "SuperTraderLauncher"
    launcher_path = macos_dir / exe_name

    if app_bundle.exists():
        shutil.rmtree(app_bundle)
    resources_dir.mkdir(parents=True, exist_ok=True)
    macos_dir.mkdir(parents=True, exist_ok=True)

    launcher_text = (
        "#!/bin/zsh\n"
        "set -euo pipefail\n"
        f'PROJECT_DIR="{str(project_dir)}"\n'
        'exec "$PROJECT_DIR/launch_super_trader.command"\n'
    )
    _write_text(launcher_path, launcher_text)
    _make_executable(launcher_path)

    shutil.copy2(icns_path, resources_dir / "super_trader.icns")

    plist_text = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDisplayName</key><string>{app_name}</string>
  <key>CFBundleExecutable</key><string>{exe_name}</string>
  <key>CFBundleIconFile</key><string>super_trader.icns</string>
  <key>CFBundleIdentifier</key><string>com.supertrader.app</string>
  <key>CFBundleName</key><string>{app_name}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSBackgroundOnly</key><false/>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
"""
    _write_text(info_plist, plist_text)

    print(f"[install] app_bundle={app_bundle}")
    print("[install] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
