# Super Trader

Super Trader is a multi-market trading hub with three coordinated engines:
- Crypto (live-capable)
- Stocks (Alpaca-backed)
- Forex (OANDA-backed)

The desktop hub UI implementation is in `ui/hub/main.py` (legacy compatibility shim at `ui/pt_hub.py`), with runtime orchestration in `runtime/pt_runner.py`.
Crypto coin workspaces are now managed under `market_data/coins` by default (auto-migrated from legacy root-level coin folders when needed).
Primary repository: `https://github.com/JoeyEight/Super-Trader.git`

## Overview

Super Trader is a desktop-based, multi-market trading application designed to centralize trading activity across crypto, stocks, and forex from one coordinated hub. The app provides a unified interface for monitoring runtime status, launching and stopping trading services, reviewing diagnostics, managing market-specific engines, and coordinating advisory tools without requiring the operator to manage each market separately.

The application is structured around three primary trading engines:

- **Crypto**: live-capable crypto trading runtime with coin-specific workspaces and model/data directories.
- **Stocks**: Alpaca-backed stock trading workflow with scanner, thinker, trader, watchlist, and diagnostics support.
- **Forex**: OANDA-backed forex trading workflow with macro-event awareness, scanner/trader status files, and execution safeguards.

Super Trader is built to act as an operator-controlled trading hub rather than a fully autonomous black box. Runtime services are supervised through the hub and supporting runner modules, while safety gates, readiness checks, broker credentials, exposure controls, diagnostics, and paper/live configuration remain the responsibility of the user. Optional OpenAI-powered modules are advisory only and do not place trades directly or bypass the local trading guards.

This version expands the original concept into a broader multi-market desktop trading hub with updated runtime orchestration, broker integrations, diagnostics, readiness tooling, OpenAI advisory hooks, and market-specific runtime structure.

## Attribution

Super Trader was inspired by the original **PowerTrader_AI** project created by **Stephen Hughes**.

Original project:
`https://github.com/garagesteve1155/PowerTrader_AI`

Credit and appreciation go to Stephen Hughes for the original codebase and concept that helped inspire this version of the app.

## Safety First
- This software can place real orders when configured for live mode.
- Keep `paper_only_unless_checklist_green=true` until checklist is green.
- Review all broker credentials, risk caps, and exposure controls before enabling auto-trade.

## Current Repo Layout
- `app/` shared utilities (settings, paths, health, runtime helpers)
- `brokers/` Alpaca/OANDA broker adapters
- `engines/` thinker/trader/trainer modules
- `runtime/` process orchestration (`pt_runner`, `pt_markets`, `pt_autopilot`)
- `ui/` Tkinter hub application
- `tests/` unit/integration tests
- `docs/` runbook/changelog/checklists
- `hub_data/` runtime output (status, logs, diagnostics, incidents)
- `market_data/coins/<COIN>/` coin-specific model/data directories (crypto training/runtime)

## Install
Recommended Python: `3.10.x`

macOS / Linux:
```bash
python3 -m venv venv
./venv/bin/python -m pip install --upgrade pip setuptools wheel
./venv/bin/python -m pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
py -3 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you use the launcher, it will create `venv/` automatically and install missing core dependencies before opening the hub.

Runtime packages currently used by the app code:
- `requests`, `kucoin-python`, `PyNaCl`, `cryptography`, `colorama`
- `matplotlib`, `psutil`, `Pillow` (desktop icon/app packaging path)
- `python-dotenv`, `pandas` (used by `sources/*` helpers)
- `orjson` (fast JSON decode/encode), `watchdog` (event-driven file change signals), `prometheus-client` (optional metrics hooks)
- OpenAI advisory services use `requests` with the Responses API (no separate `openai` SDK dependency is required in this repo)
- Test/dev helpers: `pytest`, `pytest-xdist`, `py-spy`, `uv`

`requirements.txt` is split into:
- Core runtime dependencies (needed to run hub + markets)
- Test/diagnostics/tooling dependencies (for smoke tests and profiling)

## Run
### Hub UI
macOS / Linux:
```bash
./launch_super_trader.command
```

Windows (Command Prompt / PowerShell):
```bat
launch_super_trader.bat
```

Alternative:
```bash
./venv/bin/python -m ui.pt_hub
```

Important runtime behavior:
- `Start Trades` launches the runtime supervisor as a detached background process.
- Closing the hub window does not stop trading by itself.
- Use `Stop Trades` first if you want crypto, stocks, and forex runtime activity to stop cleanly.
- If the computer goes to sleep, the Python processes pause and do not keep trading until the machine wakes again.

## Desktop App Install (Icon + Applications/Programs)
### macOS (`/Applications`)
```bash
./venv/bin/python packaging/install_macos_app.py
```

This installs `/Applications/Super Trader.app` with an icon and launches the repo's `launch_super_trader.command`.

### Windows (Start Menu / Programs)
```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\install_windows_shortcut.ps1
```

Optional desktop icon:
```powershell
powershell -ExecutionPolicy Bypass -File .\packaging\install_windows_shortcut.ps1 -DesktopShortcut
```

This creates:
- `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Super Trader\Super Trader.lnk`
- `%LOCALAPPDATA%\Programs\Super Trader\launch_super_trader.bat`

### Runner (all background services)
```bash
./venv/bin/python -m runtime.pt_runner
```

### Markets loop only (scanner + market trader steps)
```bash
./venv/bin/python -m runtime.pt_markets
```

### Autopilot one-shot tune
```bash
./venv/bin/python -m runtime.pt_autopilot --once
```

## Preflight (Run Before Live)
Use the readiness checker before testing:
```bash
python3 runtime/tools/preflight_readiness.py
```

Strict mode fails on warnings and critical issues:
```bash
python3 runtime/tools/preflight_readiness.py --strict
```

Report output default:
- `hub_data/preflight_readiness.json`

## Quality Suite
```bash
python3 runtime/tools/run_quality_suite.py
```

Optional strict gates:
```bash
python3 runtime/tools/run_quality_suite.py --require-artifacts --require-stability --require-preflight
```

## Smoke Test
Quick end-to-end runtime smoke check:
```bash
./venv/bin/python runtime/smoke_test_all.py
```

Output report:
- `hub_data/smoke_test_report.json`

## Key Runtime Files
### Core status
- `hub_data/trader_status.json`
- `hub_data/runner_ready.json`
- `hub_data/runner.pid`
- `hub_data/runtime_state.json`
- `hub_data/market_loop_status.json`
- `hub_data/runtime_startup_checks.json`
- `hub_data/notification_center.json`

### Market status
- `hub_data/stocks/stock_thinker_status.json`
- `hub_data/stocks/stock_trader_status.json`
- `hub_data/forex/forex_thinker_status.json`
- `hub_data/forex/forex_trader_status.json`
- `hub_data/forex/forexfactory_calendar_cache.json`

### Diagnostics
- `hub_data/stocks/scan_diagnostics.json`
- `hub_data/forex/scan_diagnostics.json`
- `hub_data/stocks/universe_quality.json`
- `hub_data/forex/universe_quality.json`
- `hub_data/scanner_cadence_drift.json`
- `hub_data/runtime_events.jsonl`
- `hub_data/incidents.jsonl`

### Logs
- `hub_data/logs/runner.log`
- `hub_data/logs/markets.log`
- `hub_data/logs/autopilot.log`
- `hub_data/logs/thinker.log`
- `hub_data/logs/trader.log`

### UI behavior to expect
- Stocks and forex run from the same detached runtime supervisor used by crypto.
- Stocks and forex use the native in-app charts and watchlists; the old TradingView launch path is not part of the active UI flow.
- Notification Center reflects live runtime state plus recent unresolved incidents; stale resolved incidents are filtered out by the current app code.
- Forex `event_feed` warnings indicate macro-event feed degradation and do not by themselves disable forex trading; execution gates still apply independently.
- OpenAI advisory services are optional and fail closed (local logic remains authoritative on errors/timeouts/missing key).
- For lowest CPU usage, leave OpenAI advisory/review services disabled unless you are actively using them.
- Market modules can be enabled/disabled per user in Settings (`Enabled markets`).
- Disabled markets are hidden from the main tabs and from market-specific settings panels.
- Disabled markets do not emit missing-credential startup warnings or reject-pressure runtime alert noise.
- If all markets are disabled accidentally, the app auto-enables Crypto to keep the UI/runtime usable.

## OpenAI (Optional Advisory Layer)
All OpenAI-powered modules are advisory only. They do not place trades directly and do not bypass local guards.

Configure from the hub Settings -> OpenAI key/editor popup:
- `openai_decision_enabled` (cross-market decision advisory)
- `openai_position_review_enabled`
- `openai_capital_planner_enabled`
- `openai_root_cause_enabled`
- `openai_explanations_enabled`
- `openai_strategy_optimizer_enabled`
- `openai_market_context_enabled`
- `openai_postmortem_enabled`
- `openai_nightly_review_enabled`

Runtime output files:
- `hub_data/openai/*_status.json`
- `hub_data/openai/*.json`

## Credentials
### Crypto (Robinhood)
- Preferred files:
  - `keys/r_key.txt`
  - `keys/r_secret.txt`
- Or env:
  - `POWERTRADER_RH_API_KEY`
  - `POWERTRADER_RH_PRIVATE_B64`

### Stocks (Alpaca)
- Env or settings-backed values:
  - `POWERTRADER_ALPACA_API_KEY_ID`
  - `POWERTRADER_ALPACA_SECRET_KEY`

### Stocks Data (Twelve Data, optional)
- Set `stock_data_provider` to `twelvedata` in Settings.
- Provide the API key via:
  - `POWERTRADER_TWELVEDATA_API_KEY`, or
  - `keys/twelvedata_api_key.txt`
- Rate guard knobs:
  - `twelvedata_api_credits_per_minute`
  - `twelvedata_daily_credits`
  - `twelvedata_scan_symbol_cap`

### Forex (OANDA)
- Env or settings-backed values:
  - `POWERTRADER_OANDA_ACCOUNT_ID`
  - `POWERTRADER_OANDA_API_TOKEN`

## Operator Notes
- Changelog: [docs/CHANGELOG.md](docs/CHANGELOG.md)
- Runbook: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Settings migration notes: [docs/SETTINGS_MIGRATIONS.md](docs/SETTINGS_MIGRATIONS.md)

## Disclaimer
Use at your own risk. You are responsible for all broker/account configuration, risk limits, and any resulting trades.
