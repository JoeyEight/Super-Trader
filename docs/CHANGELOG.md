# Changelog

## v0.8.9 - 2026-03-09
- Added live scanner threshold self-tuning for Stocks and Forex:
  - thinkers now blend volatility-based thresholding with rejection-replay recommendations
  - blend is bounded by per-cycle step caps to avoid abrupt threshold swings in production loops
  - final threshold components are persisted (`base`, `volatility`, `replay`, `weight`) for diagnostics
- Added new settings (defaults + sanitization + UI controls + profile presets):
  - `stock_replay_adaptive_enabled`
  - `stock_replay_adaptive_weight`
  - `stock_replay_adaptive_step_cap_pct`
  - `forex_replay_adaptive_enabled`
  - `forex_replay_adaptive_weight`
  - `forex_replay_adaptive_step_cap_pct`
- Added reusable replay helper APIs in `app/rejection_replay.py` so scanner loops and replay reports use the same recommendation logic.
- Improved Stocks/Forex Overview panel text to show adaptive-threshold tuning context (base/volatility/replay/weight) for faster operator debugging.
- Added tests for replay helper behavior and new settings clamps/sanitization.

## v0.8.8 - 2026-03-09
- Hardened AI Assist no-patch handling to prevent indefinite `OPEN` ticket states:
  - added structured no-patch reasons (`llm_output_invalid`, `llm_no_patch_generated`, etc.)
  - added retry/backoff initialization when proposal generation fails without a diff
  - added retry exhaustion blocking for repeated no-patch/invalid-output responses
- Added AI Assist UX improvements:
  - `Retry Ticket` action to re-run proposal generation for the selected ticket ID
  - expanded reason/help messaging in Patch Preview for blocked and retry states
  - extended human-readable apply reason mapping for new failure codes
- Added direct Autofix retry command path for existing tickets:
  - `runtime/pt_autofix.py --retry-ticket <id> [--retry-auto-apply]`
  - AI Assist Retry button now re-runs proposal generation on the selected ticket ID.
- Added command-center badge actions:
  - runtime badge click -> quick diagnostics
  - APIs badge click -> settings
  - checks/alerts badge click -> alerts panel
- Added new Autofix settings defaults/sanitization keys for no-patch/invalid-output block policy:
  - `autofix_request_block_on_invalid_output`
  - `autofix_request_block_on_no_patch`
- Expanded Autofix runtime tests to cover no-patch reason classification and retry exhaustion blocking.

## v0.8.7 - 2026-03-09
- Added rejected-candidate replay engine for scanner threshold tuning:
  - `app/rejection_replay.py`
  - `runtime/tools/replay_rejections.py`
- Added hub integrations for replay analysis:
  - command palette actions
  - toolbar/menu actions
  - replay report viewer with threshold scenarios and reject-reason breakdowns
- Added markdown operator-notes workflow with timestamped JSONL operator logs:
  - `app/operator_notes.py`
  - dedicated notes editor in hub with save/reload/log-entry actions
- Added new settings defaults/sanitization for replay target entries and operator-note history window.
- Added unit tests for replay analytics and operator notes persistence.
- Expanded diagnostics bundle/smoke report to include replay and operator-notes artifacts.

## v0.8.6 - 2026-03-05
- Added preflight readiness checker utility: `runtime/tools/preflight_readiness.py`.
- Preflight now validates script paths, writable runtime directories, rollout/mode posture, broker credential presence, and key hygiene reminders.
- Added quality-suite integration for readiness gating:
  - `runtime/tools/run_quality_suite.py --require-preflight`
- Added unit tests for preflight readiness reporting.
- Rewrote `README.md` to reflect current package layout and shadow/live validation workflow.

## v0.8.5 - 2026-03-05
- Added Stocks/Forex scanner-quality entry gates:
  - require thinker `health.data_ok` before new entries (`stock_require_data_quality_ok_for_entries`, `forex_require_data_quality_ok_for_entries`)
  - reject-pressure cutoff gating (`stock_require_reject_rate_max_pct`, `forex_require_reject_rate_max_pct`)
- Added cached-fallback hard block age controls:
  - `stock_cached_scan_hard_block_age_s`
  - `forex_cached_scan_hard_block_age_s`
- Added cached-fallback reduced-size controls (used only when cached fallback entries are explicitly allowed):
  - `stock_cached_scan_entry_size_mult`
  - `forex_cached_scan_entry_size_mult`
- Added trader telemetry for gate-state + effective entry sizing:
  - `entry_gate_flags`
  - `trade_notional_entry_usd` / `trade_units_entry`
  - `entry_size_scale`
- Surfaced gate-state and reduced-size hints in Stocks/Forex dashboard state lines for faster operator diagnostics.
- Added unit coverage for data-quality gating, reject-pressure gating, and cached-fallback reduced-size behavior in Stocks/Forex traders.

## v0.8.4 - 2026-03-05
- Added leader stability hysteresis for Stocks/Forex scanners (`stock_leader_stability_margin_pct`, `forex_leader_stability_margin_pct`) to reduce top-pick churn when candidates are near-tied.
- Added cached-scan execution safety gates in Stocks/Forex traders (`stock_block_entries_on_cached_scan`, `forex_block_entries_on_cached_scan`) to block new entries while thinker data is in fallback mode.
- Added trader entry-gate telemetry (`entry_eval_top_reason`, `entry_eval_reason_counts`) and surfaced it in market state text for faster diagnostics.
- Added fallback-age critical classification in stability audit (`fallback_cached` + stale age escalates market severity).
- Added Stocks cached-fallback scan mode (parity with Forex) to preserve leaders/charts when scanner data paths degrade.
- Added Stocks leader publication mode with watch-fallback output when no long setups are available (`leader_mode` surfaced to UI/runtime).
- Added runtime thinker/snapshot cached-fallback handling in `pt_markets` to reduce hard-error flapping during transient broker/API failures.
- Added crypto feed hardening in `pt_thinker`: unsupported KuCoin-pair cooldown suppression, non-retryable price-error fast-fail, and cached-price fallback.
- Added throttled logger utility (`log_throttled`) and applied it to noisy crypto feed paths.
- Added new safety/tuning settings:
  - `market_fallback_scan_max_age_s`
  - `market_fallback_snapshot_max_age_s`
  - `kucoin_unsupported_cooldown_s`
  - `crypto_price_error_log_cooldown_s`
  - `stock_scan_publish_watch_leaders`
  - `stock_scan_watch_leaders_count`
- Extended stability audit with log-spam detection (`logs.level`) and fail-gate on critical log storms.
- Added tests for stock scanner fallback/watch-leader behavior, fallback age guard helpers, log-throttle behavior, and new settings clamps.

## v0.8.3 - 2026-03-05
- Added Stocks/Forex multi-symbol chart cache output (`top_chart_map`) for stronger cross-panel chart consistency.
- Added focus-aware Stocks/Forex chart rendering and export fallback logic to use cached map bars before network fetch.
- Added Forex scanner cached-fallback mode on network/HTTP failures to preserve leaders/charts during transient outages.
- Added adaptive loss-streak size scaling controls for Stocks and Forex execution (`*_loss_streak_size_step_pct`, `*_loss_streak_size_floor_pct`).
- Added runtime stability audit utility (`runtime/tools/stability_audit.py`) with strict pass/fail mode.
- Added market trend chart coverage metrics (`chart_coverage`) including cache symbol count and fallback flag.
- Added quality-runner flags for strict stability auditing (`--require-stability`, `--skip-stability`).
- Added tests covering Forex scanner fallback mode, stability audit behavior, and new settings clamps.
- Added scanner cadence drift tracking (`scanner_cadence_drift.json`) with warning/critical alert hooks.
- Added scanner churn diagnostics (candidate/leader churn %) for stocks and forex.
- Added per-market universe quality reports with gate pass summaries and reject/source breakdowns.
- Added stocks open/close scanner score dampening controls and forex session-aware score weighting controls.
- Added retry-after lockout honoring in broker order retry backoff (including long lockout windows).
- Added runtime-state surfacing for cadence drift and market-loop age/cadence snapshots.
- Expanded Stocks/Forex panel UX with churn/cadence visibility and quality summary hints.
- Added File menu export for scanner quality JSON bundle.
- Added pass-3 release checklist and pass-3 artifact checker utility.
- Expanded unit coverage for cadence drift, scanner quality helpers, session/window policies, and retry-after behavior.
- Added runtime event emission and runtime-state summary rollups for broker retry-after waits.
- Added stale market-loop watchdog incident signaling and alert-rule freshness thresholds.
- Added scan diagnostics schema v2 + compatibility normalization for legacy readers.
- Added trend payload quality/cadence aggregate fields for stocks and forex.
- Added retention cleanup for stale scanner-quality export artifacts.
- Added one-command quality runner (`runtime/tools/run_quality_suite.py`) and documented quality-tool installs.
- Added runtime artifact bootstrap utility (`runtime/tools/bootstrap_runtime_artifacts.py`) for legacy/missing hub_data files.

## v0.8.0 - 2026-03-05
- Added runtime quota monitoring and broker health scoreboard.
- Added execution guard (temporary disable after repeated broker failures).
- Added quick-fix suggestions in runtime alerts/UI.
- Added live-mode checklist gate + confirmation flow.
- Added key-rotation reminder status output.
- Added lint/type/format/pre-commit/CI baseline files.
- Added runbook, handoff guide, and release checklist docs.
- Added maintenance utilities for stale artifacts and diagnostics archives.

## v0.8.1 - 2026-03-05
- Added settings schema migrations with automatic upgrade notes.
- Added global drawdown circuit-breaker and per-market daily guardrail tile.
- Added first-run onboarding wizard and contextual risk tooltips in settings.
- Added market-awareness notes (maintenance, stock hours, forex session bias) in dashboard panels.
- Added chart PNG export for crypto account/coin charts and stocks/forex market charts.
- Improved empty-state UX for chart and table views.
- Added integration tests for runner lifecycle, smoke harness outputs, and UI status hydration.
- Added deterministic scanner/trader fixtures and missing-file/no-JSON robustness tests.
- Enabled stricter mypy requirements for runtime modules.

## v0.8.2 - 2026-03-05
- Added market-loop telemetry heartbeat (`market_loop_status.json`) with cycle timing metadata.
- Added jittered market-loop scheduling and throttled settings reload controls.
- Added runtime-state visibility for drawdown guard, stop flag, and market loop freshness.
- Added incident trend counters (`count_1h`, `count_24h`) in runtime incident summary.
- Added JSONL retention trimming for incidents/runtime events with configurable line caps.
- Extended market awareness with countdown/session transition fields and maintenance levels.
- Extended alerting with drawdown/stop-flag critical reasons and quick-fix guidance.
- Added hub safety/status UX: loop freshness, stop-flag/drawdown indicators, and incident-1h count.
- Added new operator exports: market status snapshot JSON and runtime summary TXT.
- Added one-click quick diagnostics action in hub (smoke harness trigger + report path).
- Added pass-2 backlog tracker and expanded tests for awareness, health rules, and runtime logging.
