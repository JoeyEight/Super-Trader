# Super Trader Incident Runbook

## 0. Preflight Before Shadow/Live
- Run readiness checks before major environment changes:
  - `python runtime/tools/preflight_readiness.py`
  - `python runtime/tools/preflight_readiness.py --strict`
- Report path:
  - `hub_data/preflight_readiness.json`

## 1. Runtime Not Starting
- Check `hub_data/runtime_startup_checks.json`.
- Fix any `errors` first.
- Verify script paths in `gui_settings.json`.

## 2. Broker/API Instability
- Check `hub_data/runtime_state.json`:
  - `api_quota`
  - `broker_health`
  - `execution_guard`
- If `execution_guard` is active, wait for cooldown and verify credentials/network.

## 3. Scanner Produces No Leaders
- Review:
  - `hub_data/stocks/scan_diagnostics.json`
  - `hub_data/forex/scan_diagnostics.json`
- Lower strictness in score/quality gates incrementally.

## 4. Trading Disabled in Live Mode
- Confirm checklist in runtime snapshot is green:
  - `checks.ok == true`
  - `alerts.severity == "ok"`
  - `api_quota.status != "critical"`

## 5. Key Hygiene
- Check `hub_data/key_rotation_status.json`.
- Rotate stale credentials when `due_count > 0`.

## 6. Core Logs
- `hub_data/logs/runner.log`
- `hub_data/logs/markets.log`
- `hub_data/logs/autopilot.log`
- `hub_data/runtime_events.jsonl`
- `hub_data/incidents.jsonl`

## 7. Scanner Cadence Troubleshooting
- Check cadence state in:
  - `hub_data/scanner_cadence_drift.json`
  - `hub_data/runtime_state.json -> scan_cadence`
- Severity interpretation:
  - `ok`: cadence near configured interval.
  - `warning`: loop is materially late; scanner freshness is degraded.
  - `critical`: loop lateness is severe and should be treated as execution risk.
- Troubleshooting sequence:
  1. Confirm `market_bg_stocks_interval_s` and `market_bg_forex_interval_s` are not too aggressive for current network/API conditions.
  2. Check API quota pressure (`runtime_state.api_quota`) and broker instability notes.
  3. Reduce scanner/trader cadence and retry once stable.

## 8. Universe Quality Tuning
- Primary files:
  - `hub_data/stocks/universe_quality.json`
  - `hub_data/forex/universe_quality.json`
  - `hub_data/stocks/scan_diagnostics.json`
  - `hub_data/forex/scan_diagnostics.json`
- Churn interpretation:
  - `candidate_churn_pct`: high values mean the candidate list is changing quickly cycle-to-cycle.
  - `leader_churn_pct`: high values mean top-ranked symbols/pairs are unstable.
- Practical tuning:
  1. If reject rate is high and dominant reason is `data_quality`, relax stale/valid-ratio thresholds slightly.
  2. If dominant reason is `spread`, tighten universe or reduce off-hours scanning.
  3. If churn stays high with low leader count, lower cadence pressure before widening universe breadth.

## 9. Retry-After Lockout Behavior
- Broker lockout waits are honored from API `Retry-After` signals (including long waits).
- Observe waits in:
  - `hub_data/runtime_events.jsonl` (`broker_retry_after_wait`)
  - `hub_data/runtime_state.json -> broker_backoff`
- When lockout pressure is high:
  1. Do not increase retry frequency.
  2. Let cooldown clear before forcing manual retries.
  3. Validate credentials/endpoints if waits persist abnormally.

## 10. Scanner Fallback Behavior (Stocks/Forex)
- Stocks and Forex scanners can now serve cached leaders when data/network fetch fails and prior scan data exists.
- Signals to watch:
  - `hub_data/stocks/stock_thinker_status.json -> fallback_cached = true`
  - `hub_data/forex/forex_thinker_status.json -> fallback_cached = true`
  - `hub_data/*/scan_diagnostics.json -> state=READY` with message containing `using cached scan`.
- Expected behavior:
  1. Chart and leader panels remain populated from last known-good scan.
  2. `health.data_ok` is set to `false` so UI/runtime alerts still surface degraded data.
  3. Trader safety gates remain active (no blind execution escalation).
  4. If enabled, new entries are blocked while fallback is active:
     - `stock_block_entries_on_cached_scan = true`
     - `forex_block_entries_on_cached_scan = true`
  5. Even when cached fallback entries are allowed, two additional guards apply:
     - hard-age block:
       - `stock_cached_scan_hard_block_age_s`
       - `forex_cached_scan_hard_block_age_s`
     - reduced-size fallback entries:
       - `stock_cached_scan_entry_size_mult`
       - `forex_cached_scan_entry_size_mult`
  6. Scanner-quality gates can block entries independently of fallback mode:
     - require data quality health:
       - `stock_require_data_quality_ok_for_entries`
       - `forex_require_data_quality_ok_for_entries`
     - reject-pressure threshold:
       - `stock_require_reject_rate_max_pct`
       - `forex_require_reject_rate_max_pct`
  7. Fallback leader stability can be tuned:
     - `stock_leader_stability_margin_pct`
     - `forex_leader_stability_margin_pct`

## 11. Log Storm Detection
- Stability audit now includes repeated-line spam detection.
- Run:
  - `python runtime/tools/stability_audit.py`
- Inspect:
  - `logs.level` (`ok`, `warning`, `critical`)
  - `logs.top_repeat_count`
  - `logs.top_repeat_line`
- If `logs.level=critical`, quality gates should be treated as failed until repetition is reduced.

## 12. Cached Fallback Age Severity
- Stability audit now escalates stale cached market data:
  - `fallback_cached=true` and `fallback_age_s >= 1800` -> market level becomes `critical`.
- Inspect in:
  - `python runtime/tools/stability_audit.py`
  - `markets.stocks.fallback_age_s`
  - `markets.forex.fallback_age_s`

## 13. Stability Audit
- Run:
  - `python runtime/tools/stability_audit.py`
  - `python runtime/tools/stability_audit.py --strict`
- The report summarizes:
  - runtime checks/alert severity
  - 24h incident severity counts
  - stocks/forex scanner level (`ok`/`warning`/`critical`)
  - chart-cache coverage and fallback activity
- Strict mode returns non-zero when stability pass criteria fail.
