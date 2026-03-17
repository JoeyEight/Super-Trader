# Developer Quality Gates

## Install Quality Tooling
```bash
python -m pip install -r requirements.txt
python -m pip install pytest ruff mypy types-requests pre-commit
```

## Run Full Quality Suite
```bash
python runtime/tools/run_quality_suite.py
```

This runs:
- `ruff` lint checks
- `mypy` type checks
- `unittest` test suite
- pass-3 artifact checker only when `--require-artifacts` is set
- runtime stability audit only when `--require-stability` is set

## Optional Fast Modes
```bash
python runtime/tools/run_quality_suite.py --skip-artifacts
python runtime/tools/run_quality_suite.py --skip-type
python runtime/tools/run_quality_suite.py --skip-tests
python runtime/tools/run_quality_suite.py --require-artifacts
python runtime/tools/run_quality_suite.py --require-artifacts --bootstrap-artifacts
python runtime/tools/run_quality_suite.py --require-artifacts --bootstrap-artifacts --require-stability
python runtime/tools/run_quality_suite.py --skip-stability
```

## Runtime Artifact Bootstrap
If your local `hub_data` is stale or missing pass checks:

```bash
python runtime/tools/bootstrap_runtime_artifacts.py
```

## Runtime Stability Audit
```bash
python runtime/tools/stability_audit.py
python runtime/tools/stability_audit.py --strict
python runtime/tools/stability_audit.py --write
```

Notes:
- Stability output now includes `logs` repetition metrics (`level`, `top_repeat_count`, `top_repeat_line`).
- `--strict` fails when log spam is classified as `critical`.
