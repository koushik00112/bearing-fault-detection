.PHONY: install lint test smoke download-sample download prepare results prepare-32k results-32k serve-fixture

install:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev,cnn,tracking]"

lint:
	.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy bearing

test:
	.venv/bin/pytest --cov=bearing --cov-report=term-missing

# Whole pipeline on synthetic data, no download needed. Not a result.
smoke:
	.venv/bin/python -m bearing.prepare synthetic --out data/processed/synthetic
	.venv/bin/python -m bearing.run --data data/processed/synthetic --out results/synthetic-smoke --quick --seeds 0,1

# Two bearings (~330 MB): check the loader against real files before the full download.
download-sample:
	.venv/bin/python -m bearing.data.download paderborn --bearings K001,KA01

download:
	.venv/bin/python -m bearing.data.download paderborn --all
	.venv/bin/python -m bearing.data.download cwru

prepare:
	.venv/bin/python -m bearing.prepare paderborn --raw data/raw/paderborn --out data/processed/paderborn

results:
	.venv/bin/python -m bearing.run --data data/processed/paderborn --out results/paderborn

# Ablation (ADR 0002): same 128 ms windows at 32 kHz, keeping the 8-16 kHz band.
prepare-32k:
	.venv/bin/python -m bearing.prepare paderborn --raw data/raw/paderborn --out data/processed/paderborn-32k --decimate 2 --window 4096 --stride 2048

# B and B2 only: the question in ADR 0002 is whether 8-16 kHz helps with or without the
# bearing-identity shortcut. All five scenarios at 32 kHz would add ~4 h of CNN training.
results-32k:
	.venv/bin/python -m bearing.run --data data/processed/paderborn-32k --out results/paderborn-32k --scenarios B_recording,B2_bearing

serve-fixture:
	.venv/bin/python scripts/fixture_model.py models/fixture
	MODEL_DIR=models/fixture .venv/bin/uvicorn bearing.serve.app:app --port 8100
