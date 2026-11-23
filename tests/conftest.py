"""Test configuration: a small, fast, isolated copy of the pipeline in a temp directory."""
import os
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="cfr-test-"))

_cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text())
_cfg["data"]["ibm_path"] = str(ROOT / _cfg["data"]["ibm_path"])
_cfg["data"]["synthetic"].update(n_customers=40, start="2021-01-01", end="2023-06-30", out_dir=str(TMP / "synthetic"))
_cfg["paths"] = {"artifacts": str(TMP / "artifacts"), "mlflow_uri": str(TMP / "mlruns")}
(TMP / "config.yaml").write_text(yaml.safe_dump(_cfg))

# Must be set before any cashflow_radar module reads the (cached) config or engine.
os.environ["CFR_CONFIG"] = str(TMP / "config.yaml")
os.environ["DATABASE_URL"] = f"sqlite:///{TMP / 'test.db'}"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
