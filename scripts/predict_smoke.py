"""Send one synthetic window of each class to a running model service and check the reply.

python scripts/predict_smoke.py http://localhost:8100
"""

import sys

import httpx

from bearing.data import synthetic

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
with httpx.Client(base_url=url, timeout=30) as c:
    info = c.get("/readyz").raise_for_status().json()
    print("ready:", info)
    windows = []
    for label in ("healthy", "inner", "outer"):
        rec = synthetic.make_recording(label, "X", "c1500", 0, seed=42, seconds=0.2)
        windows.append({"channels": {k: v[:2048].tolist() for k, v in rec.signals.items()}})
    r = c.post("/predict", json={"fs": 16000.0, "windows": windows}).raise_for_status().json()
    for label, p in zip(("healthy", "inner", "outer"), r["predictions"], strict=True):
        pred, fault_p = p["predicted_class"], p["fault_probability"]
        print(f"true={label:<8} predicted={pred:<8} fault_p={fault_p:.2f}")
        assert abs(sum(p["probabilities"].values()) - 1) < 1e-3
    assert "predictions_total" in c.get("/metrics").text
print("smoke test passed")
