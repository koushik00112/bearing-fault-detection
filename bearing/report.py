"""Turn results.json into summary.md plus figures."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bearing.evaluate import mean_std  # noqa: E402

SCENARIO_TITLES = {
    "A_leaky_window": "A: random windows (**leaky, deliberately**)",
    "B_recording": "B: recording-level split",
    "B2_bearing": "B2: bearing-level split",
    "C_loco": "C: leave one operating condition out",
    "D_artificial_to_real": "D: train artificial damage, test real damage",
}


def per_seed_scores(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[float]]:
    """(scenario, model) -> one macro-F1 per seed (folds averaged within a seed)."""
    by_seed: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in rows:
        by_seed[(r["scenario"], r["model"], r["seed"])].append(r["macro_f1"])
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (scenario, model, _seed), scores in sorted(by_seed.items()):
        out[(scenario, model)].append(float(np.mean(scores)))
    return out


def _fmt(ms: dict[str, float]) -> str:
    return f"{ms['mean']:.3f} ± {ms['std']:.3f}"


def _confusion_figure(rows: list[dict[str, Any]], classes: list[str], path: Path) -> None:
    cells: dict[tuple[str, str], np.ndarray] = {}
    for r in rows:
        key = (r["scenario"], r["model"])
        cells[key] = cells.get(key, 0) + np.array(r["confusion"])
    scenarios = [s for s in SCENARIO_TITLES if any(k[0] == s for k in cells)]
    model_names = sorted({k[1] for k in cells})
    fig, axes = plt.subplots(
        len(scenarios),
        len(model_names),
        squeeze=False,
        figsize=(3.2 * len(model_names), 3 * len(scenarios)),
    )
    for i, s in enumerate(scenarios):
        for j, m in enumerate(model_names):
            ax = axes[i][j]
            cm = cells.get((s, m))
            if cm is None:
                ax.axis("off")
                continue
            norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(norm, vmin=0, vmax=1, cmap="Blues")
            for a in range(len(classes)):
                for b in range(len(classes)):
                    ax.text(
                        b,
                        a,
                        f"{norm[a, b]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if norm[a, b] > 0.5 else "black",
                    )
            ax.set_xticks(range(len(classes)), classes, fontsize=7)
            ax.set_yticks(range(len(classes)), classes, fontsize=7)
            ax.set_title(f"{s} / {m}", fontsize=8)
            if j == 0:
                ax.set_ylabel("true")
            if i == len(scenarios) - 1:
                ax.set_xlabel("predicted")
    fig.suptitle("Row-normalised confusion matrices (summed over seeds and folds)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _noise_figure(curves: dict[str, dict[str, dict[str, float]]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, points in sorted(curves.items()):
        labels = list(points)
        means = [points[k]["mean"] for k in labels]
        stds = [points[k]["std"] for k in labels]
        ax.errorbar(range(len(labels)), means, yerr=stds, marker="o", capsize=3, label=model)
        ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("test-time noise (SNR)")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(result: dict[str, Any], out: Path) -> Path:
    rows, classes = result["rows"], result["classes"]
    info = result.get("run_info", {})
    dataset = info.get("dataset", {})
    scores = per_seed_scores(rows)
    scenarios = [s for s in SCENARIO_TITLES if any(k[0] == s for k in scores)]
    model_names = sorted({k[1] for k in scores})

    lines = ["# Results", ""]
    if dataset.get("source") == "synthetic":
        lines += [
            "> **SYNTHETIC DATA. These numbers test the pipeline and mean nothing about "
            "real bearings. Do not report them.**",
            "",
        ]
    if info.get("quick"):
        lines += ["> **QUICK MODE** (reduced epochs/trees): smoke test only, not a result.", ""]
    lines += [
        f"- Data: `{dataset.get('source')}`, {dataset.get('n_windows')} windows, "
        f"fs={dataset.get('fs')} Hz, window={dataset.get('window')}, "
        f"stride={dataset.get('stride')}, channels={dataset.get('channels')}",
        f"- Raw data manifest sha256: `{dataset.get('raw_manifest_sha256')}`",
        f"- Code version: `{dataset.get('code_version')}`; run {info.get('started')} → "
        f"{info.get('finished')}",
        f"- Seeds: {info.get('config', {}).get('seeds')}. Cells are macro-F1, mean ± sample std "
        "over seeds (scenario C: mean of its folds within each seed).",
        "",
        "## Macro-F1 by scenario and model",
        "",
        "| Scenario | " + " | ".join(model_names) + " |",
        "|---|" + "---|" * len(model_names),
    ]
    for s in scenarios:
        cells = [_fmt(mean_std(scores[(s, m)])) if (s, m) in scores else "–" for m in model_names]
        lines.append(f"| {SCENARIO_TITLES[s]} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Per-class recall (mean over seeds and folds)",
        "",
        "| Scenario | Model | " + " | ".join(classes) + " |",
        "|---|---|" + "---|" * len(classes),
    ]
    for s in scenarios:
        for m in model_names:
            rs = [r for r in rows if r["scenario"] == s and r["model"] == m]
            if not rs:
                continue
            vals = []
            for c in classes:
                v = [r["per_class_recall"][c] for r in rs if c in r["per_class_recall"]]
                vals.append(f"{np.mean(v):.3f}" if v else "–")
            lines.append(f"| {s} | {m} | " + " | ".join(vals) + " |")

    c_rows = [r for r in rows if r["scenario"] == "C_loco"]
    if c_rows:
        lines += [
            "",
            "## Scenario C by held-out condition (macro-F1, mean ± std over seeds)",
            "",
            "| Held-out condition | " + " | ".join(model_names) + " |",
            "|---|" + "---|" * len(model_names),
        ]
        for fold in sorted({r["fold"] for r in c_rows}):
            cells = []
            for m in model_names:
                v = [r["macro_f1"] for r in c_rows if r["fold"] == fold and r["model"] == m]
                cells.append(_fmt(mean_std(v)) if v else "–")
            lines.append(f"| {fold} | " + " | ".join(cells) + " |")

    noisy = [r for r in rows if "noise_curve" in r]
    if noisy:
        curves: dict[str, dict[str, dict[str, float]]] = {}
        for m in model_names:
            rs = [r["noise_curve"] for r in noisy if r["model"] == m]
            if rs:
                curves[m] = {k: mean_std([c[k] for c in rs]) for k in rs[0]}
        levels = list(next(iter(curves.values())))
        lines += [
            "",
            f"## Noise robustness ({noisy[0]['scenario']}, macro-F1)",
            "",
            "White Gaussian noise added to test windows only.",
            "",
            "| Model | " + " | ".join(levels) + " |",
            "|---|" + "---|" * len(levels),
        ]
        for m, pts in curves.items():
            lines.append(f"| {m} | " + " | ".join(_fmt(pts[k]) for k in levels) + " |")
        _noise_figure(curves, out / "noise_curve.png")
        lines += ["", "![noise curve](noise_curve.png)"]

    eff = result.get("efficiency", {})
    if eff:
        lines += [
            "",
            "## Latency and size (single window, CPU, this machine)",
            "",
            "| Model | p50 ms | p95 ms | Size on disk | Parameters |",
            "|---|---|---|---|---|",
        ]
        for m, e in sorted(eff.items()):
            lines.append(
                f"| {m} | {e['latency_ms']['p50']:.2f} | {e['latency_ms']['p95']:.2f} | "
                f"{e['size_bytes'] / 1024:.0f} KiB | {e.get('n_parameters', '–')} |"
            )

    if result.get("errors"):
        err = pd.DataFrame(result["errors"])
        err.to_csv(out / "error_analysis.csv", index=False)
        worst = (
            err.groupby(["scenario", "model", "bearing", "label"])["accuracy"]
            .mean()
            .reset_index()
            .sort_values("accuracy")
            .head(15)
        )
        lines += [
            "",
            "## Error analysis: 15 worst test bearings (first seed)",
            "",
            "Full table: `error_analysis.csv` (per bearing and condition).",
            "",
            "| Scenario | Model | Bearing | True class | Accuracy |",
            "|---|---|---|---|---|",
        ]
        for _, r in worst.iterrows():
            lines.append(
                f"| {r['scenario']} | {r['model']} | {r['bearing']} | {r['label']} | "
                f"{r['accuracy']:.2f} |"
            )

    _confusion_figure(rows, classes, out / "confusion.png")
    lines += ["", "## Confusion matrices", "", "![confusion matrices](confusion.png)", ""]
    path = out / "summary.md"
    path.write_text("\n".join(lines))
    return path
