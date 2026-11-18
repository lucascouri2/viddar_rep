"""Generate the figures of the final paper (obstruction and manipulation benchmarks) from results/.

Run from the repository root after the runs of notebooks/benchmark_obstrucao.ipynb and
notebooks/benchmark_manipulacao.ipynb (no model is loaded; only saved predictions are read):
    .venv/Scripts/python report/draft_final/figures/make_figures.py

Each chart is written as PDF next to a CSV with the same numbers. The images ex_align_*.jpg come from
report/draft_final_old/figures/make_figures.py (alignment_example) and are not regenerated here.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from viddar import reporting, viz  # noqa: E402
from viddar.data import get_content_mask, load_mask, load_rgb  # noqa: E402

OUT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OBS_MANIFEST = ROOT / "data/manifests/viddar_obstruction.csv"
MODELS = {"qwen3-vl-2b": "Qwen3-VL-2B", "qwen3-vl-4b": "Qwen3-VL-4B", "qwen3-vl-8b-4bit": "Qwen3-VL-8B",
          "llava-next-8b-4bit": "LLaVA-Next-8B", "gemma4-12b-4bit": "Gemma 4 12B"}
VARIANTS = ["paper", "neutral", "paper_decomposed"]
SHORT = {"paper": "P", "neutral": "N", "paper_decomposed": "PD"}
MARKERS = {"paper": "o", "neutral": "s", "paper_decomposed": "D"}

plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "pdf.fonttype": 42})


def _style(ax) -> None:
    viz._style(ax)
    ax.tick_params(labelsize=7)


def operating_points() -> None:
    """Small multiples, one panel per VLM: each prompt variant on the manipulation task in ROC space (FPR x recall)."""
    runs = reporting.load_manipulation_runs(RESULTS, list(MODELS), VARIANTS)
    errors = reporting.error_comparison(runs, "paper")
    errors[["model", "variant", "fpr", "recall", "accuracy"]].round(4).to_csv(OUT / "fig_operating_points.csv",
                                                                              index=False)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.16, 2.05), facecolor=viz.SURFACE, sharey=True)
    for ax, (model, name) in zip(axes, MODELS.items()):
        ax.plot([0, 1], [0, 1], color=viz.INK_2, linewidth=0.7, linestyle="--", zorder=1)
        ax.scatter([0], [0], s=60, marker="x", color=viz.INK_2, linewidth=1.0, zorder=2)
        t = errors[errors["model"] == model].set_index("variant")
        for variant in VARIANTS:
            if variant not in t.index:
                continue
            r = t.loc[variant]
            ax.scatter(r["fpr"], r["recall"], s=30, marker=MARKERS[variant], facecolor=viz.variant_color(variant),
                       edgecolor=viz.SURFACE, linewidth=0.6, zorder=3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1], ["0", "50", "100"])
        ax.set_yticks([0, 0.5, 1], ["0", "50", "100"])
        ax.set_xlabel("FPR (%)", fontsize=7)
        ax.set_title(name, fontsize=7.5, loc="left")
        _style(ax)
    axes[0].set_ylabel("recall (%)", fontsize=7)
    handles = [plt.Line2D([], [], linestyle="", marker=MARKERS[v], markersize=5, markerfacecolor=viz.variant_color(v),
                          markeredgecolor=viz.variant_color(v)) for v in VARIANTS]
    labels = [f"{SHORT[v]}: {v.replace('_', ' ')}" for v in VARIANTS]
    handles.append(plt.Line2D([], [], linestyle="", marker="x", color=viz.INK_2, markersize=5))
    labels.append("detector trivial (nunca dispara)")
    handles.append(plt.Line2D([], [], linestyle="--", color=viz.INK_2, linewidth=0.7))
    labels.append("acaso (recall = FPR)")
    fig.legend(handles, labels, frameon=False, fontsize=6.5, labelcolor=viz.INK_2, loc="lower center",
               ncols=5, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.9, bottom=0.27, wspace=0.12)
    fig.savefig(OUT / "fig_op.pdf")
    plt.close(fig)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _crop_box(mask: np.ndarray, shape, pad: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _bbox(mask)
    w, h = x1 - x0, y1 - y0
    return (max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h)),
            min(shape[1], int(x1 + pad * w)), min(shape[0], int(y1 + pad * h)))


def _outline(img: np.ndarray, mask: np.ndarray, color: str, thickness: int) -> np.ndarray:
    """Draw the bounding box of `mask` (the annotated key object) so it does not hide the object itself."""
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    out = img.copy()
    x0, y0, x1, y1 = _bbox(mask)
    t = thickness
    out[max(0, y0 - t):y0, x0 - t:x1 + t] = rgb
    out[y1:y1 + t, x0 - t:x1 + t] = rgb
    out[y0 - t:y1 + t, max(0, x0 - t):x0] = rgb
    out[y0 - t:y1 + t, x1:x1 + t] = rgb
    return out


def _resized(img: np.ndarray, box: tuple[int, int, int, int], height: int) -> Image.Image:
    im = Image.fromarray(img).crop(box)
    return im.resize((int(im.width * height / im.height), height), Image.LANCZOS)


def obstruction_fp_examples(ids=("warning-sign_002", "boiling-pot_010"), pad: float = 0.25,
                            height: int = 420) -> None:
    """Non-obstructed samples flagged by 4 of 5 VLMs: I_r with the annotated key object (red box), and I_a.

    Both images are cropped around the key object and the virtual content.
    """
    man = pd.read_csv(OBS_MANIFEST).set_index("id")
    for sample_id in ids:
        row = man.loc[sample_id].copy()
        for col in ["raw_path", "ar_path", "key_mask_path", "content_mask_path"]:
            row[col] = str((OBS_MANIFEST.parent / row[col]).resolve())
        row["id"] = sample_id
        raw, ar = np.asarray(load_rgb(row["raw_path"])), np.asarray(load_rgb(row["ar_path"]))
        key = load_mask(row["key_mask_path"])
        box = _crop_box(key | get_content_mask(row), raw.shape, pad)
        _resized(_outline(raw, key, viz.SERIES[7], 10), box, height).save(OUT / f"ex_fp_{sample_id}_raw.jpg",
                                                                         quality=88)
        _resized(ar, box, height).save(OUT / f"ex_fp_{sample_id}_ar.jpg", quality=88)


if __name__ == "__main__":
    operating_points()
    obstruction_fp_examples()
    print("figures written to", OUT)
