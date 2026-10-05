"""Generate the figures of the progress report from the saved runs in results/.

Run from the repository root:
    .venv/Scripts/python report/draft1/figures/make_figures.py
"""

from __future__ import annotations

import shutil
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

from viddar import viz  # noqa: E402
from viddar.data import get_content_mask, load_mask, load_rgb  # noqa: E402
from viddar.experiments import load_run  # noqa: E402

OUT = Path(__file__).resolve().parent
OBS_MAN = ROOT / "data/manifests/viddar_obstruction.csv"
MAN_MAN = ROOT / "data/manifests/viddar_manipulation.csv"
MODELS = {"qwen3-vl-2b": "Qwen3-VL-2B", "qwen3-vl-4b": "Qwen3-VL-4B", "llava-next-8b-4bit": "LLaVA-Next-8B (4 bits)"}

plt.rcParams.update({"font.size": 9, "font.family": "serif", "pdf.fonttype": 42})


def alpha_sweep() -> None:
    runs = {label: load_run(ROOT / f"results/obstruction/{m}/viddar").predictions for m, label in MODELS.items()}
    runs["Prior knowledge"] = load_run(ROOT / "results/obstruction/no_vlm/prior").predictions
    fig, table = viz.alpha_sweep(runs)
    ax = fig.axes[0]
    ax.set_title("", loc="left")  # the LaTeX caption carries the title
    ax.legend(frameon=False, fontsize=8, labelcolor=viz.INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncols=2)
    fig.set_size_inches(6.4, 3.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_alpha.pdf")
    table.to_csv(OUT / "fig_alpha.csv", index=False)
    plt.close(fig)


def factor_rates() -> None:
    preds = {label: load_run(ROOT / f"results/manipulation/{m}/paper").predictions for m, label in MODELS.items()}
    first = next(iter(preds.values()))
    factors = [("A", "A", "pred_A"), ("S", "S", "pred_S"), ("I", "I", "pred_I"), ("M (decisão final)", "label", "pred")]
    series = {"Rótulo (verdade)": [first[g].mean() for _, g, _ in factors]}
    for label, p in preds.items():
        series[label] = [p[c].fillna(0).mean() for _, _, c in factors]
    table = pd.DataFrame(series, index=[f for f, _, _ in factors])
    table.to_csv(OUT / "fig_factors.csv")

    fig, ax = plt.subplots(figsize=(6.4, 3.0), facecolor=viz.SURFACE)
    x = np.arange(len(factors))
    width = 0.19
    for j, (name, values) in enumerate(series.items()):
        bars = ax.bar(x + (j - 1.5) * (width + 0.012), values, width, color=viz.SERIES[j], label=name)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in values], fontsize=6.5, color=viz.INK_2, padding=1.5)
    ax.set_xticks(x, table.index)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("proporção de respostas \"yes\"")
    ax.legend(frameon=False, fontsize=8, labelcolor=viz.INK_2, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncols=4)
    viz._style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_factors.pdf")
    plt.close(fig)


def _save_jpg(img: np.ndarray, path: Path, width: int = 520) -> None:
    im = Image.fromarray(img)
    im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.save(path, quality=88)


def obstruction_example(sample_id: str = "electrical-outlet_004") -> None:
    preds = load_run(ROOT / "results/obstruction/qwen3-vl-2b/viddar", OBS_MAN).predictions
    row = preds.set_index("id").loc[sample_id]
    raw = np.asarray(load_rgb(row["raw_path"]))
    _save_jpg(viz._overlay(raw, load_mask(row["key_mask_path"]), viz.KEY_OVERLAY), OUT / "ex_obs_key.jpg")
    _save_jpg(np.asarray(load_rgb(row["ar_path"])), OUT / "ex_obs_ar.jpg")
    _save_jpg(viz._overlay(raw, get_content_mask(row), viz.CONTENT_OVERLAY), OUT / "ex_obs_content.jpg")
    print(f"{sample_id}: VLM={row['key_phrase']!r} ratio={row['ratio']:.2f} key_iou={row['key_iou']:.2f}")


def manipulation_example(sample_id: str = "2") -> None:
    man = pd.read_csv(MAN_MAN, dtype={"id": str}).set_index("id")
    for col, name in [("raw_path", "ex_man_raw.jpg"), ("ar_path", "ex_man_ar.jpg")]:
        _save_jpg(np.asarray(load_rgb(str((MAN_MAN.parent / man.loc[sample_id, col]).resolve()))), OUT / name, 360)


def authors_diagrams() -> None:
    for name in ["diagram_obstruction.png", "diagram_information_manipulation.png"]:
        shutil.copy(ROOT / "image" / name, OUT / name)


if __name__ == "__main__":
    alpha_sweep()
    factor_rates()
    obstruction_example()
    manipulation_example()
    authors_diagrams()
    print("figures written to", OUT)
