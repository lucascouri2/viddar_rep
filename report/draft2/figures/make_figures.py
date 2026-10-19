"""Generate the figures of the progress report from the saved runs in results/.

Run from the repository root (needs results/ from the ViDDAR runs and results/vwise/ from
scripts/analyze_vwise.py and scripts/pilot_vwise_keyobject.py):
    .venv/Scripts/python report/draft2/figures/make_figures.py
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


VWISE = ROOT / "data/placas/vwise_final"


def _vwise_frame(segment: str, k: int = 0) -> Image.Image:
    seg_dir = next(VWISE.glob(f"*/{segment}"))
    return Image.open(sorted(seg_dir.glob("*.png"))[k]).convert("RGB")


def _grid(images: list[Image.Image], cols: int, tile: tuple[int, int], gap: int = 6) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    W, H = tile
    out = Image.new("RGB", (cols * W + (cols - 1) * gap, rows * H + (rows - 1) * gap), "white")
    for i, im in enumerate(images):
        im = im.copy()
        im.thumbnail((W, H), Image.LANCZOS)
        x = (i % cols) * (W + gap) + (W - im.width) // 2
        y = (i // cols) * (H + gap) + (H - im.height) // 2
        out.paste(im, (x, y))
    return out


def vwise_classes() -> None:
    """One frame per scene class, read left-to-right, top-to-bottom in the order of the caption."""
    picks = ["17_segmento_435", "2_segmento_592", "7_segmento_770", "22_segmento_1431", "16_segmento_20", "33_segmento_6"]
    _grid([_vwise_frame(s, 1) for s in picks], 3, (480, 270)).save(OUT / "vw_classes.jpg", quality=86)


def vwise_overlays() -> None:
    """Frames that already contain 2D overlays baked into the video."""
    picks = [("6_segmento_128", 0), ("12_segmento_25", 0), ("19_segmento_100", 1), ("41_segmento_1", 1),
             ("45segmento1402", 1), ("44segmento11", 1)]
    _grid([_vwise_frame(s, k) for s, k in picks], 3, (480, 270)).save(OUT / "vw_overlays.jpg", quality=86)


def vwise_locality() -> None:
    """Context crops with visible evidence that a scene is outside Latin America."""
    review = pd.read_csv(ROOT / "data/annotations/vwise_sign_review.csv").set_index("idx")
    crops = []
    for i in [66, 24, 7, 118, 99]:  # Sea-Tac (Seattle), Port of Long Beach, MPH sign, Japanese sign, European warning sign
        r = review.loc[i]
        im = Image.open(VWISE / r["frame"]).convert("RGB")
        cx, cy = (r["x0"] + r["x1"]) / 2, (r["y0"] + r["y1"]) / 2
        half = max(r["w"], r["h"]) * 1.6 + 30
        box = (max(0, cx - 1.6 * half), max(0, cy - half), min(im.width, cx + 1.6 * half), min(im.height, cy + half))
        crops.append(im.crop(box))
    wagons = Image.open(sorted((VWISE / "recortes_28").glob("*/*.png"))[0]).convert("RGB")
    crops.append(wagons.crop((0, 300, 760, 780)))  # hopper wagons labelled "ЗЕРНО" (grain) in Cyrillic
    _grid(crops, 3, (420, 260)).save(OUT / "vw_locality.jpg", quality=88)


def vwise_sign_crops() -> None:
    """Rows: real traffic signs, possibly degraded signs, false positives of the sign detector."""
    review = pd.read_csv(ROOT / "data/annotations/vwise_sign_review.csv").set_index("idx")
    rows = [[2, 11, 58, 84, 98, 113], [29, 64, 86, 157], [32, 62, 10, 134, 20, 94]]
    for name, ids in zip(["vw_signs_real.jpg", "vw_signs_degraded.jpg", "vw_signs_fp.jpg"], rows):
        crops = []
        for i in ids:
            r = review.loc[i]
            im = Image.open(VWISE / r["frame"]).convert("RGB")
            pad = max(r["w"], r["h"]) * 0.3 + 6
            crops.append(im.crop((max(0, r["x0"] - pad), max(0, r["y0"] - pad),
                                  min(im.width, r["x1"] + pad), min(im.height, r["y1"] + pad))))
        _grid(crops, len(ids), (220, 220)).save(OUT / name, quality=88)


def authors_diagrams() -> None:
    for name in ["diagram_obstruction.png", "diagram_information_manipulation.png"]:
        shutil.copy(ROOT / "image" / name, OUT / name)


if __name__ == "__main__":
    alpha_sweep()
    factor_rates()
    obstruction_example()
    manipulation_example()
    authors_diagrams()
    vwise_classes()
    vwise_overlays()
    vwise_locality()
    vwise_sign_crops()
    print("figures written to", OUT)
