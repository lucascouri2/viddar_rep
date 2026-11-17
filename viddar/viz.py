"""Plots and sample viewers used by the experiments notebook (matplotlib, static)."""

from __future__ import annotations

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import get_content_mask, load_mask, load_rgb

# Reference categorical order (blue, orange, aqua, yellow, ...); colour follows the series, not its rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KEY_OVERLAY, CONTENT_OVERLAY = SERIES[2], SERIES[1]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def alpha_sweep(runs: dict[str, pd.DataFrame], alphas=np.round(np.arange(0.05, 0.55, 0.025), 3),
                paper_alpha: float = 0.25):
    """Accuracy as a function of the Eq. 4 threshold, one line per run (needs `ratio` and `label`).

    Returns (figure, table) so the same numbers are available as a table.
    """
    table = pd.DataFrame({"alpha": alphas})
    fig, ax = plt.subplots(figsize=(7.5, 4), facecolor=SURFACE)
    for i, (name, df) in enumerate(runs.items()):
        acc = [float(((df["ratio"] >= a).astype(int) == df["label"]).mean()) for a in alphas]
        table[name] = acc
        ax.plot(alphas, acc, color=SERIES[i % len(SERIES)], linewidth=2, marker="o", markersize=4, label=name)
    # Direct end labels, nudged apart so overlapping lines do not stack their names.
    ends = sorted(((table[name].iloc[-1], name) for name in runs), key=lambda t: t[0])
    span = max(table[list(runs)].to_numpy().max() - table[list(runs)].to_numpy().min(), 0.05) if runs else 1
    min_gap, placed = 0.06 * span, []
    for y, name in ends:
        y_label = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_label)
        ax.annotate(name, (alphas[-1], y), xytext=(alphas[-1] + 0.012, y_label), textcoords="data",
                    va="center", fontsize=9, color=INK_2)
    ax.axvline(paper_alpha, color=INK_2, linewidth=1, linestyle="--")
    ax.text(paper_alpha, 1.0, " α do artigo", color=INK_2, fontsize=9, va="top",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("α (limiar da Eq. 4)")
    ax.set_ylabel("acurácia")
    ax.set_title("Acurácia da detecção de obstrução em função de α", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncols=min(len(runs), 4))
    ax.set_xlim(alphas[0] - 0.01, alphas[-1] + 0.09)
    _style(ax)
    fig.tight_layout()
    return fig, table


def factor_rates(pred: pd.DataFrame):
    """Ground-truth vs predicted positive rate for A, S, I and the final decision M."""
    rows = []
    for factor, gt_col, pred_col in [("A", "A", "pred_A"), ("S", "S", "pred_S"), ("I", "I", "pred_I"),
                                     ("M (decisão final)", "label", "pred")]:
        if gt_col in pred.columns and pred_col in pred.columns:
            rows.append({"fator": factor, "rótulo (verdade)": pred[gt_col].mean(),
                         "previsto pelo VLM": pred[pred_col].fillna(0).mean()})
    table = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    x = np.arange(len(table))
    width = 0.36
    for j, col in enumerate(["rótulo (verdade)", "previsto pelo VLM"]):
        bars = ax.bar(x + (j - 0.5) * (width + 0.02), table[col], width, color=SERIES[j], label=col)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in table[col]], fontsize=8, color=INK_2, padding=2)
    ax.set_xticks(x, table["fator"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("proporção de \"yes\"")
    ax.set_title("Taxa de respostas positivas: rótulo × VLM", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncols=2)
    _style(ax)
    fig.tight_layout()
    return fig, table


# Manipulation variants in a fixed order, so each keeps its colour in every chart (path A of the report).
MANIPULATION_VARIANTS = ["paper", "neutral", "definition", "paper_decomposed", "definition_decomposed"]


def variant_color(variant: str) -> str:
    base = variant.removesuffix("+A_geo")
    i = MANIPULATION_VARIANTS.index(base) if base in MANIPULATION_VARIANTS else len(MANIPULATION_VARIANTS)
    return SERIES[i % len(SERIES)]


def operating_points(table: pd.DataFrame, title: str = ""):
    """FPR (x) × FNR (y) of each variant of one model; (0, 0) is the ideal corner.

    `table` needs columns variant, fpr, fnr; optional hybrid_fpr, hybrid_fnr draw a hollow marker for
    the same variant with the geometric A (path C2) and an arrow from the VLM-only point.
    The dashed diagonal FPR + FNR = 1 is a classifier that ignores the input (balanced accuracy 0.5).
    """
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor=SURFACE)
    # The chance line is named in the legend, not on the plot, where it collided with point labels.
    ax.plot([0, 1], [1, 0], color=INK_2, linewidth=1, linestyle="--", label="acaso (FPR + FNR = 1)")
    # Variants with the same operating point (e.g. paper_decomposed and definition_decomposed, whose
    # decisions are identical) share one direct label instead of overprinting each other.
    point_labels: dict[tuple[float, float], list[str]] = {}
    for _, r in table.iterrows():
        point_labels.setdefault((round(r["fpr"], 6), round(r["fnr"], 6)), []).append(r["variant"])
    # Labels near the right edge go to the left of the point; labels of nearby points are stacked
    # vertically (12 pt apart) instead of overprinting.
    placed: list[tuple[float, float, int]] = []
    for (x, y), names in sorted(point_labels.items(), key=lambda kv: -kv[0][1]):
        left = x > 0.6
        shift = sum(1 for px, py, _ in placed if abs(py - y) < 0.06 and abs(px - x) < 0.3)
        placed.append((x, y, shift))
        ax.annotate(" = ".join(names), (x, y), xytext=(-7 if left else 7, 5 - 12 * shift),
                    textcoords="offset points", fontsize=8, color=INK_2, ha="right" if left else "left")
    for _, r in table.iterrows():
        color = variant_color(r["variant"])
        ax.scatter(r["fpr"], r["fnr"], s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=r["variant"])
        if "hybrid_fpr" in r and not pd.isna(r.get("hybrid_fpr")):
            ax.annotate("", (r["hybrid_fpr"], r["hybrid_fnr"]), (r["fpr"], r["fnr"]),
                        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.2})
            ax.scatter(r["hybrid_fpr"], r["hybrid_fnr"], s=70, facecolor=SURFACE, edgecolor=color, linewidth=2,
                       zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.set_xlabel("FPR (benignos marcados como ataque)")
    ax.set_ylabel("FNR (ataques não detectados)")
    ax.set_title(title or "Pontos de operação por variante", loc="left", fontsize=11)
    handles, labels = ax.get_legend_handles_labels()
    if "hybrid_fpr" in table.columns and table["hybrid_fpr"].notna().any():
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
                                  markeredgecolor=INK_2, markeredgewidth=2, markersize=8))
        labels.append("vazado: com A geométrico (C2)")
    ax.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left",
              bbox_to_anchor=(1.02, 1.0))
    _style(ax)
    fig.tight_layout()
    cols = [c for c in ["variant", "fpr", "fnr", "hybrid_fpr", "hybrid_fnr"] if c in table.columns]
    return fig, table[cols]


def gap_strip(features: pd.DataFrame, feature: str = "gap_norm", title: str = ""):
    """One dot per pair: the geometric feature split by the ground-truth A (pairs without detection omitted)."""
    df = features.dropna(subset=[feature])
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.5, 3.0), facecolor=SURFACE)
    rows = []
    for j, (a, name) in enumerate([(1, "A = 1 (alinhado)"), (0, "A = 0 (desalinhado)")]):
        v = df.loc[df["A"] == a, feature].to_numpy()
        ax.scatter(v, j + rng.uniform(-0.15, 0.15, len(v)), s=40, color=SERIES[j], edgecolor=SURFACE,
                   linewidth=1.5, label=name, zorder=3)
        rows.append({"rótulo": name, "n": len(v), "mediana": np.median(v) if len(v) else np.nan,
                     "p25": np.percentile(v, 25) if len(v) else np.nan,
                     "p75": np.percentile(v, 75) if len(v) else np.nan})
    ax.set_yticks([0, 1], ["A = 1", "A = 0"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel(feature)
    ax.set_title(title or f"{feature} por rótulo de alinhamento", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _style(ax)
    fig.tight_layout()
    return fig, pd.DataFrame(rows)


def show_alignment(row: pd.Series, key_mask_path: str | None, pad: float = 0.6, width: float = 9):
    """I_a with the virtual content m_c and the detected key object m_k, cropped around both."""
    from .data import load_mask

    raw = np.asarray(load_rgb(row["raw_path"]))
    ar = np.asarray(load_rgb(row["ar_path"]))
    content = get_content_mask(row)
    key = load_mask(key_mask_path) if key_mask_path else np.zeros_like(content)
    union = content | key
    ys, xs = np.nonzero(union) if union.any() else (np.array([0, ar.shape[0] - 1]), np.array([0, ar.shape[1] - 1]))
    h, w = ys.max() - ys.min(), xs.max() - xs.min()
    y0, y1 = max(0, int(ys.min() - pad * h)), min(ar.shape[0], int(ys.max() + pad * h))
    x0, x1 = max(0, int(xs.min() - pad * w)), min(ar.shape[1], int(xs.max() + pad * w))
    shown = _overlay(_overlay(raw, key, KEY_OVERLAY), content, CONTENT_OVERLAY)
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.6))
    for ax, img, t in [(axes[0], ar, "I_a (vista do usuário)"),
                       (axes[1], shown, "I_r + objeto detectado (verde) + conteúdo virtual (laranja)")]:
        ax.imshow(img[y0:y1, x0:x1])
        ax.set_title(t, fontsize=9, color=INK_2)
        ax.axis("off")
    info = [f"id={row['id']}", f"A={int(row['A'])}", f"objeto={row.get('key_phrase', '')!r}"]
    if not pd.isna(row.get("gap_norm")):
        info.append(f"gap_norm={row['gap_norm']:.2f}")
    fig.suptitle(" · ".join(info), fontsize=9, color=INK)
    fig.tight_layout()
    return fig


def _overlay(img: np.ndarray, mask: np.ndarray, color: str, alpha: float = 0.45) -> np.ndarray:
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    out = img.astype(float).copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * rgb
    return out.astype(np.uint8)


def show_obstruction(row: pd.Series, width: float = 12):
    """Raw image with the ground-truth key-object mask, AR image, and content mask over the raw image."""
    raw = np.asarray(load_rgb(row["raw_path"]))
    ar = np.asarray(load_rgb(row["ar_path"]))
    content = get_content_mask(row)
    panels = []
    key_path = row.get("key_mask_path")
    if isinstance(key_path, str) and key_path:
        panels.append((_overlay(raw, load_mask(key_path), KEY_OVERLAY), "I_r + objeto-chave (verdade)"))
    else:
        panels.append((raw, "I_r"))
    panels.append((ar, "I_a (vista do usuário)"))
    panels.append((_overlay(raw, content, CONTENT_OVERLAY), "I_r + conteúdo virtual m_c"))
    fig, axes = plt.subplots(1, len(panels), figsize=(width, width / len(panels) * 1.4))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=9, color=INK_2)
        ax.axis("off")
    pred = row.get("pred")
    info = [f"id={row['id']}", f"rótulo={'obstruído' if row['label'] == 1 else 'não obstruído'}",
            f"previsto={'?' if pd.isna(pred) else ('obstruído' if pred == 1 else 'não obstruído')}"]
    if "key_object" in row and isinstance(row["key_object"], str):
        info.append(f"objeto-chave={row['key_object']}")
    if "key_phrase" in row and isinstance(row["key_phrase"], str):
        info.append(f"VLM disse={row['key_phrase']!r}")
    if "ratio" in row and not pd.isna(row.get("ratio")):
        info.append(f"razão={row['ratio']:.2f}")
    if "key_iou" in row and not pd.isna(row.get("key_iou")):
        info.append(f"IoU objeto={row['key_iou']:.2f}")
    fig.suptitle(" · ".join(info), fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def show_manipulation(row: pd.Series, width: float = 9, print_response: bool = True):
    """Raw and AR images side by side, labels vs predictions, and the VLM's full answer."""
    fig, axes = plt.subplots(1, 2, figsize=(width, width * 0.95))
    for ax, key, title in [(axes[0], "raw_path", "I_r (crua)"), (axes[1], "ar_path", "I_a (aumentada)")]:
        ax.imshow(load_rgb(row[key]))
        ax.set_title(title, fontsize=9, color=INK_2)
        ax.axis("off")

    def fmt(v):
        return "?" if pd.isna(v) else str(int(v))

    gt = " ".join(f"{f}={fmt(row.get(f))}" for f in ("A", "S", "I")) + f" M={fmt(row['label'])}"
    pr = " ".join(f"{f}={fmt(row.get('pred_' + f))}" for f in ("A", "S", "I")) + f" M={fmt(row.get('pred'))}"
    fig.suptitle(f"id={row['id']}   rótulo: {gt}   VLM: {pr}", fontsize=9, color=INK)
    fig.tight_layout()
    if print_response and isinstance(row.get("vlm_response"), str):
        print("\n".join(textwrap.fill(line, 110) for line in row["vlm_response"].splitlines()))
    return fig
