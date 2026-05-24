#!/usr/bin/env python3
"""
visualize_sweep.py — Chuang et al. (2024) LLM 意見力学 スイープ結果 可視化スクリプト

results/latest (または --sweep_dir 指定先) の sweep_summary.csv を読み，
確証バイアス × トポロジ の格子について最終 Diversity D / Bias B / 分極を集計し，
ヒートマップと棒グラフで可視化する (論文 Table 1 風)．

Usage:
    uv run chuang-tools visualize-sweep
    uv run chuang-tools visualize-sweep --sweep_dir results/20260524_160000_sweep

Outputs:
    output_dir/
    ├── sweep_diversity_heatmap.png ← Diversity D (bias × topology) ヒートマップ
    ├── sweep_bias_heatmap.png      ← Bias B (bias × topology) ヒートマップ
    └── sweep_bias_vs_confirmation.png ← 確証バイアス別の D / B 棒グラフ
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "Hiragino Sans"

COLOR_BG = "#FAFAF8"
BIAS_ORDER = ["none", "weak", "strong"]


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """sweep_summary.csv を読み込む．"""
    path = os.path.join(sweep_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"sweep_summary.csv が見つかりません: {path}")
    return pd.read_csv(path)


def _ordered_biases(df: pd.DataFrame) -> list[str]:
    present = list(dict.fromkeys(df["bias"].tolist()))
    ordered = [b for b in BIAS_ORDER if b in present]
    ordered += [b for b in present if b not in ordered]
    return ordered


def pivot_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """(bias, topology) ごとに metric の試行平均をピボットする．"""
    agg = df.groupby(["bias", "topology"])[metric].mean().reset_index()
    table = agg.pivot(index="bias", columns="topology", values=metric)
    biases = [b for b in _ordered_biases(df) if b in table.index]
    return table.loc[biases]


def save_heatmap(table: pd.DataFrame, title: str, out_path: str, cmap: str) -> None:
    """bias × topology のヒートマップを保存する．"""
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * table.shape[1], 1.4 + 0.9 * table.shape[0]),
                           facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    data = table.to_numpy(dtype=float)
    im = ax.imshow(data, cmap=cmap, aspect="auto")

    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels(table.columns, rotation=0)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("トポロジ")
    ax.set_ylabel("確証バイアス")
    ax.set_title(title, fontsize=12)

    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def save_bias_vs_confirmation(df: pd.DataFrame, out_path: str) -> None:
    """確証バイアス別の平均 Diversity D / Bias B を棒グラフで比較する (論文 Table 1)．"""
    biases = _ordered_biases(df)
    d_means = [df[df["bias"] == b]["final_diversity"].mean() for b in biases]
    b_means = [df[df["bias"] == b]["final_bias"].mean() for b in biases]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor=COLOR_BG)

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(biases, d_means, color="#FF9800", alpha=0.85)
    for i, v in enumerate(d_means):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("確証バイアス")
    ax.set_ylabel("Diversity D (平均)")
    ax.set_title("確証バイアス↑ → 多様性 D↑ (論文 Table 1)")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(biases, b_means, color="#F44336", alpha=0.85)
    ax.axhline(0.0, color="#888888", lw=0.8, linestyle="--")
    for i, v in enumerate(b_means):
        ax.text(i, v, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
    ax.set_xlabel("確証バイアス")
    ax.set_ylabel("Bias B (平均)")
    ax.set_ylim(-2.1, 2.1)
    ax.set_title("確証バイアス別の Bias B")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chuang-tools visualize-sweep",
        description="Chuang et al. (2024) LLM 意見力学 スイープ結果 可視化スクリプト",
    )
    p.add_argument(
        "--sweep_dir",
        "--sweep-dir",
        default="results/latest",
        help="スイープ出力ディレクトリ (default: results/latest)",
    )
    p.add_argument(
        "--output_dir",
        "--output-dir",
        default=None,
        help="図の保存先ディレクトリ (default: {sweep_dir}/figures)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    out_dir = args.output_dir if args.output_dir else os.path.join(args.sweep_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)

    print("=== Chuang et al. (2024) LLM 意見力学 スイープ可視化 ===")
    print(f"スイープ: {args.sweep_dir}")
    print(f"出力先:   {out_dir}")
    print("-------------------------------------------------")

    print("[1/4] sweep_summary.csv を読み込み中 ...")
    df = load_summary(args.sweep_dir)
    print(f"      bias {df['bias'].nunique()} 種 × topology {df['topology'].nunique()} 種")

    print("[2/4] Diversity D ヒートマップを保存中 ...")
    save_heatmap(
        pivot_metric(df, "final_diversity"),
        "最終 Diversity D (確証バイアス × トポロジ)",
        os.path.join(out_dir, "sweep_diversity_heatmap.png"),
        cmap="YlOrRd",
    )

    print("[3/4] Bias B ヒートマップを保存中 ...")
    save_heatmap(
        pivot_metric(df, "final_bias"),
        "最終 Bias B (確証バイアス × トポロジ)",
        os.path.join(out_dir, "sweep_bias_heatmap.png"),
        cmap="coolwarm",
    )

    print("[4/4] 確証バイアス別 D/B 棒グラフを保存中 ...")
    save_bias_vs_confirmation(df, os.path.join(out_dir, "sweep_bias_vs_confirmation.png"))

    print("-------------------------------------------------")
    print("確証バイアス別の平均 Diversity D (単調増大が論文 Table 1 の知見):")
    for b in _ordered_biases(df):
        d = df[df["bias"] == b]["final_diversity"].mean()
        print(f"  {b:<7} → D̄ = {d:.3f}")

    print("-------------------------------------------------")
    print("完了．出力ファイル一覧:")
    for f in sorted(os.listdir(out_dir)):
        size_kb = os.path.getsize(os.path.join(out_dir, f)) / 1024
        print(f"  {f:35s} ({size_kb:6.1f} KB)")


if __name__ == "__main__":
    main()
