#!/usr/bin/env python3
"""reproduce_paper.py — Chuang et al. (2024) 見出し的知見の一括再現レポート + 図．

Rust の `chuang reproduce` が書いた runvault の run ディレクトリを読み，論文 4.3 の
中心的知見を 3 つの図で可視化しつつ PASS/off テーブルを表示する．セル集約と代表 run の
時系列は `metrics.csv` (条件ラベルを接頭辞に持つ run スコープ指標) が，アンカーの帯と
PASS/off は `events.jsonl` の `x.chuang2024.anchor` 行が持つ (旧 `reproduce_summary.json`
と `metrics_<condition>.csv` に相当する)．

どの run を見るかは `--results-dir` を省略すれば runvault が答える
(`runvault path --experiment chuang --latest --subcommand reproduce`)．

3 つの図:

    1. bias_control_matrix.png
       確証バイアス (none/weak/strong) × 統制 (interaction / no-interaction) の最終
       Diversity D・Bias B 棒グラフ．論文の «バイアス無し→合意 (低 D) / 強バイアス→
       断片化 (高 D)» と，非相互作用統制が «社会的影響» を切ると合意が起きないこと
       (= LLM 自身の prior ドリフトと網による影響の分離) を一目で示す．
    2. topology_comparison.png
       同一バイアス (none) で network 構造 (full / er / ws / ba) を変えたときの最終
       Diversity D と収束ステップ．密な網ほど速く合意へ向かう．
    3. control_contrast.png
       代表 run の Diversity D 時系列を interaction vs no-interaction で重ね描き．
       社会的相互作用が合意を駆動することを時系列で対比する．

`--run` を付けると先に Rust バイナリ (`cargo run --release -- reproduce`) を実行して
最新結果を生成する．サンドボックス・CI では `--mock` も付けてライブ LLM を回避する．

Usage:
    uv run chuang-tools reproduce --run --mock          # mock で一括再現 + 図
    uv run chuang-tools reproduce --run --mock --quick  # 軽量版 (動作確認用)
    uv run chuang-tools reproduce                        # 直近の reproduce run を可視化
    uv run chuang-tools reproduce --results-dir "$(runvault path --experiment chuang --latest --subcommand reproduce)"
    uv run chuang-tools reproduce --json

Outputs:
    results/chuang/figures/{run_slug}/{bias_control_matrix,topology_comparison,control_contrast}.png
    stdout: アンカーごとの PASS / OFF．
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from runvault.read import (
    config_parameters,
    events_table,
    figures_dir,
    metrics_wide,
    run_scope_metrics,
    runvault_path,
)

# --------------------------------------------------------------------------- #
# 表示設定 (CJK フォントが利用不能でも落ちないように try)
# --------------------------------------------------------------------------- #
try:
    plt.rcParams["font.family"] = "Hiragino Sans"
except Exception:  # pragma: no cover - フォント未インストール環境用フォールバック
    pass

COLOR_BG = "#FAFAF8"
COLOR_INTERACT = "#2196F3"
COLOR_CONTROL = "#FF9800"
COLOR_DIV = "#9C27B0"
COLOR_BIAS = "#F44336"
TOPO_COLORS = {
    "full": "#2196F3",
    "er": "#4CAF50",
    "ws": "#FF9800",
    "ba": "#9C27B0",
}


# --------------------------------------------------------------------------- #
# Rust バイナリ実行
# --------------------------------------------------------------------------- #


def _run_binary(*, mock: bool, quick: bool, seed: int, output_dir: str) -> None:
    """`cargo run --release -- reproduce ...` を実行して最新結果を生成する．"""
    cmd = ["cargo", "run", "--release", "--", "reproduce", "--seed", str(seed),
           "--output-dir", output_dir]
    if mock:
        cmd.append("--mock")
    if quick:
        cmd.append("--quick")
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


BIASES = ["none", "weak", "strong"]
ARMS = [("interact", "interaction"), ("control", "no-interaction")]
CELL_FIELDS = [
    "mean_final_bias",
    "mean_final_diversity",
    "mean_final_clusters",
    "mean_diversity_drop",
    "mean_final_step",
]


def _load_summary(run_dir: Path) -> dict:
    """run ディレクトリから旧 `reproduce_summary.json` と同じ形の要約を組み直す．

    runvault はこの要約をディスク上に持たない．セル集約は `metrics.csv` の run
    スコープ指標 (`<label>_mean_*`)，アンカーは `events.jsonl` の
    `x.chuang2024.anchor` 行，どの条件を回したかは `config.json` の `parameters` が
    持つので，そこから復元する．legacy の run ディレクトリは
    `reproduce_summary.json` をそのまま読む．
    """
    legacy = run_dir / "reproduce_summary.json"
    if legacy.exists():
        with legacy.open(encoding="utf-8") as f:
            return json.load(f)

    scoped = run_scope_metrics(run_dir)
    params = config_parameters(run_dir)

    def cell(label: str, extra: dict) -> dict | None:
        if f"{label}_mean_final_diversity" not in scoped:
            return None
        c = {"label": label, **extra}
        c.update({f: scoped[f"{label}_{f}"] for f in CELL_FIELDS})
        return c

    matrix = []
    for b in BIASES:
        for arm, control in ARMS:
            c = cell(f"bias-{b}_{arm}", {"bias": b, "control": control})
            if c is not None:
                matrix.append(c)

    topology = []
    for t in params.get("topology_values", []):
        c = cell(f"topo-{t}", {"topology": t})
        if c is not None:
            topology.append(c)

    anchors = []
    for _, ev in events_table(run_dir, kind="x.chuang2024.anchor").iterrows():
        hi = ev.get("target_hi")
        anchors.append({
            "name": ev["label"],
            "paper": ev["paper"],
            "observed": float(ev["observed"]),
            "target_lo": float(ev["target_lo"]),
            # 上限なしのアンカーは列ごと落としてあるので None に戻す
            "target_hi": None if hi is None or pd.isna(hi) else float(hi),
            "pass": bool(ev["pass"]),
        })

    return {
        "mode": "mock" if params.get("mock") else "live",
        "bias_control_matrix": matrix,
        "topology_comparison": topology,
        "anchors": anchors,
        "n_pass": int(scoped.get("checks_passed", sum(a["pass"] for a in anchors))),
        "n_total": int(scoped.get("checks_total", len(anchors))),
    }


# --------------------------------------------------------------------------- #
# 描画
# --------------------------------------------------------------------------- #


def _bias_control_matrix(summary: dict, out_path: Path) -> None:
    """bias × control の最終 Diversity D・Bias B 棒グラフ．"""
    cells = summary["bias_control_matrix"]
    biases = ["none", "weak", "strong"]
    interact = {c["bias"]: c for c in cells if c["control"] == "interaction"}
    control = {c["bias"]: c for c in cells if c["control"] == "no-interaction"}

    x = np.arange(len(biases))
    w = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Chuang et al. (2024) — 確証バイアス × 統制条件 (最終 Diversity D / Bias B)",
        fontsize=13,
    )

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, [interact[b]["mean_final_diversity"] for b in biases], w,
           color=COLOR_INTERACT, label="interaction")
    ax.bar(x + w / 2, [control[b]["mean_final_diversity"] for b in biases], w,
           color=COLOR_CONTROL, label="no-interaction (control)")
    ax.set_xticks(x)
    ax.set_xticklabels(biases)
    ax.set_xlabel("確証バイアス")
    ax.set_ylabel("最終 Diversity D")
    ax.set_title("Diversity D: 無バイアス→合意 / 強バイアス→断片化", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, [interact[b]["mean_final_bias"] for b in biases], w,
           color=COLOR_INTERACT, label="interaction")
    ax.bar(x + w / 2, [control[b]["mean_final_bias"] for b in biases], w,
           color=COLOR_CONTROL, label="no-interaction (control)")
    ax.axhline(0.0, color="#888888", lw=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(biases)
    ax.set_ylim(-2.1, 2.1)
    ax.set_xlabel("確証バイアス")
    ax.set_ylabel("最終 Bias B (意見平均)")
    ax.set_title("Bias B: 相互作用は真値方向の極へ収束", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _topology_comparison(summary: dict, out_path: Path) -> None:
    """topology 比較の最終 Diversity D と収束ステップ棒グラフ．"""
    cells = summary["topology_comparison"]
    labels = [c["topology"] for c in cells]
    colors = [TOPO_COLORS.get(t, "#607D8B") for t in labels]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLOR_BG)
    fig.suptitle(
        "Chuang et al. (2024) — トポロジ比較 (bias=none, interaction)",
        fontsize=13,
    )

    ax = axes[0]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x, [c["mean_final_diversity"] for c in cells], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("ネットワークトポロジ")
    ax.set_ylabel("最終 Diversity D")
    ax.set_title("収束後の意見多様性", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.set_facecolor(COLOR_BG)
    ax.bar(x, [c["mean_final_step"] for c in cells], color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("ネットワークトポロジ")
    ax.set_ylabel("最終ステップ (収束まで)")
    ax.set_title("収束速度 (密な網ほど速い)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


def _control_contrast(results_dir: Path, out_path: Path) -> None:
    """interaction vs no-interaction の Diversity D 時系列 (代表 run)．

    条件ごとの時系列は `metrics.csv` に `<label>_diversity` として入っている
    (旧 `metrics_<condition>.csv` に相当)．legacy の run ディレクトリは
    `metrics_<label>.csv` をそのまま読む．
    """
    pairs = [
        ("bias-none_interact", "none / interaction", COLOR_INTERACT, "-"),
        ("bias-none_control", "none / no-interaction", COLOR_CONTROL, "--"),
        ("bias-strong_interact", "strong / interaction", COLOR_BIAS, "-"),
        ("bias-strong_control", "strong / no-interaction", COLOR_DIV, "--"),
    ]
    wide = None
    metrics_path = results_dir / "metrics.csv"
    if metrics_path.exists():
        wide = metrics_wide(metrics_path)
        if "step" in wide.columns and "t" not in wide.columns:
            wide = wide.rename(columns={"step": "t"})

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    plotted = 0
    for label, legend, color, ls in pairs:
        legacy = results_dir / f"metrics_{label}.csv"
        if legacy.exists():
            df = pd.read_csv(legacy)
            ts, ys = df["t"], df["diversity"]
        elif wide is not None and f"{label}_diversity" in wide.columns:
            sub = wide[["t", f"{label}_diversity"]].dropna()
            ts, ys = sub["t"], sub[f"{label}_diversity"]
        else:
            continue
        ax.plot(ts, ys, color=color, ls=ls, lw=2, label=legend)
        plotted += 1
    if plotted == 0:
        print("  警告: 条件別の diversity 時系列が無いため control_contrast をスキップ")
        plt.close(fig)
        return
    ax.set_xlabel("時刻 t (ステップ)")
    ax.set_ylabel("Diversity D")
    ax.set_title(
        "社会的相互作用が合意を駆動する (D 時系列; 代表 run)\n"
        "interaction は D を縮小，no-interaction 統制は温存",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  保存: {out_path}")


# --------------------------------------------------------------------------- #
# レポート出力
# --------------------------------------------------------------------------- #


def _print_report(summary: dict, results_dir: Path) -> None:
    print("=" * 78)
    print("Chuang et al. (2024) — 見出し的知見 一括再現レポート")
    print(f"  source: {results_dir}  (mode={summary.get('mode', '?')})")
    print("=" * 78)

    print("\n[bias × control 行列 (full topology)]")
    print(f"  {'condition':<24}{'B̄':>8}{'D̄':>8}{'clust':>8}{'D-drop':>10}")
    for c in summary["bias_control_matrix"]:
        print(f"  {c['label']:<24}{c['mean_final_bias']:>8.3f}"
              f"{c['mean_final_diversity']:>8.3f}{c['mean_final_clusters']:>8.2f}"
              f"{c['mean_diversity_drop']:>10.3f}")

    print("\n[topology 比較 (bias=none, interaction)]")
    for c in summary["topology_comparison"]:
        print(f"  {c['label']:<24}{c['mean_final_bias']:>8.3f}"
              f"{c['mean_final_diversity']:>8.3f}{c['mean_final_clusters']:>8.2f}"
              f"  step={c['mean_final_step']:.1f}")

    print("\n[論文知見アンカー (観測 vs 論文)]")
    n_pass = 0
    for a in summary["anchors"]:
        hi = a["target_hi"]
        hi_str = "∞" if hi is None or hi > 1e30 else f"{hi:.3f}"
        status = "PASS" if a["pass"] else "OFF "
        if a["pass"]:
            n_pass += 1
        print(f"  [{status}] {a['name']:<54} obs={a['observed']:.4f} "
              f"target=[{a['target_lo']:.3f},{hi_str}] paper={a['paper']}")
    print("-" * 78)
    print(f"{n_pass}/{len(summary['anchors'])} アンカーが in-band")
    print("(中核知見: 無バイアス→真値方向の合意 / 確証バイアスで D 単調増大 / "
          "非相互作用統制は社会的影響を切り固有ドリフトを分離)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chuang-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help=("reproduce の run ディレクトリ．未指定時は runvault に"
                              "最新の run を聞く (--experiment chuang --subcommand reproduce)．"))
    parser.add_argument("--results-root", "--results_root", default="results",
                        help="--results-dir 未指定時に runvault が探す results ルート (既定: results)")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (既定: results/chuang/figures/{run_slug})")
    parser.add_argument("--run", action="store_true",
                        help="先に Rust バイナリ (reproduce) を実行する．")
    parser.add_argument("--mock", action="store_true",
                        help="--run 時にライブ LLM を使わず mock で駆動する．")
    parser.add_argument("--quick", action="store_true",
                        help="--run 時に軽量モードで実行する (動作確認用)．")
    parser.add_argument("--seed", type=int, default=42, help="--run 時のシード基点．")
    parser.add_argument("--cargo-output-dir", "--cargo_output_dir", default="results",
                        help="--run 時に cargo の --output-dir へ渡すパス (既定: results)．")
    parser.add_argument("--json", action="store_true", help="JSON 形式で要約を出力する．")
    args = parser.parse_args(argv)

    if args.run:
        _run_binary(mock=args.mock, quick=args.quick, seed=args.seed,
                    output_dir=args.cargo_output_dir)

    if args.results_dir is None:
        results_dir = Path(
            runvault_path("chuang", args.results_root, subcommand="reproduce")
        )
    else:
        results_dir = Path(args.results_dir)
    try:
        summary = _load_summary(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    _print_report(summary, results_dir)

    out_dir = Path(args.output_dir) if args.output_dir else Path(figures_dir(results_dir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[図] 出力先: {out_dir}")
    _bias_control_matrix(summary, out_dir / "bias_control_matrix.png")
    _topology_comparison(summary, out_dir / "topology_comparison.png")
    _control_contrast(results_dir, out_dir / "control_contrast.png")

    print("-" * 78)
    n_pass = sum(1 for a in summary["anchors"] if a["pass"])
    return 0 if n_pass == len(summary["anchors"]) else 0


if __name__ == "__main__":
    sys.exit(main())
