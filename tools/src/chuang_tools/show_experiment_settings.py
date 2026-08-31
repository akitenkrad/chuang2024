"""chuang-tools show-experiment-settings — 実行結果の設定表示．

runvault の run ディレクトリの config.json (封筒．条件は `parameters` の下) を読み，
実行時に使われた全パラメータを整形表示する．run / sweep / sweep-point / reproduce の
どれかは run.json の `subcommand` が答える．LLM の同一性 (モデル・endpoint・温度) は
run.json の `llm` ブロックが，呼び出し数と cache-hit 率は metrics.csv の run スコープ
指標が持つ．legacy の flat な config.json / sweep_config.json / run_metadata.json も
そのまま読める．

run ディレクトリのパスは次で取れる:
    runvault path --experiment chuang --latest --subcommand run --standalone
    runvault path --experiment chuang --latest --subcommand sweep

Usage:
    chuang-tools show-experiment-settings
    chuang-tools show-experiment-settings --results-dir "$(runvault path --experiment chuang --latest --subcommand run --standalone)"
    chuang-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from runvault.read import (
    config_parameters,
    load_run_meta,
    run_scope_metrics,
    runvault_path,
)
from socsim_tools.io import load_run_metadata
from socsim_tools.settings import render_run_config, render_run_metadata

# config キー → 表示ラベル (右コロン位置を揃えるため空白パディング済み)．
# render_run_config が `f"{label}: {value}"` で整形するため，ラベルは末尾の
# `: ` を含めず，従来の run レンダラと同じ桁揃えになるようパディングする．
# `output_dir` は落とした — run ディレクトリが出力先そのものなので条件ではない．
FIELD_LABELS = {
    "n_agents": "エージェント数 N ",
    "topic": "トピック         ",
    "framing": "フレーミング     ",
    "bias": "確証バイアス     ",
    "memory_mode": "メモリ方式       ",
    "interact": "相互作用         ",
    "topology": "トポロジ         ",
    "events_per_step": "events/step      ",
    "max_steps": "最大ステップ T   ",
    "tol": "収束 tol         ",
    "seed": "シード (コア)    ",
    "llm_temperature": "LLM 温度         ",
    "llm_seed": "LLM seed         ",
    "mock": "mock 駆動        ",
}


def resolve_results_dir(path_like: str) -> Path:
    """シンボリックリンク (legacy の `results/latest`) を実体に解決する．"""
    p = Path(path_like)
    if p.is_symlink():
        return Path(os.path.realpath(p))
    return p


def _load_config(results_dir: Path) -> tuple[dict, Path, str]:
    """run ディレクトリの実験条件と，それがどのサブコマンドのものかを返す．

    runvault の config.json は封筒で，条件は `parameters` の下にある．どのサブ
    コマンドかは run.json が答える (`sweep_config.json` はもう書かれない)．
    """
    # 設定が無いことは «まだ sweep_config.json の方かもしれない» という意味なので，
    # ここでは欠落を失敗として扱わない (下で sweep_config.json を見る)．
    params = config_parameters(results_dir, required=False)
    if params is not None:
        meta = load_run_meta(results_dir, required=False)
        if meta is not None:
            kind = str(meta.get("subcommand", "run"))
        else:
            # legacy: 自前で書いていた config.json は "command" を持つ
            kind = "sweep" if params.get("command") == "sweep" else "run"
        return params, results_dir / "config.json", kind

    sweep_cfg = results_dir / "sweep_config.json"
    if sweep_cfg.exists():
        with sweep_cfg.open() as f:
            return json.load(f), sweep_cfg, "sweep"

    raise FileNotFoundError(
        f"設定ファイルが見つかりません: {results_dir}\n"
        f"  期待されるファイル: config.json (runvault の封筒 / legacy の flat) "
        f"または sweep_config.json (legacy の sweep)"
    )


def render_sweep_config(cfg: dict, source: Path) -> str:
    """sweep 親 run の設定テーブル (グリッド定義そのもの)．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"確証バイアス     : {', '.join(cfg.get('bias_values', []))}")
    lines.append(f"フレーミング     : {', '.join(cfg.get('framing_values', []))}")
    lines.append(f"トポロジ         : {', '.join(cfg.get('topology_values', []))}")
    lines.append(f"メモリ方式       : {cfg.get('memory', '-')}")
    lines.append(f"トピック         : {cfg.get('topic', '-')}")
    lines.append(f"エージェント数 N : {cfg.get('n_agents', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"events/step      : {cfg.get('events_per_step', '-')}")
    lines.append(f"最大ステップ T   : {cfg.get('max_steps', '-')}")
    lines.append(f"収束 tol         : {cfg.get('tol', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_sweep_point_config(cfg: dict, source: Path) -> str:
    """スイープの子 run (条件 1 点 × runs 試行) の条件．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (sweep-point — スイープの条件 1 点)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"確証バイアス     : {cfg.get('bias', '-')}")
    lines.append(f"フレーミング     : {cfg.get('framing', '-')}")
    lines.append(f"トポロジ         : {cfg.get('topology', '-')}")
    lines.append(f"メモリ方式       : {cfg.get('memory', '-')}")
    lines.append(f"トピック         : {cfg.get('topic', '-')}")
    lines.append(f"エージェント数 N : {cfg.get('n_agents', '-')}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"events/step      : {cfg.get('events_per_step', '-')}")
    lines.append(f"最大ステップ T   : {cfg.get('max_steps', '-')}")
    lines.append(f"収束 tol         : {cfg.get('tol', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_reproduce_config(cfg: dict, source: Path) -> str:
    """reproduce run の条件．"""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("実行設定 (reproduce)")
    lines.append("=" * 70)
    lines.append(f"設定ファイル: {source}")
    lines.append("-" * 70)
    lines.append(f"エージェント数 N : {cfg.get('n_agents', '-')}")
    lines.append(f"トピック         : {cfg.get('topic', '-')}")
    lines.append(f"フレーミング     : {cfg.get('framing', '-')}")
    lines.append(f"トポロジ         : {', '.join(cfg.get('topology_values', []))}")
    lines.append(f"試行数 runs      : {cfg.get('runs', '-')}")
    lines.append(f"events/step      : {cfg.get('events_per_step', '-')}")
    lines.append(f"最大ステップ T   : {cfg.get('max_steps', '-')}")
    lines.append(f"収束 tol         : {cfg.get('tol', '-')}")
    lines.append(f"シード基点       : {cfg.get('seed', '-')}")
    lines.append(f"LLM 温度         : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed         : {cfg.get('llm_seed', '-')}")
    lines.append(f"mock 駆動        : {cfg.get('mock', '-')}")
    lines.append(f"quick            : {cfg.get('quick', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def llm_summary(results_dir: Path) -> dict | None:
    """LLM の同一系情報を run.json と metrics.csv から組む．

    移行前の `run_metadata.json` に相当する．モデル・endpoint・温度は run.json の
    `llm` ブロックが (endpoint は provider として保持される)，呼び出し数と cache-hit は
    metrics.csv の run スコープ指標が正本．legacy の run ディレクトリは
    `run_metadata.json` をそのまま読む．
    """
    meta = load_run_meta(results_dir, required=False)
    if meta is None:
        return load_run_metadata(results_dir)
    llm = meta.get("llm")
    if llm is None:
        return None
    scoped = run_scope_metrics(results_dir)
    params = config_parameters(results_dir, required=False) or {}
    summary = {
        "llm_model": llm.get("model_snapshot", "-"),
        "llm_endpoint": llm.get("provider", "-"),
        "llm_temperature": llm.get("temperature", "-"),
        "llm_seed": params.get("llm_seed", "-"),
    }
    if "llm_calls" in scoped:
        summary["total_calls"] = int(scoped["llm_calls"])
    if "llm_cache_hits" in scoped:
        summary["cache_hits"] = int(scoped["llm_cache_hits"])
    if "llm_cache_hit_rate" in scoped:
        summary["cache_hit_rate"] = scoped["llm_cache_hit_rate"]
    return summary


def render_llm_summary(summary: dict) -> str:
    """LLM ブロックを整形する (run.json の llm + run スコープ指標)．"""
    lines: list[str] = []
    lines.append("")
    lines.append("LLM 実行メタデータ (run.json の llm ブロック + run スコープ指標)")
    lines.append("-" * 70)
    lines.append(f"モデル           : {summary.get('llm_model', '-')}")
    lines.append(f"provider         : {summary.get('llm_endpoint', '-')}")
    lines.append(f"温度             : {summary.get('llm_temperature', '-')}")
    lines.append(f"seed             : {summary.get('llm_seed', '-')}")
    lines.append(f"呼び出し総数     : {summary.get('total_calls', '-')}")
    lines.append(f"cache-hit        : {summary.get('cache_hits', '-')}")
    rate = summary.get("cache_hit_rate")
    if rate is not None:
        lines.append(f"cache-hit 率     : {rate * 100:.1f}%")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chuang-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default=None,
        help=(
            "run ディレクトリ．未指定時は runvault に最新の run を聞く "
            "(--experiment chuang --subcommand run --standalone)．"
        ),
    )
    parser.add_argument(
        "--results-root",
        "--results_root",
        default="results",
        help="--results-dir 未指定時に runvault が探す results ルート (default: results)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="表ではなく JSON 形式で出力する．",
    )
    args = parser.parse_args(argv)

    if args.results_dir is None:
        results_dir = Path(
            runvault_path("chuang", args.results_root, subcommand="run", standalone=True)
        )
    else:
        results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"エラー: ディレクトリが存在しません: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg, cfg_path, kind = _load_config(results_dir)
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    summary = llm_summary(results_dir)
    is_legacy = load_run_meta(results_dir, required=False) is None

    if args.json:
        payload = {
            "source": str(cfg_path),
            "kind": kind,
            "config": cfg,
            "run_metadata": summary,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path, FIELD_LABELS))
        elif kind == "sweep-point":
            print(render_sweep_point_config(cfg, cfg_path))
        elif kind == "reproduce":
            print(render_reproduce_config(cfg, cfg_path))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if summary is not None:
            print(render_run_metadata(summary) if is_legacy else render_llm_summary(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
