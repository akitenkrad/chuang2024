"""chuang-tools — Chuang et al. (2024) LLM 意見力学 ツール統合 CLI．

Usage:
    chuang-tools visualize [...]
    chuang-tools visualize-sweep [...]
    chuang-tools show-experiment-settings [...]
    chuang-tools reproduce [...]

各サブコマンドに続く引数は，対応するモジュールの argparse がそのまま受け取る．
サブコマンドレベルで `--help` を付けると，そのサブコマンド自身のヘルプが表示される．

dispatcher の組み立ては共有ヘルパ `socsim_tools.cli.build_dispatcher` に委譲する
(prog 名・サブコマンド・ヘルプ文・argv ルーティングは従来と同一)．可視化/設定表示の
実体 (visualize / visualize_sweep / show_experiment_settings) は repo 固有のまま．
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="chuang-tools",
    description="Chuang et al. (2024) LLM 意見力学 可視化・分析ツール",
    subcommands={
        "visualize": (
            "単一実行結果 (意見軌跡・B/D/分散時系列) の可視化",
            "chuang_tools.visualize:main",
        ),
        "visualize-sweep": (
            "スイープ結果 (確証バイアス×トポロジの B/D) の可視化",
            "chuang_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "実行結果ディレクトリの設定 (config / sweep_config / run_metadata) の表示",
            "chuang_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "論文 4.3 見出し的知見 (合意→断片化 / 非相互作用統制 / topology 比較) の一括再現と図",
            "chuang_tools.reproduce_paper:main",
        ),
    },
)


if __name__ == "__main__":
    main()
