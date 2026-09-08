"""Phase 1c Part A: 探索 (2025-01-01 .. 2026-05-31)。

**この期間で出た数字に統計的な意味は無い。** 既に何度も見ている期間であり、
Phase 1 / 1b でも使っている。ここでの目的は「候補を選ぶこと」だけ。
エッジの有無を判定するのは Part B (2023-2024、未使用) でのみ行う。

実行: python3 phase1c_partA.py
出力: reports/phase1c_exploration.md, reports/phase1c_*.csv,
      reports/phase1c_registered_candidates.json  ← Part B の事前登録記録
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase1c_engine import build_context, evaluate_setting
from phase1c_grid import (
    HORIZONS, INTERVALS, MAX_CANDIDATES_FROM_GRID, NEIGHBOUR_TOP_FRACTION,
    PART_A_END, PART_A_START, PART_B_END, PART_B_START, SYMBOLS,
    UNCONDITIONAL_CANDIDATE, WARMUP_BARS,
    apply_selection_rules, best_horizon, build_grid, compute_scores, plateau_metric,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
FAMILY_LABEL = {"adx": "族 1: ADX", "wavetrend": "族 2: WaveTrend",
                "squeeze": "族 3: Squeeze"}


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    grid = build_grid()

    rows = []
    for interval in INTERVALS:
        contexts = [build_context(symbol, interval, PART_A_START, PART_A_END)
                    for symbol in SYMBOLS]
        print(f"  {interval}: 文脈構築完了 ({[c.bar_count for c in contexts]} 本)",
              flush=True)
        for setting in grid:
            rows.extend(evaluate_setting(setting, contexts))
        print(f"  {interval}: {len(grid)} 設定 評価完了", flush=True)

    results = pd.DataFrame(rows)
    scores, scored_results = compute_scores(results)
    qualified, breakdown = apply_selection_rules(scored_results, scores, grid)
    plateau = plateau_metric(scores, grid)

    scored_results.to_csv(REPORT_DIR / "phase1c_partA_grid.csv", index=False)
    breakdown.to_csv(REPORT_DIR / "phase1c_partA_selection.csv", index=False)
    plateau.to_csv(REPORT_DIR / "phase1c_partA_plateau.csv", index=False)

    candidates = _register_candidates(qualified, scored_results, breakdown, grid)
    (REPORT_DIR / "phase1c_registered_candidates.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_report(results, scored_results, scores, qualified, breakdown,
                  plateau, candidates)
    print("レポート:", REPORT_DIR / "phase1c_exploration.md")
    print("事前登録:", REPORT_DIR / "phase1c_registered_candidates.json")


def _register_candidates(qualified, scored_results, breakdown, grid) -> list[dict]:
    """Part B に持ち込む候補を確定させる。ここで固めたら Part B の結果で変えない。"""
    grid_by_key = {setting.key: setting for setting in grid}
    candidates = []

    for row in qualified.itertuples():
        setting = grid_by_key[row.setting_key]
        candidates.append({
            "family": setting.family,
            "params": setting.params,
            "interval": row.interval,
            "horizon": int(row.best_horizon),
            "origin": "Part A のグリッド探索 (選定ルール通過)",
            "part_a_score": float(row.score),
            "part_a_mean_atr": float(row.mean_atr),
            "part_a_mean_bps": float(row.mean_bps),
            "part_a_n_signals": int(row.n_signals),
        })

    # Phase 1b の副次系列を無条件で追加する。
    # ホライズンだけは他の候補と同じ機械的ルール (Part A で順位が最良のもの) で決める。
    setting = next(
        s for s in grid
        if s.family == UNCONDITIONAL_CANDIDATE["family"]
        and s.params == UNCONDITIONAL_CANDIDATE["params"]
    )
    interval = UNCONDITIONAL_CANDIDATE["interval"]
    horizon = best_horizon(scored_results, setting.key, interval)
    cell = scored_results[(scored_results["setting_key"] == setting.key)
                          & (scored_results["interval"] == interval)
                          & (scored_results["horizon"] == horizon)].iloc[0]
    already = any(c["family"] == setting.family and c["params"] == setting.params
                  and c["interval"] == interval for c in candidates)
    if not already:
        candidates.append({
            "family": setting.family,
            "params": setting.params,
            "interval": interval,
            "horizon": int(horizon),
            "origin": UNCONDITIONAL_CANDIDATE["origin"],
            "part_a_score": float(
                breakdown[(breakdown["setting_key"] == setting.key)
                          & (breakdown["interval"] == interval)]["score"].iloc[0]),
            "part_a_mean_atr": float(cell["mean_atr"]),
            "part_a_mean_bps": float(cell["mean_bps"]),
            "part_a_n_signals": int(cell["n_signals"]),
        })
    return candidates


def _grid_table(results: pd.DataFrame, breakdown: pd.DataFrame, family: str) -> str:
    """族のグリッド全体を表にする。良かった設定だけを載せない。"""
    subset = breakdown[breakdown["family"] == family]
    pivot_score = subset.pivot_table(index="setting_key", columns="interval",
                                     values="score")
    pivot_mean = subset.pivot_table(index="setting_key", columns="interval",
                                    values="mean_atr")
    pivot_horizon = subset.pivot_table(index="setting_key", columns="interval",
                                       values="best_horizon")
    frame = pd.DataFrame(index=pivot_score.index)
    label = results.drop_duplicates("setting_key").set_index("setting_key")["label"]
    frame["設定"] = label
    for interval in INTERVALS:
        frame[f"{interval} 順位"] = pivot_score[interval].round(1)
        frame[f"{interval} 平均ATR"] = [
            f"{m:+.3f} (N={int(h)})"
            for m, h in zip(pivot_mean[interval], pivot_horizon[interval])
        ]
    frame = frame.reset_index(drop=True)
    return frame.to_markdown(index=False)


def _write_report(results, scored_results, scores, qualified, breakdown,
                  plateau, candidates) -> None:
    lines: list[str] = []
    add = lines.append
    cell_count = len(results)

    add("# Phase 1c Part A: パラメータ探索（2025-01-01 .. 2026-05-31）")
    add("")
    add("## 0. この期間の数字に統計的な意味は無い")
    add("")
    add("**まず最初に断っておく。**")
    add("")
    add(f"- 探索したセル数: **{cell_count:,}**"
        f"（65 設定 × {len(INTERVALS)} 時間足 × {len(HORIZONS)} ホライズン）")
    add(f"- 全設定が本当に無価値でも、偶然だけで p < 0.05 のセルが "
        f"**約 {cell_count * 0.05:.0f} 個**出る規模である")
    add(f"- この期間（{PART_A_START} .. {PART_A_END}）は Phase 1 / 1b で既に何度も見ている")
    add("")
    add("**したがって Part A では p 値も信頼区間も計算しない。**")
    add("計算すれば「発見」に見えてしまうが、それは多重比較の産物にすぎない。")
    add("ここでの目的は候補を選ぶことだけであり、")
    add(f"エッジの有無を判定するのは未使用の Part B（{PART_B_START} .. {PART_B_END}）でのみ行う。")
    add("")
    add("Hold-out（2026-06-01 .. 2026-08-31）は今回も開封しない。")
    add("")

    add("## 1. 選定ルール（グリッドを回す前にコードとして確定・コミット済み）")
    add("")
    add("`phase1c_grid.py` に実装。実行前にコミットしてある。")
    add("")
    add("| # | ルール |")
    add("|---|---|")
    add("| 1 | スコア = 6 ホライズンでの順位の平均（同じ時間足・同じホライズンの中で"
        "65 設定を平均 ATP リターンの降順に並べた順位。小さいほど良い）|")
    add(f"| 2 | **近傍安定性（必須）**: 各パラメータ ±1 段階の隣接設定の"
        f"スコア中央値が、その時間足の上位 {NEIGHBOUR_TOP_FRACTION:.0%} に入ること |")
    add("| 3 | **3 銘柄の符号一致（必須）**: 判定に使うホライズンで BTC/ETH/SOL の符号が揃うこと |")
    add(f"| 4 | 条件を満たす中からスコア順に上位 **{MAX_CANDIDATES_FROM_GRID} つ**（族をまたいで合計）|")
    add("| 5 | これに ADX length=14 / threshold=25 / 15分足 を**無条件で追加**"
        "（Phase 1b の副次系列。選定ルールを経由しない）|")
    add("")
    add("判定に使うホライズンは、その (設定, 時間足) で**順位が最良だったもの**を機械的に採用する。")
    add("Part B の結果を見てから変えることはしない。")
    add("")
    add("**なぜ近傍安定性が必要か**: 本物の効果ならパラメータを少し動かしても急には消えない。")
    add("運で当たった設定は「隣が全部悪いのに 1 点だけ良い」孤立した山になる。")
    add("**自分のスコアではなく隣のスコアを見る**ことで、自分が偶然良かっただけの設定を落とす。")
    add("")

    add("## 2. グリッド全体（良かった設定だけを載せない）")
    add("")
    add("順位 = 6 ホライズンの平均順位（65 設定中。1 が最良、65 が最悪、中央値は 33）。")
    add("平均ATR = その設定の最良ホライズンでの符号方向リターン（ATR 単位）。")
    add("")
    for family in ("adx", "wavetrend", "squeeze"):
        add(f"### {FAMILY_LABEL[family]}")
        add("")
        add(_grid_table(results, breakdown, family))
        add("")

    blue = results[results["family"] == "squeeze"].groupby("label")["blue_bars"].max()
    with_blue = int((blue > 0).sum())
    add(f"**族 3 の補足**: BB 期間 ≠ KC 期間の設定では中心線がずれ、"
        f"青（no_sqz）が実際に発生した（27 設定中 **{with_blue}** 設定で青あり、"
        f"最大 {int(blue.max()):,} 本）。")
    add("Phase 1 追試で「既定パラメータでは青が構造的に発生しない」と結論した根拠"
        "（BB 期間 = KC 期間なら中心線が一致する）が、逆側からも確認できた。")
    add("")

    add("## 3. 「平らな丘」か「孤立した山」か")
    add("")
    add("自分のスコアと、隣接設定のスコア中央値との Spearman 相関。")
    add("1 に近い = 表面がなめらかで平らな丘がある。0 に近い = 隣と無関係でノイズ。")
    add("")
    pivot = plateau.pivot_table(index="family", columns="interval", values="spearman")
    pivot.index = [FAMILY_LABEL[i] for i in pivot.index]
    add(pivot.round(3).to_markdown())
    add("")
    for family in ("adx", "wavetrend", "squeeze"):
        subset = plateau[plateau["family"] == family]
        mean_correlation = subset["spearman"].mean()
        if mean_correlation > 0.5:
            verdict = "**平らな丘がある**（隣接設定と滑らかにつながっている）"
        elif mean_correlation > 0.2:
            verdict = "**弱い連続性**（部分的に丘があるが、ノイズも大きい）"
        else:
            verdict = "**孤立した山ばかり**（隣と無関係。良い設定は偶然の可能性が高い）"
        add(f"- {FAMILY_LABEL[family]}: 平均 Spearman = {mean_correlation:+.3f} → {verdict}")
    add("")

    add("## 4. 選定ルールの適用結果")
    add("")
    add(f"- 全 {len(breakdown)} 組（65 設定 × 3 時間足）のうち")
    add(f"  - 近傍安定性を通過: **{int(breakdown['passes_neighbour'].sum())}**")
    add(f"  - 3 銘柄の符号一致を通過: **{int(breakdown['passes_signs'].sum())}**")
    add(f"  - 両方通過: **{int(breakdown['passes_all'].sum())}**")
    add("")
    add("### 通過した組（スコア順・上位20件まで）")
    add("")
    passed = breakdown[breakdown["passes_all"]].sort_values("score").head(20)
    if len(passed):
        frame = passed[["setting_key", "interval", "score", "score_percentile",
                        "neighbour_percentile", "best_horizon", "mean_atr",
                        "mean_bps", "n_signals"]].copy()
        frame.columns = ["設定", "時間足", "順位", "自分%ile", "隣%ile", "最良N",
                         "平均ATR", "平均bps", "n"]
        add(frame.round(3).to_markdown(index=False))
        add("")
        add(f"上位 {MAX_CANDIDATES_FROM_GRID} 件が候補になる。")
    else:
        add("**通過した組は 0 件。**")
    add("")

    add("### 惜しくも落ちた設定（スコア上位だが必須条件で落ちたもの）")
    add("")
    dropped = breakdown[~breakdown["passes_all"]].sort_values("score").head(15)
    frame = dropped[["setting_key", "interval", "score", "neighbour_percentile",
                     "passes_neighbour", "passes_signs", "mean_atr"]].copy()
    frame.columns = ["設定", "時間足", "順位", "隣%ile", "近傍安定性", "符号一致", "平均ATR"]
    add(frame.round(3).to_markdown(index=False))
    add("")
    add("スコアが良くても隣が悪ければ落ちる。これが近傍安定性の効果である。")
    add("")

    add("## 5. Part B に持ち込む候補（確定）")
    add("")
    add("**この一覧を確定させてからでないと、2023-2024 のデータには触れない。**")
    add("")
    add("| # | 族 | パラメータ | 時間足 | 判定ホライズン | 由来 | Part A 平均ATR | Part A 平均bps |")
    add("|---|---|---|---|---|---|---|---|")
    for i, candidate in enumerate(candidates, start=1):
        params = ", ".join(f"{k}={v}" for k, v in candidate["params"].items())
        add(f"| {i} | {candidate['family']} | {params} | {candidate['interval']} | "
            f"N={candidate['horizon']} | {candidate['origin']} | "
            f"{candidate['part_a_mean_atr']:+.4f} | {candidate['part_a_mean_bps']:+.2f} |")
    add("")
    add(f"候補数: **{len(candidates)}**。Part B の Bonferroni 補正は "
        f"0.05 / {len(candidates)} = **{0.05 / len(candidates):.4f}**。")
    add("")
    add("Part A の平均値は参考情報であり、**判定には一切使わない**。")
    add("この期間で良く見えるのは当然（そう選んだのだから）。")
    add("")
    add("記録: `reports/phase1c_registered_candidates.json`")
    add("")

    (REPORT_DIR / "phase1c_exploration.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
