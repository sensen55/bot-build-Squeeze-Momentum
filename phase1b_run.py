"""Phase 1b の実行と Markdown レポート生成。

実行: python3 phase1b_run.py
出力: reports/phase1b_leader_and_squeeze.md, reports/phase1b_*.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from main_indicators import WARMUP_BARS
from phase1_nearmiss import PERIOD_EDGES
from phase1b_event_study import (
    ALPHA, ALPHA_Q1, ALPHA_Q2, COST_FLOOR_BPS, DESCRIPTIVE_LOOKBACKS, HORIZONS,
    INTERVALS, LEADERS, Q1_CELL_COUNT, Q2_CELL_COUNT, SQUEEZE_LOOKBACK_K,
    STUDY_END, STUDY_START, SYMBOLS,
    analyse_leader, analyse_squeeze_groups, block_bootstrap_mean_ci,
    build_signal_table, select_non_overlapping,
)
from scipy import stats

REPORT_DIR = Path(__file__).resolve().parent / "reports"

LEADER_LABEL = {"wavetrend": "WaveTrend", "adx": "ADX"}


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)

    # --- シグナル表の構築 ----------------------------------------------------
    tables: dict[tuple[str, str], list] = {}
    secondary_tables: dict[tuple[str, str], list] = {}
    count_rows = []
    for leader in LEADERS:
        for interval in INTERVALS:
            built = [build_signal_table(leader, symbol, interval, "primary")
                     for symbol in SYMBOLS]
            tables[(leader, interval)] = built
            secondary_tables[(leader, interval)] = [
                build_signal_table(leader, symbol, interval, "secondary")
                for symbol in SYMBOLS]
            print(f"  {leader:9s} {interval:3s}: "
                  f"{[len(t.signals) for t in built]} "
                  f"(副次 {[len(t.signals) for t in secondary_tables[(leader, interval)]]})",
                  flush=True)
            for table in built:
                counts = table.signals["group"].value_counts()
                overlap = {}
                for horizon in (6, 48):
                    keep = select_non_overlapping(
                        table.signals["index"].to_numpy(), horizon)
                    overlap[f"n_non_overlap_{horizon}"] = len(keep)
                count_rows.append({
                    "leader": leader, "interval": interval, "symbol": table.symbol,
                    "n_signals": len(table.signals), "n_bars": table.bar_count,
                    "n_secondary": len(
                        [t for t in secondary_tables[(leader, interval)]
                         if t.symbol == table.symbol][0].signals),
                    "n_S1": int(counts.get("S1", 0)), "n_S2": int(counts.get("S2", 0)),
                    "n_S3": int(counts.get("S3", 0)),
                    "zero_division_bars": table.zero_division_bars,
                    **overlap,
                })
    counts_frame = pd.DataFrame(count_rows)
    counts_frame.to_csv(REPORT_DIR / "phase1b_signal_counts.csv", index=False)

    # --- 問 1: 主役単体 ------------------------------------------------------
    q1_rows = []
    for leader in LEADERS:
        for interval in INTERVALS:
            for horizon in HORIZONS:
                print(f"  問1 {leader} {interval} N={horizon}", flush=True)
                q1_rows.append({"leader": leader, "interval": interval,
                                **analyse_leader(tables[(leader, interval)], horizon)})
    q1 = pd.DataFrame(q1_rows)
    q1["passes"] = (q1["ci_low_bps"] > COST_FLOOR_BPS) & (q1["p_value"] < ALPHA_Q1)
    q1.to_csv(REPORT_DIR / "phase1b_q1_leader.csv", index=False)

    # 副次シグナル定義の記述統計。依頼書どおり判定には使わない
    # (Bonferroni の族にも含めない)。
    secondary_rows = []
    for leader in LEADERS:
        for interval in INTERVALS:
            for horizon in HORIZONS:
                print(f"  副次 {leader} {interval} N={horizon}", flush=True)
                secondary_rows.append({
                    "leader": leader, "interval": interval,
                    **analyse_leader(secondary_tables[(leader, interval)], horizon),
                })
    secondary = pd.DataFrame(secondary_rows)
    secondary.to_csv(REPORT_DIR / "phase1b_secondary_descriptive.csv", index=False)

    # --- 問 2: 主役が合格した場合のみ ---------------------------------------
    q1_passing_leaders = [
        leader for leader in LEADERS
        if bool(q1[(q1["leader"] == leader) & q1["passes"]].shape[0])
    ]
    q2_rows, descriptive_rows = [], []
    for leader in q1_passing_leaders:
        for interval in INTERVALS:
            for horizon in HORIZONS:
                print(f"  問2 {leader} {interval} N={horizon}", flush=True)
                q2_rows.append({
                    "leader": leader, "interval": interval,
                    **analyse_squeeze_groups(
                        tables[(leader, interval)], horizon, SQUEEZE_LOOKBACK_K),
                })
    # k の記述統計 (判定には使わない) は主役の合否に関わらず出す
    for leader in LEADERS:
        for interval in INTERVALS:
            for k in DESCRIPTIVE_LOOKBACKS:
                descriptive_rows.append({
                    "leader": leader, "interval": interval,
                    **analyse_squeeze_groups(tables[(leader, interval)], 6, k),
                })

    q2 = pd.DataFrame(q2_rows)
    descriptive = pd.DataFrame(descriptive_rows)
    if len(q2):
        q2["passes"] = (
            (q2["diff_ci_low"] > 0.0)
            & (q2["welch_p"] < ALPHA_Q2)
            & (q2["s1_ci_low_bps"] > COST_FLOOR_BPS)
        )
        q2.to_csv(REPORT_DIR / "phase1b_q2_squeeze.csv", index=False)
    descriptive.to_csv(REPORT_DIR / "phase1b_lookback_descriptive.csv", index=False)

    scrutiny = _scrutinise_best(q1, q2, tables) if (len(q2) and q2["passes"].any()) else []

    _write_report(counts_frame, q1, q2, descriptive, secondary,
                  q1_passing_leaders, scrutiny)
    print("レポート:", REPORT_DIR / "phase1b_leader_and_squeeze.md")


def _scrutinise_best(q1, q2, tables) -> list[str]:
    """問 2 が合格した場合、最良セルを Phase 1 と同じ観点で精査する。"""
    best = q2.loc[q2[q2["passes"]]["diff_ci_low"].idxmax()]
    leader, interval, horizon = best["leader"], best["interval"], int(best["horizon"])
    built = tables[(leader, interval)]
    column = f"ret_{horizon}_atr"
    lines = [f"最良セル: **{LEADER_LABEL[leader]} / {interval} / N={horizon}**", ""]

    lines += ["**(a) 銘柄別内訳（S1 の符号方向リターン、ATR 単位）**", "",
              "| 銘柄 | n | 平均(ATR) | t値 | p値 |", "|---|---:|---:|---:|---:|"]
    for table in built:
        subset = table.signals[
            (table.signals["group"] == "S1") & table.signals[column].notna()]
        signed = (subset["direction"] * subset[column]).to_numpy()
        if len(signed) < 2:
            lines.append(f"| {table.symbol} | {len(signed)} | - | - | - |")
            continue
        result = stats.ttest_1samp(signed, 0.0)
        lines.append(f"| {table.symbol} | {len(signed)} | {signed.mean():.4f} | "
                     f"{result.statistic:.2f} | {result.pvalue:.4f} |")
    lines.append("")

    combined = pd.concat([t.signals for t in built], ignore_index=True).dropna(
        subset=[column])
    combined["signed"] = combined["direction"] * combined[column]
    lines += ["**(b) 期間 3 分割（S1 − S3 の差）**", "",
              "| 期間 | S1 n | S3 n | 差(ATR) | Welch p |",
              "|---|---:|---:|---:|---:|"]
    edges = pd.to_datetime(PERIOD_EDGES, utc=True)
    for start, end in zip(edges[:-1], edges[1:]):
        window = combined[(combined["time"] >= start) & (combined["time"] < end)]
        s1 = window[window["group"] == "S1"]["signed"].to_numpy()
        s3 = window[window["group"] == "S3"]["signed"].to_numpy()
        if len(s1) < 2 or len(s3) < 2:
            lines.append(f"| {start.date()} .. {end.date()} | {len(s1)} | {len(s3)} | - | - |")
            continue
        welch = stats.ttest_ind(s1, s3, equal_var=False)
        lines.append(f"| {start.date()} .. {end.date()} | {len(s1)} | {len(s3)} | "
                     f"{s1.mean() - s3.mean():.4f} | {welch.pvalue:.4f} |")
    lines.append("")

    lines += ["**(c) 隣接ホライズン（S1 − S3 の差）**", "",
              "| N本後 | 差(ATR) | Welch p | 差の95%CI下限 |", "|---:|---:|---:|---:|"]
    for h in HORIZONS:
        result = analyse_squeeze_groups(built, h, SQUEEZE_LOOKBACK_K)
        mark = " ←最良" if h == horizon else ""
        lines.append(f"| {h} | {result['difference_atr']:.4f} | "
                     f"{result['welch_p']:.4f} | {result['diff_ci_low']:.4f}{mark} |")
    lines.append("")

    lines += ["**(d) 窓非重複の実効標本（S1 のみ）**", ""]
    for table in built:
        s1_index = table.signals[table.signals["group"] == "S1"]["index"].to_numpy()
        keep = select_non_overlapping(s1_index, horizon)
        lines.append(f"- {table.symbol}: S1 全 {len(s1_index)} 件 → "
                     f"窓非重複 {len(keep)} 件")
    lines.append("")
    return lines


def _write_report(counts, q1, q2, descriptive, secondary,
                  q1_passing_leaders, scrutiny) -> None:
    lines: list[str] = []
    add = lines.append

    add("# Phase 1b: 「主役 = WaveTrend / ADX、補助 = スクイーズ」仮説の検証")
    add("")
    add(f"- 対象: {', '.join(SYMBOLS)} / {', '.join(INTERVALS)}")
    add(f"- 期間: {STUDY_START} .. {STUDY_END}（Hold-out 2026-06-01..2026-08-31 は封印・未使用）")
    add(f"- EMA の立ち上がりを避けるため、先頭 **{WARMUP_BARS} 本**を解析対象外にした")
    add("")
    add("Phase 1 の結論（スクイーズ解放時の `val` の符号に方向情報なし）は変更していない。")
    add("これは別の仮説の検証であり、再挑戦ではない。")
    add("")

    add("## 0. 事前に確定した判定ルール（実行前に固定・後から緩めていない）")
    add("")
    add("| 項目 | 問 1（主役単体）| 問 2（補助の価値）|")
    add("|---|---|---|")
    add(f"| セル数 | {Q1_CELL_COUNT}（2 主役 × 3 時間足 × 6 ホライズン）| "
        f"{Q2_CELL_COUNT}（合格主役の 3 時間足 × 6 ホライズン）|")
    add(f"| 有意水準（Bonferroni）| {ALPHA}/{Q1_CELL_COUNT} = **{ALPHA_Q1:.5f}** | "
        f"{ALPHA}/{Q2_CELL_COUNT} = **{ALPHA_Q2:.5f}** |")
    add(f"| 合格条件 | 95%CI 下限（bps）> {COST_FLOOR_BPS} bps かつ補正後に有意 | "
        f"差の 95%CI 下限 > 0 かつ補正後に有意 かつ S1 の絶対水準 > {COST_FLOOR_BPS} bps |")
    add("| 主検定の単位 | ATR 正規化リターン（コスト判定のみ bps）| 同左 |")
    add("")
    add("**Phase 1 のレビューを受けた変更**: 主検定を ATR 単位にした。")
    add("Phase 1 では bps でプールしたため値動きの大きい SOL に平均が引っ張られた")
    add("（1h/N=48 の +35.2 bps は SOL 単独で +78.8 bps だった）。")
    add("ATR 単位なら「その銘柄の普段の値動きの何倍動いたか」で公平に比較できる。")
    add("")
    add(f"問 2 の「解放直後」は **直近 {SQUEEZE_LOOKBACK_K} 本以内に黒→灰の解放があった**")
    add("と定義し、事前に固定した。k = 3 / 12 は記述統計としてのみ併記し、判定には使わない。")
    add("")

    add("## 1. 主役 2 つの実装が原典と一致することの根拠")
    add("")
    add("`tests/test_main_indicators.py` の 30 項目すべて合格。内訳:")
    add("")
    add("| 検証 | 結果 |")
    add("|---|---|")
    add("| EMA が漸化式 `y[i]=α·x[i]+(1−α)·y[i−1]`（α=2/(n+1)）と一致 | 合格 |")
    add("| Wilder 平滑化が α=1/n の漸化式と一致 | 合格 |")
    add("| `adjust=True`（pandas 既定）と `adjust=False` が別物であることを確認 | 合格 |")
    add("| EMA(14) と Wilder(14) が別物であることを確認（α=0.1333 vs 0.0714）| 合格 |")
    add("| +DM/−DM の「大きい方だけ採用、小さい方はゼロ」を合成データで確認 | 合格 |")
    add("| WaveTrend 8 系列が独立実装（素朴なループ）と一致（相対誤差 < 1e−9）| 合格 |")
    add("| ADX 7 系列が独立実装と一致（相対誤差 < 1e−9）| 合格 |")
    add("| ADX が 0..100、DI± が非負 | 合格 |")
    add("| WaveTrend / ADX がスケール非依存（価格 1000 倍で不変）| 合格 |")
    add("| ルックアヘッド検査器の陰性対照 6 種をすべて検出 | 合格 |")
    add("| WaveTrend / ADX に対する切断テスト | ルックアヘッドなし |")
    add("")
    add("`tests/test_phase1b.py` の 13 項目すべて合格（将来リターンの手計算検算、")
    add("シグナル抽出の整合性、S1/S2/S3 群分けの独立検算、高速ブートストラップの一致確認）。")
    add("")
    zero_division = int(counts["zero_division_bars"].max())
    add(f"**ゼロ除算の扱い**: WaveTrend の `ci = (ap−esa)/(0.015·d)` は `d = 0` のとき発散する。")
    add(f"該当バーは NaN にした。実データでの発生は各データセット **{zero_division} 本**で、")
    add("いずれも先頭バー（`esa[0] = ap[0]` なので `d[0] = 0` になる構造上の必然）であり、")
    add(f"先頭 {WARMUP_BARS} 本の除外に含まれるため解析には影響しない。")
    add("ADX 側で平滑化 TR が 0 になったバーは 0 本だった。")
    add("")

    add("## 2. シグナル数")
    add("")
    shown = counts.copy()
    shown["シグナル率"] = (shown["n_signals"] / shown["n_bars"]).map(lambda v: f"{v:.2%}")
    add(shown[["leader", "interval", "symbol", "n_signals", "n_bars", "シグナル率",
               "n_secondary", "n_S1", "n_S2", "n_S3",
               "n_non_overlap_6", "n_non_overlap_48"]].rename(columns={
        "leader": "主役", "interval": "時間足", "symbol": "銘柄",
        "n_signals": "シグナル数", "n_bars": "バー数", "n_secondary": "副次条件",
        "n_S1": "S1解放直後", "n_S2": "S2縮小中", "n_S3": "S3それ以外",
        "n_non_overlap_6": "窓非重複(N=6)", "n_non_overlap_48": "窓非重複(N=48)",
    }).to_markdown(index=False))
    add("")
    add("窓非重複の実効標本は、N=48 では元の 1〜2 割まで落ちる。")
    add("見かけのシグナル数ほど独立な観測があるわけではない。")
    add("")

    add("## 3. 問 1: 主役は単体で方向情報を持つか【主検定 36 セル】")
    add("")
    add("符号方向リターン = direction[t] × (close[t+N] − open[t+1]) / ATR[t]")
    add("")
    add("判断はシグナルバー t の終値確定時点、約定は t+1 の始値。必ず 1 バー以上離れている。")
    add("")
    for leader in LEADERS:
        subset = q1[q1["leader"] == leader]
        add(f"### {LEADER_LABEL[leader]}")
        add("")
        frame = subset[["interval", "horizon", "n", "mean_atr", "p_value",
                        "ci_low_atr", "ci_high_atr", "mean_bps", "ci_low_bps",
                        "win_rate", "mean_atr_BTCUSDT", "mean_atr_ETHUSDT",
                        "mean_atr_SOLUSDT", "signs_agree", "non_overlap_p"]].copy()
        frame.columns = ["時間足", "N本後", "n", "平均(ATR)", "p値", "CI下限(ATR)",
                         "CI上限(ATR)", "平均(bps)", "CI下限(bps)", "勝率",
                         "BTC", "ETH", "SOL", "符号一致", "窓非重複p"]
        for column in frame.columns:
            if frame[column].dtype.kind == "f":
                decimals = 2 if column in ("平均(bps)", "CI下限(bps)") else 4
                frame[column] = frame[column].map(lambda v, d=decimals: f"{v:,.{d}f}")
        add(frame.to_markdown(index=False))
        add("")
        add(f"- Bonferroni 補正後に有意（p < {ALPHA_Q1:.5f}）: "
            f"**{int((subset['p_value'] < ALPHA_Q1).sum())} / {len(subset)}**")
        add(f"- 95%CI 下限が {COST_FLOOR_BPS} bps 超: "
            f"**{int((subset['ci_low_bps'] > COST_FLOOR_BPS).sum())} / {len(subset)}**")
        add(f"- 3 銘柄で符号が一致: **{int(subset['signs_agree'].sum())} / {len(subset)}**")
        add(f"- **合格セル: {int(subset['passes'].sum())} / {len(subset)}**")
        add("")

    add("### 副次シグナル定義（記述統計のみ・判定には使わない）")
    add("")
    add("依頼書の副次条件。WaveTrend は売られすぎ帯（`wt1 < −53`）での上抜けと")
    add("買われすぎ帯（`> 53`）での下抜けに限定したもの、ADX は閾値 **25** の上抜け。")
    add("")
    add("ADX の 25 上抜けは 20 上抜けとは**別のバー**なので、主シグナルの部分集合ではなく")
    add("独立したシグナル集合として集計している。")
    add("")
    add("**これらは Bonferroni の族に含めておらず、ここから合格を主張することはできない。**")
    add("")
    secondary_frame = secondary[["leader", "interval", "horizon", "n", "mean_atr",
                                 "p_value", "ci_low_bps", "win_rate",
                                 "signs_agree"]].copy()
    secondary_frame.columns = ["主役", "時間足", "N本後", "n", "平均(ATR)", "p値",
                               "CI下限(bps)", "勝率", "符号一致"]
    for column in secondary_frame.columns:
        if secondary_frame[column].dtype.kind == "f":
            decimals = 2 if column == "CI下限(bps)" else 4
            secondary_frame[column] = secondary_frame[column].map(
                lambda v, d=decimals: f"{v:,.{d}f}")
    add(secondary_frame.to_markdown(index=False))
    add("")
    add(f"- 参考: 95%CI 下限が {COST_FLOOR_BPS} bps を超えたセル "
        f"**{int((secondary['ci_low_bps'] > COST_FLOOR_BPS).sum())} / {len(secondary)}**、"
        f"補正なしの p < 0.05 が "
        f"**{int((secondary['p_value'] < ALPHA).sum())} / {len(secondary)}**"
        f"（36 セル中 0.05 水準で偶然 2 個前後は出る）")
    add("")

    # 副次族に目立つ結果が出た場合、隠さずに扱いを明示する
    notable = secondary[
        (secondary["ci_low_bps"] > COST_FLOOR_BPS) & (secondary["p_value"] < ALPHA)
    ]
    if len(notable):
        add("#### この副次結果の扱い（重要）")
        add("")
        add("副次族に、無視できない数字が出ている。**都合が悪いので伏せる、はしない。**")
        add("まず事実を並べる。")
        add("")
        table = notable[["leader", "interval", "horizon", "n", "mean_atr", "p_value",
                         "ci_low_bps", "mean_atr_BTCUSDT", "mean_atr_ETHUSDT",
                         "mean_atr_SOLUSDT", "signs_agree"]].copy()
        table.columns = ["主役", "時間足", "N本後", "n", "平均(ATR)", "p値",
                         "CI下限(bps)", "BTC", "ETH", "SOL", "符号一致"]
        for column in table.columns:
            if table[column].dtype.kind == "f":
                decimals = 2 if column == "CI下限(bps)" else 4
                table[column] = table[column].map(lambda v, d=decimals: f"{v:,.{d}f}")
        add(table.to_markdown(index=False))
        add("")
        bonferroni_survivors = int((notable["p_value"] < ALPHA_Q1).sum())
        add("**有利な点**")
        add("")
        add("- 複数のホライズンで連続してプラスであり、Phase 1 の 1h/N=48 のような")
        add("  「隣接ホライズンで消える孤立ピーク」の形ではない")
        add("- 3 銘柄すべてで符号が一致している（Phase 1 では SOL 単独依存だった）")
        add(f"- 95%CI 下限が最良コスト {COST_FLOOR_BPS} bps を超えている")
        add("")
        add("**不利な点**")
        add("")
        add("- この 36 セルは、実行前に「記述統計のみ・判定には使わない」と")
        add("  **宣言済みの族**である")
        add(f"- 仮にこの族へ同じ Bonferroni（{ALPHA}/{Q1_CELL_COUNT} = {ALPHA_Q1:.5f}）を")
        add(f"  当てると、生き残るのは **{bonferroni_survivors} / {len(secondary)} セル**だけ")
        add("- 時間足を跨いだ一貫性がない（同じ定義でも 5 分足・1 時間足では")
        add("  CI 下限がコストを超えない）")
        add("- 窓が重なっているため、実効標本は表の n より少ない")
        add("- そして最も重要な点: **閾値 20 が不合格で 25 が良かったから 25 を採る、")
        add("  というのは依頼書が明示的に禁じた「パラメータを振って良い方を選ぶ」")
        add("  行為そのもの**である")
        add("")
        add("**したがって Phase 1b の判定は不合格のまま変えない。**")
        add("")
        add("この系列を本気で検証したいなら、それは**新しい事前登録を要する別プロジェクト**")
        add("になる。封印中の Hold-out（2026-06-01 .. 2026-08-31）は、まさにそのために")
        add("取ってある。着手するかどうかの判断は依頼者に委ねる。")
        add("")

    add("## 4. 問 1 の判定")
    add("")
    for leader in LEADERS:
        subset = q1[q1["leader"] == leader]
        verdict = "合格セルあり" if subset["passes"].any() else "**合格セル 0 — 不合格**"
        add(f"- {LEADER_LABEL[leader]}: {verdict}"
            f"（{int(subset['passes'].sum())} / {len(subset)}）")
    add("")

    if not q1_passing_leaders:
        add("**両主役とも合格セル 0。事前の中止基準により、問 2 は実施しない。**")
        add("")
        add("中止基準の文言: 「問 1 で両主役とも合格セル 0 → 終了。スクイーズの検証も")
        add("ここで終わる（主役に情報がなければ、補助で選別しても情報は生まれない）」")
        add("")

    add("## 5. 問 2: スクイーズは主役を改善するか")
    add("")
    if not len(q2):
        add("**未実施。** 問 1 で合格した主役が無かったため。")
        add("")
        add("参考として、群ごとの記述統計（N=6、判定には使わない）だけ以下に示す。")
        add("")
        table = descriptive[descriptive["k"] == SQUEEZE_LOOKBACK_K][
            ["leader", "interval", "n_S1", "mean_atr_S1", "n_S2", "mean_atr_S2",
             "n_S3", "mean_atr_S3", "difference_atr", "welch_p"]].copy()
        table.columns = ["主役", "時間足", "S1 n", "S1 平均(ATR)", "S2 n", "S2 平均(ATR)",
                         "S3 n", "S3 平均(ATR)", "S1−S3", "Welch p"]
        for column in table.columns:
            if table[column].dtype.kind == "f":
                table[column] = table[column].map(lambda v: f"{v:,.4f}")
        add(table.to_markdown(index=False))
        add("")
        add(f"（k = {SQUEEZE_LOOKBACK_K}。k = {DESCRIPTIVE_LOOKBACKS} の比較は "
            "`reports/phase1b_lookback_descriptive.csv`）")
        add("")
        add("**この表から合否を主張してはいけない。** 問 1 が不合格である以上、")
        add("S1 と S3 の差が偶然どちらに転んでも、主役自体に方向情報が無いことは変わらない。")
        add("")
    else:
        frame = q2[["leader", "interval", "horizon", "n_S1", "mean_atr_S1",
                    "n_S3", "mean_atr_S3", "difference_atr", "welch_p",
                    "diff_ci_low", "s1_ci_low_bps", "passes"]].copy()
        frame.columns = ["主役", "時間足", "N本後", "S1 n", "S1 平均(ATR)", "S3 n",
                         "S3 平均(ATR)", "差(S1−S3)", "Welch p", "差CI下限",
                         "S1 CI下限(bps)", "合格"]
        add(frame.to_markdown(index=False))
        add("")
        add(f"- **合格セル: {int(q2['passes'].sum())} / {len(q2)}**")
        add("")
        if scrutiny:
            add("### 最良セルの精査")
            add("")
            lines.extend(scrutiny)

    add("## 6. 最終判定")
    add("")
    if not q1_passing_leaders:
        add("### 中止基準に到達 — Phase 1b をここで終了する")
        add("")
        add("事前に確定した合格条件を満たしたセルは、WaveTrend / ADX とも **0 / 18**"
            f"（合計 0 / {Q1_CELL_COUNT}）だった。")
        add("")
        add("「作者が言及した ADX / WaveTrend が主役で、スクイーズはその補助だった」")
        add("という仮説は、**主役の側で成立しなかった**。標準設定の WaveTrend クロスにも")
        add("ADX の 20 上抜けにも、往復コストを超える方向情報は確認できなかった。")
        add("")
        add("補助（スクイーズ）で選別しても、元の主役に情報が無い以上は情報が生まれない。")
        add("したがって問 2 には進まない。")
        add("")
    elif len(q2) and q2["passes"].any():
        add("### 問 1・問 2 とも合格セルあり — 判断を待つ")
        add("")
        add("**Phase 2 には自動で進んでいない。** 上の精査結果を見て判断してほしい。")
        add("")
    else:
        add("### 問 1 合格・問 2 不合格 — スクイーズは不要と結論")
        add("")
        add("主役単体には情報があったが、スクイーズ状態で層別しても改善しなかった。")
        add("主役単体を別プロジェクトとして検討するかは、判断を待つ。")
        add("")

    add("### 言えること / 言えないこと")
    add("")
    add("- 言える: **標準設定の** WaveTrend / ADX を、**この定義のシグナル**で使う限り、")
    add("  方向情報は確認できない。スクイーズを補助に足す前段が成立していない。")
    add("- 言えない: 「WaveTrend / ADX に一切情報が無い」とは言えない。")
    add("  パラメータ（10/21、14、閾値 20）は作者が言及した標準設定に固定しており、")
    add("  最適化していない。これは意図的な制約であり、")
    add("  中止基準到達後にパラメータを振り直すのはデータスヌーピングになる。")
    add("- WaveTrend の `wt2` は `wt1` の 4 本移動平均にすぎないため、")
    add("  このクロスは「wt1 が直近 4 本平均を上下に横切った」以上の意味を持たない。")
    add("  クロス頻度が高く（5 分足でバーの十数 % がシグナル）、")
    add("  ほぼ全バーに賭けているのに近い状態になる点は、設計上の限界として記録しておく。")
    add("")
    add("Hold-out 期間（2026-06-01 .. 2026-08-31）は**封印したまま未使用**。")
    add("")

    (REPORT_DIR / "phase1b_leader_and_squeeze.md").write_text(
        "\n".join(lines), encoding="utf-8")


def report_only() -> None:
    """既存の CSV からレポートだけ作り直す (再集計しない)。

    集計は 15 分ほどかかるので、文面の修正だけのときはこちらを使う。
    数字は前回の実行結果そのままなので、集計ロジックを変えたときは
    main() を回し直すこと。
    """
    counts = pd.read_csv(REPORT_DIR / "phase1b_signal_counts.csv")
    q1 = pd.read_csv(REPORT_DIR / "phase1b_q1_leader.csv")
    secondary = pd.read_csv(REPORT_DIR / "phase1b_secondary_descriptive.csv")
    descriptive = pd.read_csv(REPORT_DIR / "phase1b_lookback_descriptive.csv")
    q2_path = REPORT_DIR / "phase1b_q2_squeeze.csv"
    q2 = pd.read_csv(q2_path) if q2_path.exists() else pd.DataFrame()
    q1_passing_leaders = [
        leader for leader in LEADERS
        if bool(q1[(q1["leader"] == leader) & q1["passes"]].shape[0])
    ]
    _write_report(counts, q1, q2, descriptive, secondary, q1_passing_leaders, [])
    print("レポート:", REPORT_DIR / "phase1b_leader_and_squeeze.md")


if __name__ == "__main__":
    import sys

    if "--report-only" in sys.argv:
        report_only()
    else:
        main()
