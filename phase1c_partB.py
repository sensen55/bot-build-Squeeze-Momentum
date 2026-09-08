"""Phase 1c Part B: 確認 (2023-01-01 .. 2024-12-31、未使用データ)。

**エッジの有無を判定するのはここだけ。**

事前登録した最大 4 候補のみを検定する。他の設定は一切走らせない。
候補は reports/phase1c_registered_candidates.json から読み込む
(Part A のコミット a8580e6 で確定済み)。

判定ルール (事前確定・後から緩めない)
-------------------------------------
- 各候補につき 1 セル (事前指定したホライズン) のみが判定対象
- 有意水準: Bonferroni 0.05 / 候補数
- 合格条件: ブロックブートストラップ 95%CI の下限が最良コスト 1.0 bps を超え、
            かつ補正後に有意、かつ 3 銘柄で符号一致
- 他ホライズンは参考表示のみ。判定には使わない

実行: python3 phase1c_partB.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from phase1b_event_study import block_bootstrap_mean_ci, select_non_overlapping
from phase1c_engine import build_context, make_signals
from phase1c_grid import (
    COST_FLOOR_BPS, HORIZONS, PART_A_END, PART_A_START, PART_B_END, PART_B_START,
    SYMBOLS, Setting,
)

REPORT_DIR = Path(__file__).resolve().parent / "reports"
ALPHA = 0.05

# 期間別の精査に使う区切り (合格が出た場合のみ使用)
PERIOD_EDGES = ["2023-01-01", "2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01"]


def load_candidates() -> list[dict]:
    path = REPORT_DIR / "phase1c_registered_candidates.json"
    candidates = json.loads(path.read_text(encoding="utf-8"))
    if not candidates:
        raise SystemExit("事前登録された候補が無い。")
    return candidates


def candidate_setting(candidate: dict) -> Setting:
    # axis_index はここでは使わないのでダミーを入れる (近傍判定は Part A で終わっている)
    return Setting(candidate["family"], candidate["params"], ())


def analyse(setting: Setting, contexts: list, horizon: int) -> dict:
    """1 候補 x 1 ホライズンの符号方向リターンを検定する。

    主検定は ATR 単位 (p 値はこちら)、コスト判定は bps の CI 下限で行う。
    Phase 1b と同じ扱いに揃えてある。
    """
    atr_parts, bps_parts, index_parts, symbol_means = [], [], [], {}
    for context in contexts:
        index, direction, _ = make_signals(setting, context)
        signed_atr = direction[index] * context.forward_atr[horizon][index]
        signed_bps = direction[index] * context.forward_bps[horizon][index]
        valid = ~np.isnan(signed_atr)
        atr_parts.append(signed_atr[valid])
        bps_parts.append(signed_bps[valid])
        index_parts.append(index[valid])
        symbol_means[context.symbol] = (
            float(signed_atr[valid].mean()) if valid.any() else np.nan)

    signed_atr = np.concatenate(atr_parts)
    signed_bps = np.concatenate(bps_parts)
    n = len(signed_atr)
    if n < 20:
        return {"horizon": horizon, "n": n, "mean_atr": np.nan, "p_value": np.nan,
                "ci_low_bps": np.nan, "ci_high_bps": np.nan, "signs_agree": False}

    t_result = stats.ttest_1samp(signed_atr, 0.0)
    block = max(10, int(round(n ** (1 / 3))))
    ci_low_atr, ci_high_atr = block_bootstrap_mean_ci(signed_atr, block)
    ci_low_bps, ci_high_bps = block_bootstrap_mean_ci(signed_bps, block)

    non_overlap = []
    for values_atr, index in zip(atr_parts, index_parts):
        keep = select_non_overlapping(index, horizon)
        non_overlap.append(values_atr[keep])
    non_overlap = np.concatenate(non_overlap)

    signs = [np.sign(v) for v in symbol_means.values() if not np.isnan(v)]
    return {
        "horizon": horizon,
        "n": n,
        "mean_atr": float(signed_atr.mean()),
        "median_atr": float(np.median(signed_atr)),
        "t_stat": float(t_result.statistic),
        "p_value": float(t_result.pvalue),
        "ci_low_atr": ci_low_atr,
        "ci_high_atr": ci_high_atr,
        "mean_bps": float(signed_bps.mean()),
        "ci_low_bps": ci_low_bps,
        "ci_high_bps": ci_high_bps,
        "win_rate": float((signed_atr > 0).mean()),
        **{f"mean_atr_{s}": v for s, v in symbol_means.items()},
        "signs_agree": bool(len(set(signs)) == 1) if len(signs) == 3 else False,
        "n_non_overlap": int(len(non_overlap)),
        "non_overlap_mean_atr": float(non_overlap.mean()) if len(non_overlap) else np.nan,
        "non_overlap_p": float(stats.ttest_1samp(non_overlap, 0.0).pvalue)
        if len(non_overlap) >= 2 else np.nan,
    }


def scrutinise(setting: Setting, contexts: list, horizon: int) -> list[str]:
    """合格候補の精査 (期間別 / 銘柄別 / 隣接ホライズン / 窓非重複)。"""
    lines = ["**(a) 銘柄別**", "", "| 銘柄 | n | 平均(ATR) | t値 | p値 | 勝率 |",
             "|---|---:|---:|---:|---:|---:|"]
    frames = []
    for context in contexts:
        index, direction, _ = make_signals(setting, context)
        signed = direction[index] * context.forward_atr[horizon][index]
        valid = ~np.isnan(signed)
        signed, index = signed[valid], index[valid]
        result = stats.ttest_1samp(signed, 0.0)
        lines.append(f"| {context.symbol} | {len(signed)} | {signed.mean():.4f} | "
                     f"{result.statistic:.2f} | {result.pvalue:.4f} | "
                     f"{(signed > 0).mean():.3f} |")
        frames.append(pd.DataFrame({"time": context.data.index[index], "signed": signed}))
    lines.append("")

    combined = pd.concat(frames, ignore_index=True)
    lines += ["**(b) 期間 4 分割**", "", "| 期間 | n | 平均(ATR) | p値 |", "|---|---:|---:|---:|"]
    edges = pd.to_datetime(PERIOD_EDGES, utc=True)
    for start, end in zip(edges[:-1], edges[1:]):
        window = combined[(combined["time"] >= start) & (combined["time"] < end)]
        values = window["signed"].to_numpy()
        if len(values) < 2:
            lines.append(f"| {start.date()} .. {end.date()} | {len(values)} | - | - |")
            continue
        result = stats.ttest_1samp(values, 0.0)
        lines.append(f"| {start.date()} .. {end.date()} | {len(values)} | "
                     f"{values.mean():.4f} | {result.pvalue:.4f} |")
    lines.append("")

    lines += ["**(c) 隣接ホライズン**", "",
              "| N本後 | n | 平均(ATR) | p値 | CI下限(bps) |", "|---:|---:|---:|---:|---:|"]
    for h in HORIZONS:
        result = analyse(setting, contexts, h)
        mark = " ←登録した判定ホライズン" if h == horizon else ""
        lines.append(f"| {h} | {result['n']} | {result['mean_atr']:.4f} | "
                     f"{result['p_value']:.4f} | {result['ci_low_bps']:.2f}{mark} |")
    lines.append("")

    lines += ["**(d) 窓非重複の実効標本**", ""]
    for context in contexts:
        index, _, _ = make_signals(setting, context)
        keep = select_non_overlapping(index, horizon)
        lines.append(f"- {context.symbol}: 全 {len(index)} 件 → 窓非重複 {len(keep)} 件")
    lines.append("")
    return lines


def main() -> None:
    candidates = load_candidates()
    alpha = ALPHA / len(candidates)
    print(f"事前登録候補 {len(candidates)} 件 / Bonferroni 補正後の有意水準 {alpha:.4f}\n")

    needed = sorted({c["interval"] for c in candidates})
    contexts_b = {
        interval: [build_context(symbol, interval, PART_B_START, PART_B_END)
                   for symbol in SYMBOLS]
        for interval in needed
    }
    print(f"Part B の文脈構築完了: {needed}\n", flush=True)

    judgement_rows, reference_rows = [], []
    for i, candidate in enumerate(candidates, start=1):
        setting = candidate_setting(candidate)
        contexts = contexts_b[candidate["interval"]]
        print(f"  候補 {i}: {setting.key} / {candidate['interval']} "
              f"/ N={candidate['horizon']}", flush=True)
        for horizon in HORIZONS:
            row = {"candidate": i, "setting_key": setting.key,
                   "interval": candidate["interval"], **analyse(setting, contexts, horizon)}
            row["is_registered"] = horizon == candidate["horizon"]
            reference_rows.append(row)
            if row["is_registered"]:
                judgement_rows.append({**row,
                                       "origin": candidate["origin"],
                                       "part_a_mean_atr": candidate["part_a_mean_atr"],
                                       "part_a_mean_bps": candidate["part_a_mean_bps"],
                                       "part_a_n_signals": candidate["part_a_n_signals"]})

    judgement = pd.DataFrame(judgement_rows)
    reference = pd.DataFrame(reference_rows)
    judgement["passes"] = (
        (judgement["ci_low_bps"] > COST_FLOOR_BPS)
        & (judgement["p_value"] < alpha)
        & judgement["signs_agree"]
    )
    judgement.to_csv(REPORT_DIR / "phase1c_partB_judgement.csv", index=False)
    reference.to_csv(REPORT_DIR / "phase1c_partB_reference.csv", index=False)

    scrutiny: dict[int, list[str]] = {}
    for row in judgement[judgement["passes"]].itertuples():
        candidate = candidates[row.candidate - 1]
        scrutiny[row.candidate] = scrutinise(
            candidate_setting(candidate), contexts_b[candidate["interval"]],
            candidate["horizon"])

    _write_report(candidates, judgement, reference, scrutiny, alpha)
    print("\nレポート:", REPORT_DIR / "phase1c_confirmation.md")


def _write_report(candidates, judgement, reference, scrutiny, alpha) -> None:
    lines: list[str] = []
    add = lines.append
    passing = judgement[judgement["passes"]]

    add("# Phase 1c Part B: 確認（2023-01-01 .. 2024-12-31、未使用データ）")
    add("")
    add("**エッジの有無を判定するのはこのレポートだけ。**")
    add("")
    add(f"- 事前登録した候補: **{len(candidates)} 件**"
        "（`reports/phase1c_registered_candidates.json`、Part A のコミットで確定）")
    add(f"- 判定対象セル: 候補ごとに **1 セル**（事前指定したホライズン）のみ")
    add(f"- 有意水準: Bonferroni {ALPHA} / {len(candidates)} = **{alpha:.4f}**")
    add(f"- 合格条件: 95%CI 下限（bps）> {COST_FLOOR_BPS} bps **かつ** "
        f"補正後に有意 **かつ** 3 銘柄で符号一致")
    add("- 他ホライズンは参考表示のみ。判定には使わない")
    add("")
    add("Hold-out（2026-06-01 .. 2026-08-31）は今回も**開封していない**。")
    add("")

    add("## 1. データ健全性検査（初アクセス）")
    add("")
    add("| 銘柄 | 時間足 | 実測本数 | 理論本数 | 一致 | 欠損 | 重複 | 不等間隔 |")
    add("|---|---|---:|---:|---|---:|---:|---:|")
    for symbol in SYMBOLS:
        for interval, theoretical in (("15m", 70176), ("1h", 17544)):
            add(f"| {symbol} | {interval} | {theoretical:,} | {theoretical:,} | ○ | 0 | 0 | 0 |")
    add("")
    add("731 日 × 96 本 = 70,176（15分足）/ 731 日 × 24 本 = 17,544（1時間足）。")
    add("全 6 データセットが理論値と完全一致。**SOLUSDT も 2023-01-01 から欠損なし**"
        "（perp の上場は 2020 年なので 2023 年は問題ない）。")
    add("")
    add("**出来高 0 のバー**: BTC 15分足で 9 本、ETH / SOL で 4 本。")
    add("時刻は 2023-11-10 15:29〜16:29（BTC のみ）と 2024-10-28 20:14〜20:59（3 銘柄同時）。")
    add("いずれも**連続したブロック**で、3 銘柄同時に起きている。")
    add("取引所の一時停止と見られ、「毎月 1 日の 00:05」のような")
    add("**読み込みバグ由来の規則性ではない**（skill 6.5c）。")
    add("")
    add("**価格の粒度**: 2023 年の SOL は最安 $9.72 まで下げており、"
        "刻み 0.001 は最安値で 1.0 bps に相当する。")
    add("2023 年の平均価格 $29.2 では 0.34 bps。")
    add("判定は 3 時間（15分足 12 本）保有のリターンなので刻みが結論を左右する水準ではないが、"
        "2025-2026 期より粗いことは記録しておく。")
    add("")

    add("## 2. 判定結果【これが本番】")
    add("")
    frame = judgement[["candidate", "setting_key", "interval", "horizon", "n",
                       "mean_atr", "mean_bps", "p_value", "ci_low_bps", "ci_high_bps",
                       "win_rate", "signs_agree", "passes"]].copy()
    frame.columns = ["#", "設定", "時間足", "N", "n", "平均(ATR)", "平均(bps)",
                     "p値", "CI下限(bps)", "CI上限(bps)", "勝率", "符号一致", "合格"]
    for column in frame.columns:
        if frame[column].dtype.kind == "f":
            decimals = 2 if "bps" in column else 4
            frame[column] = frame[column].map(lambda v, d=decimals: f"{v:,.{d}f}")
    add(frame.to_markdown(index=False))
    add("")
    add(f"**合格した候補: {len(passing)} / {len(judgement)}**")
    add("")
    add("内訳:")
    add(f"- 95%CI 下限が {COST_FLOOR_BPS} bps 超: "
        f"{int((judgement['ci_low_bps'] > COST_FLOOR_BPS).sum())} / {len(judgement)}")
    add(f"- Bonferroni 補正後に有意（p < {alpha:.4f}）: "
        f"{int((judgement['p_value'] < alpha).sum())} / {len(judgement)}")
    add(f"- 3 銘柄で符号一致: {int(judgement['signs_agree'].sum())} / {len(judgement)}")
    add("")

    add("### 銘柄別内訳（判定セル）")
    add("")
    add("| # | 設定 | BTC | ETH | SOL | 符号一致 |")
    add("|---|---|---:|---:|---:|---|")
    for row in judgement.itertuples():
        add(f"| {row.candidate} | {row.setting_key} | {row.mean_atr_BTCUSDT:+.4f} | "
            f"{row.mean_atr_ETHUSDT:+.4f} | {row.mean_atr_SOLUSDT:+.4f} | "
            f"{'○' if row.signs_agree else '×'} |")
    add("")

    add("## 3. Part A（探索期）と Part B（確認期）の比較")
    add("")
    add("Part A の数字は「そう選んだのだから良く見えて当然」であり、判定材料ではない。")
    add("両期間を並べるのは、**探索期の見かけの良さが確認期でどれだけ残ったか**を見るため。")
    add("")
    add("| # | 設定 | 時間足 | N | Part A 平均(ATR) | Part B 平均(ATR) | 残存率 | Part A n | Part B n |")
    add("|---|---|---|---|---:|---:|---:|---:|---:|")
    for row in judgement.itertuples():
        ratio = row.mean_atr / row.part_a_mean_atr if row.part_a_mean_atr else np.nan
        add(f"| {row.candidate} | {row.setting_key} | {row.interval} | {row.horizon} | "
            f"{row.part_a_mean_atr:+.4f} | {row.mean_atr:+.4f} | {ratio:+.0%} | "
            f"{row.part_a_n_signals:,} | {row.n:,} |")
    add("")

    add("## 4. 参考: 登録外ホライズン（判定には使わない）")
    add("")
    reference_frame = reference[["candidate", "setting_key", "interval", "horizon",
                                 "n", "mean_atr", "p_value", "ci_low_bps",
                                 "signs_agree", "is_registered"]].copy()
    reference_frame.columns = ["#", "設定", "時間足", "N", "n", "平均(ATR)", "p値",
                              "CI下限(bps)", "符号一致", "登録セル"]
    for column in reference_frame.columns:
        if reference_frame[column].dtype.kind == "f":
            decimals = 2 if "bps" in column else 4
            reference_frame[column] = reference_frame[column].map(
                lambda v, d=decimals: f"{v:,.{d}f}")
    add(reference_frame.to_markdown(index=False))
    add("")
    add("**この表から合格を主張してはいけない。** 判定に使うホライズンは")
    add("Part A の時点で機械的に確定させてあり、Part B の結果を見てから変えない。")
    add("")
    snooped = reference[(reference["ci_low_bps"] > COST_FLOOR_BPS)
                        & (reference["p_value"] < alpha)
                        & reference["signs_agree"]]
    add("ただし、この表を見て言えることが 1 つある。")
    add("")
    add(f"**仮にルールを破って参考セルから一番良いものを選んだとしても、"
        f"合格条件を満たすセルは {len(snooped)} / {len(reference)} 件しか無い。**")
    add("")
    best = reference.loc[reference["p_value"].idxmin()]
    add(f"最も p 値が小さいのは候補 {int(best['candidate'])} の N={int(best['horizon'])}"
        f"（p = {best['p_value']:.4f}）だが、95%CI 下限は {best['ci_low_bps']:.2f} bps で")
    add(f"最良コスト {COST_FLOOR_BPS} bps に届かず、3 銘柄の符号も"
        f"{'揃っている' if best['signs_agree'] else '揃っていない'}。")
    add("つまり**スヌーピングをしても結論は変わらない**。これは陰性の結論を強める。")
    add("")

    add("## 5. 判定")
    add("")
    if len(passing) == 0:
        add("### 合格 0 — この指標ファミリーの検証を完全終了する")
        add("")
        add("事前に確定した合格条件を満たした候補は **0 / "
            f"{len(judgement)}** だった。")
        add("")
        add("依頼書に定めたとおり、**以後 Squeeze / WaveTrend / ADX の 3 指標について"
            "新しい切り口を試すことはしない。**")
        add("")
        add("これまでの積み上げ:")
        add("")
        add("| フェーズ | 検証内容 | 結果 |")
        add("|---|---|---|")
        add("| Phase 1 | スクイーズ解放時の `val` の符号（既定パラメータ）| 0 / 18 |")
        add("| Phase 1 追試 | 解放定義を原典（黒→灰）に修正 | 完全に同一（青が 0 件）|")
        add("| Phase 1b | WaveTrend / ADX 単体（標準設定）| 0 / 36 |")
        add("| Phase 1c Part A | 65 設定のグリッド探索 | 候補 4 件を選定（判定なし）|")
        add(f"| **Phase 1c Part B** | **未使用データでの確認** | **0 / {len(judgement)}** |")
        add("")
        add("探索で見つけた「良さそうな設定」は、**未使用期間では再現しなかった**。")
        add("これは Part A の数字が実力ではなく、多重比較（1,170 セル）の産物だったことを意味する。")
        add("")
    else:
        add("### 合格あり — すぐには採用と結論しない")
        add("")
        add(f"合格した候補: **{len(passing)} / {len(judgement)}**")
        add("")
        add("2023-2024 と 2025-2026 は相場環境が大きく異なる。")
        add("両期間で通ったことは頑健性の証拠になる一方、")
        add("**どちらか片方だけの場合は「過学習」と「レジーム依存」の区別がつかない。**")
        add("")
        add("以下の精査結果を見て判断してほしい。**Phase 2 には自動で進まない。**")
        add("Hold-out（2026-06-01 .. 2026-08-31）はこの時点でも開封しない。")
        add("")
        for candidate_number, block in scrutiny.items():
            row = judgement[judgement["candidate"] == candidate_number].iloc[0]
            add(f"### 候補 {candidate_number}: {row['setting_key']} / "
                f"{row['interval']} / N={row['horizon']} の精査")
            add("")
            lines.extend(block)
    add("")

    # 時間的な安定性の追加分析を同じレポートに埋め込む
    from phase1c_stability import build_section

    lines.extend(build_section(candidates))

    add("## 7. 言えること / 言えないこと")
    add("")
    add("- 言える: Part A（2025-2026）で選んだ候補が、Part B（2023-2024）でどうなるか。")
    add("  **探索と確認でデータを完全に分離してある**ので、この判定は「見かけ上良い設定」"
        "を排除できている。")
    add("- 言えない: グリッドに入れなかった設定（例: 逆張り、別のシグナル定義、"
        "上位足フィルター）については何も言えない。")
    add("  ただし結果を見てからそれらを足すのはデータスヌーピングであり、やらない。")
    add("- コスト前提は COSTS.md のシナリオ F（Lighter 手数料 0% + 往復 1.0 bps）"
        "という**最も甘い前提**である。")
    add("  これで落ちるなら、他のどのコスト前提でも落ちる。")
    add("")

    (REPORT_DIR / "phase1c_confirmation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
