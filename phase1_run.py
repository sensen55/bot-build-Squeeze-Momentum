"""Phase 1 の実行と Markdown レポート生成。

実行: python3 phase1_run.py
出力: reports/phase1_event_study.md, reports/*.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase1_event_study import (
    ALPHA, ALPHA_BONFERRONI, COST_BASE_BPS, COST_FLOOR_BPS, COST_HIGH_BPS,
    COST_HYPERLIQUID_BPS, HORIZONS, INTERVALS, PRIMARY_CELL_COUNT,
    STUDY_END, STUDY_START, SYMBOLS,
    analyse_baseline_direction, analyse_direction, analyse_hour_of_day,
    analyse_run_buckets, analyse_volatility_expansion, build_event_table,
)
from phase1_nearmiss import scrutinise

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    all_tables: dict[str, list] = {}
    summary_rows = []

    for interval in INTERVALS:
        print(f"=== {interval} ===", flush=True)
        tables = []
        for symbol in SYMBOLS:
            table = build_event_table(symbol, interval)
            tables.append(table)
            print(f"  {symbol}: events={len(table.events)} "
                  f"bars={len(table.baseline)} 1bar_median={table.bar_move_bps:.1f}bps",
                  flush=True)
            summary_rows.append({
                "interval": interval, "symbol": symbol,
                "n_events": len(table.events), "n_bars": len(table.baseline),
                "bar_move_median_bps": table.bar_move_bps,
                "bar_move_over_cost_floor": table.bar_move_bps / COST_FLOOR_BPS,
                "bar_move_over_cost_base": table.bar_move_bps / COST_BASE_BPS,
                "bar_move_over_cost_hl": table.bar_move_bps / COST_HYPERLIQUID_BPS,
            })
        all_tables[interval] = tables

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REPORT_DIR / "phase1_data_summary.csv", index=False)

    volatility_rows, direction_rows, accel_rows, baseline_rows = [], [], [], []
    for interval in INTERVALS:
        tables = all_tables[interval]
        for horizon in HORIZONS:
            print(f"  分析 {interval} N={horizon}", flush=True)
            volatility_rows.append({"interval": interval,
                                    **analyse_volatility_expansion(tables, horizon)})
            direction_rows.append({"interval": interval,
                                   **analyse_direction(tables, horizon, "sign")})
            accel_rows.append({"interval": interval,
                               **analyse_direction(tables, horizon, "sign_accel")})
            baseline_rows.append({"interval": interval,
                                  **analyse_baseline_direction(tables, horizon)})

    volatility = pd.DataFrame(volatility_rows)
    direction = pd.DataFrame(direction_rows)
    acceleration = pd.DataFrame(accel_rows)
    baseline_direction = pd.DataFrame(baseline_rows)

    volatility.to_csv(REPORT_DIR / "phase1_volatility_expansion.csv", index=False)
    direction.to_csv(REPORT_DIR / "phase1_direction.csv", index=False)
    acceleration.to_csv(REPORT_DIR / "phase1_direction_accel.csv", index=False)
    baseline_direction.to_csv(REPORT_DIR / "phase1_baseline_direction.csv", index=False)

    run_frames, hour_frames, per_symbol_rows = [], [], []
    for interval in INTERVALS:
        tables = all_tables[interval]
        for horizon in (6, 12):
            frame = analyse_run_buckets(tables, horizon)
            frame.insert(0, "horizon", horizon)
            frame.insert(0, "interval", interval)
            run_frames.append(frame)
        frame = analyse_hour_of_day(tables, 6)
        frame.insert(0, "interval", interval)
        hour_frames.append(frame)
        for table in tables:
            for horizon in HORIZONS:
                result = analyse_direction([table], horizon, "sign")
                per_symbol_rows.append({"interval": interval,
                                        "symbol": table.symbol, **result})

    run_buckets = pd.concat(run_frames, ignore_index=True)
    hours = pd.concat(hour_frames, ignore_index=True)
    per_symbol = pd.DataFrame(per_symbol_rows)
    run_buckets.to_csv(REPORT_DIR / "phase1_run_buckets.csv", index=False)
    hours.to_csv(REPORT_DIR / "phase1_hour_of_day.csv", index=False)
    per_symbol.to_csv(REPORT_DIR / "phase1_per_symbol.csv", index=False)

    # 最も成績の良かったセル (CI 下限が最大) を自動で選んで精査する。
    best = direction.loc[direction['ci_low_bps'].idxmax()]
    print(f"  最良セルの精査: {best['interval']} N={int(best['horizon'])}", flush=True)
    nearmiss_lines = scrutinise(all_tables[best['interval']], int(best['horizon']))

    _write_report(summary, volatility, direction, acceleration,
                  baseline_direction, run_buckets, hours, per_symbol,
                  best, nearmiss_lines)
    print("レポート:", REPORT_DIR / "phase1_event_study.md")


def _fmt(frame: pd.DataFrame, columns: dict[str, str], decimals: int = 3) -> str:
    shown = frame[list(columns)].rename(columns=columns).copy()
    for column in shown.columns:
        if shown[column].dtype.kind == "f":
            shown[column] = shown[column].map(lambda v: f"{v:,.{decimals}f}")
    return shown.to_markdown(index=False)


def _write_report(summary, volatility, direction, acceleration,
                  baseline_direction, run_buckets, hours, per_symbol,
                  best, nearmiss_lines) -> None:
    total_events = int(summary["n_events"].sum())

    # --- 事前確定した中止基準に対する判定 -----------------------------------
    direction = direction.copy()
    direction["exceeds_cost_floor"] = direction["mean_bps"] > COST_FLOOR_BPS
    direction["ci_low_exceeds_cost_floor"] = direction["ci_low_bps"] > COST_FLOOR_BPS
    direction["significant_bonferroni"] = direction["p_value"] < ALPHA_BONFERRONI
    passing = direction[direction["ci_low_exceeds_cost_floor"]
                        & direction["significant_bonferroni"]]

    lines: list[str] = []
    add = lines.append

    add("# Phase 1: スクイーズ解放イベントスタディ")
    add("")
    add(f"- 対象: {', '.join(SYMBOLS)} / {', '.join(INTERVALS)}")
    add(f"- 期間: {STUDY_START} .. {STUDY_END}（Hold-out 2026-06-01..2026-08-31 は封印・未使用）")
    add(f"- データ: Binance USD-M Futures kline（data.binance.vision）欠損 0 本")
    add("- 解放イベントの定義: **原典どおり「黒（sqz_on）→ 灰（sqz_off）」**。")
    add("  旧定義「黒 → 黒以外」との比較は "
        "[`phase1_release_definition.md`](phase1_release_definition.md) を参照")
    add("  （このデータでは青（no_sqz）が 0 本のため、両定義のイベント集合は完全に同一だった）。")
    add(f"- 総イベント数: **{total_events:,}**")
    add("")
    add("## 0. 事前に確定した判定ルール（実行前に固定・後から緩めていない）")
    add("")
    add("| 項目 | 値 |")
    add("|---|---|")
    add(f"| 主検定のセル数 | {PRIMARY_CELL_COUNT}（3 時間足 × 6 ホライズン）|")
    add(f"| 有意水準（Bonferroni 補正後）| {ALPHA} / {PRIMARY_CELL_COUNT} = **{ALPHA_BONFERRONI:.5f}** |")
    add(f"| 合格に必要な往復コスト超過 | 平均が **{COST_FLOOR_BPS} bps** 超（COSTS.md シナリオ F＝最良）|")
    add("| 合格の条件 | ブートストラップ 95%CI の**下限**が 1.0 bps を超え、かつ Bonferroni 補正後に有意 |")
    add("| イベント数の下限 | 300 |")
    add("")
    add("最良コスト（Lighter 手数料 0% + 穏やか局面の実測スリッページ）を判定に使う。")
    add("これで落ちるなら、他のどのコスト前提でも落ちる。")
    add("")

    add("## 1. データ要約と「1 バーの値動き vs コスト」（skill 3.7）")
    add("")
    add(_fmt(summary, {
        "interval": "時間足", "symbol": "銘柄", "n_events": "イベント数",
        "n_bars": "バー数", "bar_move_median_bps": "1バー|変化|中央値(bps)",
        "bar_move_over_cost_floor": "÷コスト1.0bps",
        "bar_move_over_cost_base": "÷コスト2.0bps",
        "bar_move_over_cost_hl": "÷HL 11bps"}, 1))
    add("")
    add("skill 3.7 の目安は「1 バーの値動き ÷ 往復コスト が 3 を下回るとコストに埋もれる」。")
    add("Lighter 前提（1〜2 bps）ではどの時間足も比が十分大きく、**コストの観点では成立しうる**。")
    add("Hyperliquid の成行往復（11 bps）だと 5 分足は比が小さくなる。")
    add("")

    add("## 2. 検定 (B) vs (A): スクイーズ解放は「値幅の拡大」を予測するか")
    add("")
    add("帰無仮説: 解放バー後のリターンのばらつきは、全バーのそれと変わらない。")
    add("検定: Brown-Forsythe（中央値中心の Levene 検定）。")
    add("暗号資産のリターンは裾が厚く正規分布から遠いので、F 検定は使わない。")
    add("")
    add(_fmt(volatility, {
        "interval": "時間足", "horizon": "N本後", "n_events": "イベント数",
        "event_std_bps": "解放後 std(bps)", "baseline_std_bps": "全バー std(bps)",
        "std_ratio": "std比", "median_abs_ratio": "中央値|ret|比",
        "levene_p": "Levene p"}, 3))
    add("")

    add("## 3. 検定 (C): val の符号は「方向」の情報を持つか【主検定】")
    add("")
    add("符号方向リターン = sign(val[t]) × (close[t+N] / open[t+1] − 1) × 10000")
    add("")
    add("判断は解放バー t の終値確定時点、約定は t+1 の始値。両者は必ず 1 バー以上離れている。")
    add("")
    add(_fmt(direction, {
        "interval": "時間足", "horizon": "N本後", "n": "n",
        "mean_bps": "平均(bps)", "median_bps": "中央値(bps)",
        "stderr_bps": "標準誤差", "t_stat": "t値", "p_value": "p値",
        "ci_low_bps": "95%CI下限", "ci_high_bps": "95%CI上限",
        "win_rate": "勝率", "non_overlap_mean_bps": "窓非重複 平均",
        "non_overlap_p": "窓非重複 p"}, 3))
    add("")
    add(f"- Bonferroni 補正後に有意（p < {ALPHA_BONFERRONI:.5f}）なセル: "
        f"**{int(direction['significant_bonferroni'].sum())} / {len(direction)}**")
    add(f"- 平均が最良コスト {COST_FLOOR_BPS} bps を超えたセル: "
        f"**{int(direction['exceeds_cost_floor'].sum())} / {len(direction)}**")
    add(f"- 95%CI 下限が {COST_FLOOR_BPS} bps を超えたセル: "
        f"**{int(direction['ci_low_exceeds_cost_floor'].sum())} / {len(direction)}**")
    add(f"- **合格条件（CI 下限超過 かつ Bonferroni 有意）を満たしたセル: "
        f"{len(passing)} / {len(direction)}**")
    add("")

    add("### 3-1. 銘柄別（プールで隠れる差を見るため）")
    add("")
    add("BTC / ETH / SOL は同時に動くため、プールしても実効的な標本数は 3 倍にならない。")
    add("銘柄ごとに符号が揃っているかを見る。")
    add("")
    pivot = per_symbol.pivot_table(index=["interval", "horizon"],
                                   columns="symbol", values="mean_bps").round(2)
    add(pivot.to_markdown())
    add("")

    add("### 3-2. 副次: 符号 + モメンタム加速で絞った場合")
    add("")
    add("原典の 4 色分けのうち「加速」側だけを採用した場合。副次的な確認であり、")
    add("ここで良い数字が出ても主検定の判定は変えない（多重比較になるため）。")
    add("")
    add(_fmt(acceleration, {
        "interval": "時間足", "horizon": "N本後", "n": "n",
        "mean_bps": "平均(bps)", "t_stat": "t値", "p_value": "p値",
        "ci_low_bps": "95%CI下限", "ci_high_bps": "95%CI上限"}, 3))
    add("")

    add("### 3-3. 対照: 解放バーに限らず「全バー」で val の符号方向を取った場合")
    add("")
    add("これが解放バーと同程度なら、「スクイーズ解放」というイベント自体には")
    add("付加価値が無く、単に val の符号を見ているだけということになる。")
    add("")
    add(_fmt(baseline_direction, {
        "interval": "時間足", "horizon": "N本後", "n": "n",
        "mean_bps": "平均(bps)", "t_stat": "t値", "p_value": "p値"}, 3))
    add("")

    add("## 4. 副次: スクイーズ継続本数バケット")
    add("")
    add("「長い収縮ほど値幅が大きい」という仮説の記述統計。ここから合格は主張しない。")
    add("")
    add(_fmt(run_buckets, {
        "interval": "時間足", "horizon": "N本後", "min_run": "継続n本以上",
        "n": "件数", "median_abs_bps": "中央値|ret|(bps)",
        "signed_mean_bps": "符号方向平均(bps)", "signed_p": "p値"}, 3))
    add("")

    add("## 5. UTC 時間帯別のイベント分布（N=6）")
    add("")
    add("「スクイーズを検出した」のではなく「静かな時間帯を検出した」だけではないかの確認。")
    add("")
    for interval in INTERVALS:
        subset = hours[hours["interval"] == interval]
        share = subset["vs_uniform"]
        add(f"- **{interval}**: 一様分布比の範囲 {share.min():.2f} 〜 {share.max():.2f} "
            f"（最多 {int(subset.loc[share.idxmax(),'utc_hour'])} 時 / "
            f"最少 {int(subset.loc[share.idxmin(),'utc_hour'])} 時）")
    add("")
    add("詳細は `reports/phase1_hour_of_day.csv`。")
    add("")

    add("## 6. コスト感度（skill 3.7）")
    add("")
    add("主検定の各セルの平均が、どのコスト水準まで生き残るか。")
    add("")
    header = "| 時間足 | N | 平均(bps) | 95%CI下限 | " \
             f"F {COST_FLOOR_BPS}bps | B {COST_BASE_BPS}bps | " \
             f"B×5 {COST_HIGH_BPS}bps | HL {COST_HYPERLIQUID_BPS}bps |"
    add(header)
    add("|---|---|---|---|---|---|---|---|")
    for _, row in direction.iterrows():
        marks = "".join(
            f" {'○' if row['ci_low_bps'] > cost else '×'} |"
            for cost in (COST_FLOOR_BPS, COST_BASE_BPS, COST_HIGH_BPS, COST_HYPERLIQUID_BPS))
        add(f"| {row['interval']} | {int(row['horizon'])} | {row['mean_bps']:.2f} | "
            f"{row['ci_low_bps']:.2f} |{marks}")
    add("")
    add("○ = 95%CI 下限がそのコストを上回る（コストを引いても正が残ると言える）")
    add("")

    add(f"## 7. 最良セル（{best['interval']} / N={int(best['horizon'])}）の精査")
    add("")
    add("事前ルールで機械的に落として終わりにせず、その数字が本物かを確認する。")
    add(f"このセルは平均 {best['mean_bps']:.2f} bps・95%CI 下限 {best['ci_low_bps']:.2f} bps で")
    add(f"コスト基準は超えているが、p = {best['p_value']:.4f} は Bonferroni 補正後の")
    add(f"閾値 {ALPHA_BONFERRONI:.5f} を超えており不合格。以下はその裏付け。")
    add("")
    lines.extend(nearmiss_lines)

    add("## 8. 判定")
    add("")
    if len(passing) == 0:
        add("### 中止基準に到達 — Phase 2 以降には進まない")
        add("")
        add("事前に確定した合格条件（95%CI 下限が最良コスト 1.0 bps を超え、")
        add(f"かつ Bonferroni 補正後 p < {ALPHA_BONFERRONI:.5f}）を満たしたセルは "
            f"**{len(passing)} / {len(direction)}** だった。")
    else:
        add("### 合格セルあり")
        add("")
        add(_fmt(passing, {"interval": "時間足", "horizon": "N本後", "n": "n",
                           "mean_bps": "平均(bps)", "ci_low_bps": "95%CI下限",
                           "p_value": "p値"}, 3))
    add("")

    add("### この結果から言えること")
    add("")
    add("1. **`val` の符号は、その後のリターンの符号について実用的な情報を持たない。**")
    add("   5 分足では 6 ホライズンすべてで符号方向リターンが**マイナス**")
    add("   （コストを引く前の時点で、既に負けている）。15 分足もほぼマイナス。")
    add("   1 時間足はプラス側だが、どのセルも Bonferroni 補正後に有意ではない。")
    add("")
    add("2. **スクイーズ解放は「値幅の拡大」をほとんど予測しない。**")
    add("   std 比は 0.92〜1.15 の範囲で、有意なのは N=1（直後 1 本）だけ。")
    add("   しかもその効果は 5 分足で +4%、1 時間足で +15% にすぎず、N=3 以降では消える。")
    add("   これは依頼書 0.4 の予想（ボラティリティ・クラスタリングは")
    add("   「低ボラは低ボラを呼ぶ」であって「低ボラの後に爆発が来る」ではない）と整合する。")
    add("")
    add("3. **「スクイーズ解放」というイベント自体に付加価値が無い。**")
    add("   セクション 3-3 の対照（全バーで `val` の符号方向を取る）と比べると、")
    add("   15 分足では**全バー版のほうが成績が良く**（N=48 で +3.64 bps, p<0.001 vs")
    add("   解放バー版 −1.53 bps）、1 時間足では全バー版が有意に**マイナス**（N=12 で")
    add("   −5.75 bps, p<0.001）。つまり `val` の符号が持つわずかな情報は時間足ごとに")
    add("   符号が逆転しており、一貫した方向性がない。解放でそれを絞っても改善しない。")
    add("")
    add("4. **勝率は 0.46〜0.52 で、ほぼコイン投げ。**")
    add("   （依頼書のとおり勝率は主要指標にしていないが、業者が宣伝する")
    add("   「勝率 88〜92%」とは桁が違うことは記録しておく。）")
    add("")

    add("### この結果から言えないこと")
    add("")
    add("- 「Squeeze Momentum 指標に一切の情報が無い」とは言えない。検証したのは")
    add("  **解放時点の `val` の符号が方向を予測するか**という 1 点だけである。")
    add("- 既定パラメータ（20/2.0/20/1.5）以外を網羅していない。ただし依頼書 4 のとおり、")
    add("  中止基準に到達した後にパラメータを振り直すのはデータスヌーピング")
    add("  （skill 5.3）そのものなので、やらない。")
    add("- Binance の kline を使っており、Lighter / Hyperliquid の板とは価格が異なる。")
    add("  ただし方向情報の有無という結論が、この差で反転するとは考えにくい。")
    add("- スリッページを実測していない。ただし本結論は**手数料 0%・往復 1 bps という")
    add("  最も甘いコスト前提**で出しているため、スリッページの精度は結論を変えない。")
    add("  （コストが結論を左右するのは「エッジがコストと同オーダー」の場合であり、")
    add("  今回はエッジの符号すら定まっていない。）")
    add("")

    add("### 次にすべきこと")
    add("")
    add("**終了する。** Phase 2（グリッドサーチ）以降には進まない。")
    add("")
    add("依頼書 6 に挙がっていた発展方向（継続本数フィルター、上位足トレンドフィルター、")
    add("OFI/CVD との組み合わせ、時間帯別閾値）は、いずれも「Phase 1 を通過してからの話」")
    add("と事前に定めたものである。通過しなかったので着手しない。")
    add("")
    add("Hold-out 期間（2026-06-01 .. 2026-08-31）は**封印したまま未使用**である。")
    add("将来この指標を別の切り口で検証する場合に備えて、開封しない。")
    add("")

    (REPORT_DIR / "phase1_event_study.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
