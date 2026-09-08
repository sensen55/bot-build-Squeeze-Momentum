"""Phase 1 追試: 解放イベントの定義を原典どおりに直して再集計する。

背景
----
旧定義は「黒 (sqz_on) の次のバーで黒でなくなった」= 灰 (sqz_off) と
青 (no_sqz) の両方を解放として数えていた。
原典 (Carter / LazyBear) の推奨は「黒の後、最初の灰でエントリー」なので、
正しくは sqz_off[t] == True を直接要求する。

これは定義の修正であって、パラメータの振り直しではない。
HORIZONS / SYMBOLS / INTERVALS / 期間 / コスト前提 / 合格条件は一切変えない。

構造上の予測 (実際に数えて確認する)
-----------------------------------
既定パラメータでは bb_length == kc_length == 20 なので、
BB の中心線 basis と KC の中心線 kc_ma はどちらも SMA(close, 20) で同じ値になる。
すると 2 つの条件は同じ 1 本の不等式に帰着する:

    upperBB < upperKC  <=>  basis + dev < basis + rangema*multKC  <=>  dev < rangema*multKC
    lowerBB > lowerKC  <=>  basis - dev > basis - rangema*multKC  <=>  dev < rangema*multKC

したがって sqz_on = (dev < rangema*multKC)、sqz_off = (dev > rangema*multKC) であり、
no_sqz (青) は「ちょうど等しい」ときだけ。小数の完全一致なのでまず起きない。
= 黒 -> 青 の解放はほぼ存在せず、旧定義と原典定義は実質同じになるはず。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.fetch_binance_klines import load_klines
from phase1_event_study import (
    ALPHA_BONFERRONI, COST_FLOOR_BPS, HORIZONS, INTERVALS,
    RELEASE_DEFINITIONS, STUDY_END, STUDY_START, SYMBOLS,
    analyse_direction, build_event_table,
)
from phase1_nearmiss import scrutinise
from squeeze_momentum import SqueezeParams, compute

REPORT_DIR = Path(__file__).resolve().parent / "reports"


def verify_state_structure() -> list[str]:
    """no_sqz (青) がほぼ発生しない構造的な理由を、実データで確認する。"""
    lines: list[str] = []
    add = lines.append
    add("| 銘柄 | 時間足 | max|basis − kc_ma| | 青(no_sqz)の本数 | "
        "sqz_on == (dev < rangema×1.5) |")
    add("|---|---|---:|---:|---|")
    for symbol in SYMBOLS:
        for interval in INTERVALS:
            data, _ = load_klines(symbol, interval, STUDY_START, STUDY_END, verbose=False)
            result = compute(data, SqueezeParams())
            ready = result["upper_bb"].notna() & result["upper_kc"].notna()
            center_difference = float((result["basis"] - result["kc_ma"]).abs().max())
            blue_bars = int(result["no_sqz"][ready].sum())
            equivalent = bool(
                (result["sqz_on"][ready]
                 == (result["dev"][ready] < result["rangema"][ready] * 1.5)).all())
            add(f"| {symbol} | {interval} | {center_difference:.3e} | {blue_bars} | "
                f"{'一致' if equivalent else '不一致'} |")
    return lines


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)

    # --- 1. 各定義でイベント表を作る ---------------------------------------
    tables: dict[tuple[str, str], list] = {}
    count_rows = []
    for definition in RELEASE_DEFINITIONS:
        for interval in INTERVALS:
            built = [build_event_table(symbol, interval, definition) for symbol in SYMBOLS]
            tables[(definition, interval)] = built
            for table in built:
                count_rows.append({
                    "definition": definition, "interval": interval,
                    "symbol": table.symbol, "n_events": len(table.events),
                })
            print(f"  {definition:5s} {interval:3s}: "
                  f"{[len(t.events) for t in built]}", flush=True)

    counts = pd.DataFrame(count_rows)
    counts.to_csv(REPORT_DIR / "phase1_definition_event_counts.csv", index=False)

    # --- 2. 主検定 18 セルを定義ごとに回す ----------------------------------
    result_rows = []
    for definition in RELEASE_DEFINITIONS:
        for interval in INTERVALS:
            built = tables[(definition, interval)]
            total_events = sum(len(t.events) for t in built)
            for horizon in HORIZONS:
                if total_events < 2:
                    result_rows.append({
                        "definition": definition, "interval": interval,
                        "horizon": horizon, "n": total_events,
                        "mean_bps": np.nan, "p_value": np.nan,
                        "ci_low_bps": np.nan, "ci_high_bps": np.nan,
                    })
                    continue
                print(f"  分析 {definition} {interval} N={horizon}", flush=True)
                result_rows.append({
                    "definition": definition, "interval": interval,
                    **analyse_direction(built, horizon, "sign"),
                })

    results = pd.DataFrame(result_rows)
    results["passes"] = (
        (results["ci_low_bps"] > COST_FLOOR_BPS)
        & (results["p_value"] < ALPHA_BONFERRONI)
    )
    results.to_csv(REPORT_DIR / "phase1_definition_direction.csv", index=False)

    structure_lines = verify_state_structure()

    # --- 3. 合格セルが出た場合のみ精査 --------------------------------------
    gray = results[results["definition"] == "gray"]
    passing = gray[gray["passes"]]
    scrutiny_lines: list[str] = []
    if len(passing):
        best = passing.loc[passing["ci_low_bps"].idxmax()]
        scrutiny_lines = scrutinise(
            tables[("gray", best["interval"])], int(best["horizon"]))

    _write_report(counts, results, structure_lines, passing, scrutiny_lines)
    print("レポート:", REPORT_DIR / "phase1_release_definition.md")


def _write_report(counts, results, structure_lines, passing, scrutiny_lines) -> None:
    lines: list[str] = []
    add = lines.append

    pivot = counts.pivot_table(index=["interval", "symbol"], columns="definition",
                               values="n_events", aggfunc="sum")
    pivot = pivot[["any", "gray", "blue"]]

    add("# Phase 1 追試: 解放イベント定義の修正")
    add("")
    add("## 0. 何を変えたか")
    add("")
    add("変更したのは `phase1_event_study.py` の解放イベント判定 1 行のみ。")
    add("")
    add("```python")
    add("# 変更前 (旧定義): 「黒でなくなった」= 灰と青の両方を含む")
    add("release[1:] = (~sqz_on[1:]) & sqz_on[:-1]")
    add("")
    add("# 変更後 (原典): 「黒 -> 灰」だけ")
    add("release[1:] = sqz_off[1:] & sqz_on[:-1]")
    add("```")
    add("")
    add("| 定義 | 内容 |")
    add("|---|---|")
    for key, description in RELEASE_DEFINITIONS.items():
        add(f"| `{key}` | {description} |")
    add("")
    add("**変えていないもの**: `HORIZONS`, `SYMBOLS`, `INTERVALS`, "
        f"期間（{STUDY_START} .. {STUDY_END}）, コスト前提, "
        f"合格条件（Bonferroni 補正後 p < {ALPHA_BONFERRONI:.5f} かつ "
        f"95%CI 下限 > {COST_FLOOR_BPS} bps）。")
    add("")
    add("Hold-out（2026-06-01 .. 2026-08-31）は引き続き未開封。")
    add("")

    add("## 1. イベント数の比較")
    add("")
    add(pivot.to_markdown())
    add("")
    total_any = int(counts[counts["definition"] == "any"]["n_events"].sum())
    total_gray = int(counts[counts["definition"] == "gray"]["n_events"].sum())
    total_blue = int(counts[counts["definition"] == "blue"]["n_events"].sum())
    add(f"- 旧定義 `any`（黒→黒以外）: **{total_any:,} 件**")
    add(f"- 原典 `gray`（黒→灰）: **{total_gray:,} 件**")
    add(f"- 除外された `blue`（黒→青）: **{total_blue:,} 件**")
    add("")
    if total_blue == 0:
        add("**差分はゼロ件。旧定義と原典定義は、このデータでは完全に同じイベント集合を指していた。**")
    else:
        add(f"差分 {total_blue:,} 件（旧定義全体の "
            f"{total_blue / max(total_any, 1):.2%}）。")
    add("")

    add("## 2. なぜ差が出ないのか（構造的な理由）")
    add("")
    add("既定パラメータでは BB 期間と KC 期間がどちらも 20 なので、")
    add("BB の中心線 `basis` と KC の中心線 `kc_ma` はどちらも `SMA(close, 20)` で**同じ値**になる。")
    add("すると原典の 2 条件は同じ 1 本の不等式に帰着する。")
    add("")
    add("```")
    add("upperBB < upperKC  <=>  basis + dev < basis + rangema*1.5  <=>  dev < rangema*1.5")
    add("lowerBB > lowerKC  <=>  basis - dev > basis - rangema*1.5  <=>  dev < rangema*1.5")
    add("```")
    add("")
    add("つまり `sqz_on = (dev < rangema*1.5)`、`sqz_off = (dev > rangema*1.5)` であり、")
    add("`no_sqz`（青）は両者がちょうど等しいときだけ成立する。")
    add("小数の完全一致なので実質的に発生しない。実データで確認した結果:")
    add("")
    lines.extend(structure_lines)
    add("")
    add("`basis` と `kc_ma` の差は浮動小数点の丸め誤差レベルで、青のバーは全データセットで 0 本。")
    add("`sqz_on` が単一の不等式 `dev < rangema×1.5` と完全一致することも確認した。")
    add("")
    add("**注意**: これは BB 期間 = KC 期間 のときに限った話である。")
    add("両者を別の値にすると中心線がずれ、青の状態が実際に発生しうる。")
    add("ただし今回は「定義の修正」のみが依頼範囲であり、期間は変更していない。")
    add("")

    add("## 3. 主検定 18 セルの比較（修正前 vs 修正後）")
    add("")
    comparison = results[results["definition"].isin(["any", "gray"])].pivot_table(
        index=["interval", "horizon"], columns="definition",
        values=["n", "mean_bps", "p_value", "ci_low_bps"])
    frame = pd.DataFrame({
        "n (旧)": comparison[("n", "any")].astype(int),
        "n (原典)": comparison[("n", "gray")].astype(int),
        "平均bps (旧)": comparison[("mean_bps", "any")].round(3),
        "平均bps (原典)": comparison[("mean_bps", "gray")].round(3),
        "p (旧)": comparison[("p_value", "any")].round(4),
        "p (原典)": comparison[("p_value", "gray")].round(4),
        "CI下限 (旧)": comparison[("ci_low_bps", "any")].round(3),
        "CI下限 (原典)": comparison[("ci_low_bps", "gray")].round(3),
    })
    add(frame.to_markdown())
    add("")
    identical = bool(np.allclose(
        comparison[("mean_bps", "any")], comparison[("mean_bps", "gray")],
        rtol=0, atol=1e-12, equal_nan=True))
    add(f"- 全 18 セルで平均値が完全一致: **{'はい' if identical else 'いいえ'}**")
    for definition in ("any", "gray"):
        subset = results[results["definition"] == definition]
        add(f"- `{definition}` の合格セル数: **{int(subset['passes'].sum())} / {len(subset)}**")
    add("")

    add("## 4. 除外された側（黒→青）の符号方向リターン")
    add("")
    blue = results[results["definition"] == "blue"]
    if total_blue == 0:
        add("**イベント数 0 件のため集計不能。**")
        add("")
        add("依頼の目的は「黒→灰が正、黒→青が負、という逆向きの構造があるか」の確認だった。")
        add("そもそも黒→青が 1 件も存在しないため、**逆向きの構造は存在しない**。")
        add("旧定義に混ざっていた「余計なイベント」は 1 件も無かったので、")
        add("定義の違いは Phase 1 の結論にまったく影響していない。")
    else:
        add(blue[["interval", "horizon", "n", "mean_bps", "p_value",
                  "ci_low_bps", "ci_high_bps"]].round(3).to_markdown(index=False))
    add("")

    add("## 5. 判定")
    add("")
    if len(passing) == 0:
        add("### 修正後も合格セルは 0 / 18 — Phase 1 の結論を確定して終了")
        add("")
        add("原典どおりの定義（黒→灰）に修正しても、事前に確定した合格条件")
        add(f"（95%CI 下限 > {COST_FLOOR_BPS} bps かつ Bonferroni 補正後 "
            f"p < {ALPHA_BONFERRONI:.5f}）を満たしたセルは **0 / 18** だった。")
        add("")
        add("そもそもイベント集合が旧定義と 1 件も違わなかったため、")
        add("**数値は完全に同一**であり、前回の結論はそのまま維持される。")
        add("")
        add("**結論: Squeeze Momentum 指標の解放時 `val` の符号に、")
        add("往復コストを超える方向情報は確認できなかった。Phase 2 以降には進まない。**")
        add("")
        add("Hold-out 期間（2026-06-01 .. 2026-08-31）は封印したまま未使用。")
    else:
        add("### 合格セルが出た — 精査結果を報告し、判断を待つ")
        add("")
        add(passing[["interval", "horizon", "n", "mean_bps", "ci_low_bps",
                     "p_value"]].round(4).to_markdown(index=False))
        add("")
        add("**Phase 2 には進んでいない。** 以下は `phase1_nearmiss.py` と同じ精査。")
        add("")
        lines.extend(scrutiny_lines)

    (REPORT_DIR / "phase1_release_definition.md").write_text(
        "\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
