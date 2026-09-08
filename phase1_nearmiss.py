"""主検定で最も成績の良かったセルを精査する。

事前ルールで機械的に落として終わりにせず、
「その数字が本物ではないこと」を実際に確認するための追試。

見るのは 4 点:
  1. 1 銘柄に依存していないか（1 つ抜くと消えるなら偶然）
  2. 期間を分けても安定しているか（一時期の産物ではないか）
  3. 隣接ホライズンと連続しているか（1 点だけ尖るのはノイズの特徴）
  4. 窓の重なりで実効標本がどれだけ減っているか
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from phase1_event_study import (
    EXTRA_HORIZONS, HORIZONS, analyse_direction, select_non_overlapping,
)

PERIOD_EDGES = ["2025-01-01", "2025-06-01", "2025-11-01", "2026-06-01"]


def scrutinise(tables: list, horizon: int) -> list[str]:
    column = f"ret_{horizon}_bps"
    lines: list[str] = []
    add = lines.append

    add("**(a) 銘柄別内訳** — プールの平均が 1 銘柄に依存していないか")
    add("")
    add("| 銘柄 | n | 平均(bps) | 中央値(bps) | t値 | p値 | 勝率 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for table in tables:
        events = table.events.dropna(subset=[column])
        signed = (events["val_sign"] * events[column]).to_numpy()
        result = stats.ttest_1samp(signed, 0.0)
        add(f"| {table.symbol} | {len(signed)} | {signed.mean():.2f} | "
            f"{np.median(signed):.2f} | {result.statistic:.2f} | "
            f"{result.pvalue:.4f} | {(signed > 0).mean():.3f} |")
    add("")

    add("**(b) 1 銘柄を除外したときの変化**")
    add("")
    add("| 対象 | n | 平均(bps) | p値 | 95%CI |")
    add("|---|---:|---:|---:|---|")
    for excluded in [None] + [t.symbol for t in tables]:
        subset = [t for t in tables if t.symbol != excluded]
        result = analyse_direction(subset, horizon, "sign")
        label = "全銘柄" if excluded is None else f"{excluded} 除外"
        add(f"| {label} | {result['n']} | {result['mean_bps']:.2f} | "
            f"{result['p_value']:.4f} | "
            f"[{result['ci_low_bps']:.2f}, {result['ci_high_bps']:.2f}] |")
    add("")

    add("**(c) 期間を 3 分割した安定性**")
    add("")
    add("| 期間 | n | 平均(bps) | p値 |")
    add("|---|---:|---:|---:|")
    combined = pd.concat([t.events for t in tables], ignore_index=True).dropna(subset=[column])
    combined["signed"] = combined["val_sign"] * combined[column]
    edges = pd.to_datetime(PERIOD_EDGES, utc=True)
    for start, end in zip(edges[:-1], edges[1:]):
        window = combined[(combined["time"] >= start) & (combined["time"] < end)]
        signed = window["signed"].to_numpy()
        result = stats.ttest_1samp(signed, 0.0)
        add(f"| {start.date()} .. {end.date()} | {len(signed)} | "
            f"{signed.mean():.2f} | {result.pvalue:.4f} |")
    add("")

    add("**(d) 隣接ホライズンとの連続性** — 本物の効果なら滑らかに変化するはず")
    add("")
    add("| N本後 | 平均(bps) | p値 | 95%CI |")
    add("|---:|---:|---:|---|")
    for h in sorted(set(HORIZONS + EXTRA_HORIZONS)):
        result = analyse_direction(tables, h, "sign")
        mark = " ←最良セル" if h == horizon else ""
        add(f"| {h} | {result['mean_bps']:.2f} | {result['p_value']:.4f} | "
            f"[{result['ci_low_bps']:.2f}, {result['ci_high_bps']:.2f}]{mark} |")
    add("")

    add("**(e) 窓の重なりによる実効標本の減少**")
    add("")
    for table in tables:
        events = table.events.dropna(subset=[column])
        index_array = events["index"].to_numpy()
        keep = select_non_overlapping(index_array, horizon)
        add(f"- {table.symbol}: 全 {len(index_array)} 件 → 窓非重複 {len(keep)} 件 "
            f"（実効標本は約 {len(keep) / len(index_array):.0%}）")
    add("")
    return lines


if __name__ == "__main__":
    from phase1_event_study import SYMBOLS, build_event_table

    tables = [build_event_table(symbol, "1h") for symbol in SYMBOLS]
    print("\n".join(scrutinise(tables, 48)))
