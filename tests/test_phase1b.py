"""Phase 1b の集計ロジックの検算。

skill 6.5a: 同じ数値を別々に書いた 2 つの実装で計算し、一致を確認する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fetch_binance_klines import load_klines                    # noqa: E402
from main_indicators import WARMUP_BARS, wavetrend                   # noqa: E402
from phase1_event_study import moving_block_bootstrap_mean_ci        # noqa: E402
from phase1b_event_study import (                                    # noqa: E402
    SQUEEZE_LOOKBACK_K, block_bootstrap_mean_ci, build_signal_table,
    select_non_overlapping,
)
from squeeze_momentum import SqueezeParams, compute                  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}: {detail}")
        raise AssertionError(f"{name}: {detail}")


def test_bootstrap_agrees_with_phase1() -> None:
    """高速版 (ブロック平均経由) が Phase 1 の実装と同じ答えを出すか。

    乱数列が違うので完全一致はしない。同じ推定量なので、
    区間の端がブートストラップ誤差の範囲で一致することを見る。
    """
    print("\n[1] ブートストラップ: 高速版 vs Phase 1 の実装")
    rng = np.random.default_rng(7)
    for size in (500, 5000):
        values = rng.normal(loc=0.3, scale=5.0, size=size)
        block = max(10, int(round(size ** (1 / 3))))
        slow = moving_block_bootstrap_mean_ci(values, block, rounds=4000)
        fast = block_bootstrap_mean_ci(values, block, rounds=4000)
        width = slow[1] - slow[0]
        difference = max(abs(slow[0] - fast[0]), abs(slow[1] - fast[1]))
        check(f"n={size}: 区間の端が幅の 5% 以内で一致",
              difference < 0.05 * width,
              f"slow={slow} fast={fast} 差={difference:.4f} 幅={width:.4f}")
        print(f"       slow=[{slow[0]:.4f}, {slow[1]:.4f}] "
              f"fast=[{fast[0]:.4f}, {fast[1]:.4f}]")


def test_forward_returns_manually(table) -> None:
    """将来リターンを手計算で 1 件検算する (添字のずれの確認)。"""
    print("\n[2] 将来リターンの手計算による検算")
    data, _ = load_klines(table.symbol, table.interval, "2025-01-01", "2026-06-01",
                          verbose=False)
    open_prices = data["open"].to_numpy()
    close_prices = data["close"].to_numpy()
    squeeze = compute(data, SqueezeParams())
    atr = squeeze["atr_norm"].to_numpy()

    for position in (0, 5, 50):
        row = table.signals.iloc[position]
        i, horizon = int(row["index"]), 6
        manual_bps = (close_prices[i + horizon] / open_prices[i + 1] - 1.0) * 10_000.0
        manual_atr = (close_prices[i + horizon] - open_prices[i + 1]) / atr[i]
        check(f"シグナル #{position}: bps が手計算と一致",
              abs(manual_bps - row["ret_6_bps"]) < 1e-9,
              f"{manual_bps} vs {row['ret_6_bps']}")
        check(f"シグナル #{position}: ATR 単位が手計算と一致",
              abs(manual_atr - row["ret_6_atr"]) < 1e-9,
              f"{manual_atr} vs {row['ret_6_atr']}")


def test_signal_bars_match_indicator(table) -> None:
    """シグナルバーが指標のクロスと一致し、warmup 内が除外されているか。"""
    print("\n[3] シグナル抽出の整合性")
    data, _ = load_klines(table.symbol, table.interval, "2025-01-01", "2026-06-01",
                          verbose=False)
    indicator = wavetrend(data)
    crosses = np.flatnonzero(
        (indicator["cross_up"] | indicator["cross_down"]).to_numpy())
    expected = crosses[crosses >= WARMUP_BARS]
    got = table.signals["index"].to_numpy()
    check("シグナルバーが WaveTrend のクロスと一致",
          np.array_equal(expected, got),
          f"期待 {len(expected)} 本 / 実際 {len(got)} 本")
    check("warmup (先頭 500 本) 内のシグナルは無い", int(got.min()) >= WARMUP_BARS)
    up = indicator["cross_up"].to_numpy()[got]
    check("direction が上抜け=+1 / 下抜け=-1 と一致",
          np.array_equal(table.signals["direction"].to_numpy(), np.where(up, 1.0, -1.0)))


def test_group_assignment(table) -> None:
    """S1 / S2 / S3 の群分けを独立に計算し直して突き合わせる。"""
    print("\n[4] スクイーズ群 (S1/S2/S3) の独立検算")
    data, _ = load_klines(table.symbol, table.interval, "2025-01-01", "2026-06-01",
                          verbose=False)
    squeeze = compute(data, SqueezeParams())
    sqz_on = squeeze["sqz_on"].to_numpy(dtype=bool)
    sqz_off = squeeze["sqz_off"].to_numpy(dtype=bool)

    # 素朴なループで解放バーと「直近 k 本以内」を作り直す
    release = [False] * len(data)
    for i in range(1, len(data)):
        release[i] = bool(sqz_off[i] and sqz_on[i - 1])

    expected_groups = []
    for i in table.signals["index"].to_numpy():
        window = release[max(0, i - SQUEEZE_LOOKBACK_K + 1): i + 1]
        if any(window):
            expected_groups.append("S1")
        elif sqz_on[i]:
            expected_groups.append("S2")
        else:
            expected_groups.append("S3")

    check("群分けが素朴なループ実装と一致",
          expected_groups == table.signals["group"].tolist(),
          f"不一致 {sum(a != b for a, b in zip(expected_groups, table.signals['group']))} 件")
    counts = table.signals["group"].value_counts().to_dict()
    print(f"       S1={counts.get('S1', 0)} S2={counts.get('S2', 0)} S3={counts.get('S3', 0)}")


def test_non_overlapping_selection() -> None:
    """窓非重複の選択が、実際に窓を重ねていないか。"""
    print("\n[5] 窓非重複の選択")
    index_array = np.array([0, 2, 5, 9, 10, 20, 21, 40])
    keep = select_non_overlapping(index_array, horizon=6)
    picked = index_array[keep]
    gaps = np.diff(picked)
    check("選ばれたシグナルの間隔がホライズンより大きい",
          bool((gaps > 6).all()), f"picked={picked.tolist()} gaps={gaps.tolist()}")
    print(f"       {index_array.tolist()} -> {picked.tolist()}")


def main() -> None:
    print("シグナル表の構築: wavetrend BTCUSDT 1h")
    table = build_signal_table("wavetrend", "BTCUSDT", "1h")
    print(f"  シグナル {len(table.signals)} 件 / 全 {table.bar_count} 本")

    test_bootstrap_agrees_with_phase1()
    test_forward_returns_manually(table)
    test_signal_bars_match_indicator(table)
    test_group_assignment(table)
    test_non_overlapping_selection()

    print(f"\n=== 全 {len(PASSED)} 項目 合格 ===")


if __name__ == "__main__":
    main()
