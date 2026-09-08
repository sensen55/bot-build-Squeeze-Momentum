"""Phase 1b の主役指標 (WaveTrend / ADX) の正確性検証とルックアヘッド検査。

実行: python3 tests/test_main_indicators.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fetch_binance_klines import load_klines                # noqa: E402
from lookahead_check import (                                     # noqa: E402
    assert_no_lookahead,
    verify_detector_catches_negative_controls,
)
from main_indicators import (                                     # noqa: E402
    WARMUP_BARS, AdxParams, WaveTrendParams, adx, ema, wavetrend, wilder_smooth,
)
from tests.reference_impl import reference_adx, reference_wavetrend  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}: {detail}")
        raise AssertionError(f"{name}: {detail}")


def compare(name: str, fast: np.ndarray, slow: np.ndarray, tolerance: float = 1e-9) -> None:
    fast = np.asarray(fast, dtype=float)
    slow = np.asarray(slow, dtype=float)
    nan_mismatch = np.isnan(fast) != np.isnan(slow)
    both = ~np.isnan(fast) & ~np.isnan(slow)
    scale = max(float(np.abs(slow[both]).max()) if both.any() else 1.0, 1.0)
    difference = float(np.abs(fast[both] - slow[both]).max()) if both.any() else 0.0
    check(name, (not nan_mismatch.any()) and difference / scale < tolerance,
          f"NaN 不一致 {int(nan_mismatch.sum())} 箇所 / 相対最大差 {difference / scale:.3e}")


# --------------------------------------------------------------------------
# 1. 平滑化の定義そのもの
# --------------------------------------------------------------------------
def test_smoothing_definitions() -> None:
    print("\n[1] EMA / Wilder の漸化式")
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(size=300).cumsum() + 100.0)

    for label, function, alpha in (("EMA(span=10)", ema, 2.0 / 11.0),
                                   ("Wilder(14)", wilder_smooth, 1.0 / 14.0)):
        length = 10 if "EMA" in label else 14
        got = function(series, length).to_numpy()
        manual = np.empty(len(series))
        manual[0] = series.iloc[0]
        for i in range(1, len(series)):
            manual[i] = alpha * series.iloc[i] + (1 - alpha) * manual[i - 1]
        check(f"{label} が漸化式と一致", np.allclose(got, manual, rtol=1e-12))

    adjusted = series.ewm(span=10, adjust=True).mean()
    unadjusted = series.ewm(span=10, adjust=False).mean()
    check("adjust=True と adjust=False は別物 (pandas 既定の罠)",
          not np.allclose(adjusted, unadjusted))
    check("EMA(14) と Wilder(14) は別物 (alpha が違う)",
          not np.allclose(ema(series, 14), wilder_smooth(series, 14)))
    print(f"       EMA(14) alpha={2/15:.4f} vs Wilder(14) alpha={1/14:.4f}")


# --------------------------------------------------------------------------
# 2. +DM / -DM の規則
# --------------------------------------------------------------------------
def test_directional_movement_rule() -> None:
    """「大きい方だけ採用、小さい方はゼロ」を作った合成データで確認する。"""
    print("\n[2] +DM / -DM の規則")
    # bar1: up=5, down=3   -> +DM=5, -DM=0
    # bar2: up=-2, down=4  -> +DM=0, -DM=4
    # bar3: up=3, down=3   -> どちらも相手より大きくないので両方 0
    data = pd.DataFrame({
        "open":  [100.0, 100.0, 100.0, 100.0],
        "high":  [110.0, 115.0, 113.0, 116.0],
        "low":   [ 90.0,  87.0,  83.0,  80.0],
        "close": [100.0, 100.0, 100.0, 100.0],
    })
    result = adx(data)
    check("+DM: 上が大きい日は上だけ採用",
          result["plus_dm"].tolist() == [0.0, 5.0, 0.0, 0.0],
          str(result["plus_dm"].tolist()))
    check("-DM: 下が大きい日は下だけ採用",
          result["minus_dm"].tolist() == [0.0, 0.0, 4.0, 0.0],
          str(result["minus_dm"].tolist()))
    print("       up=3/down=3 の同値ケースは両方 0 (どちらも相手より大きくない)")


# --------------------------------------------------------------------------
# 3. 独立実装との突き合わせ (skill 6.5a)
# --------------------------------------------------------------------------
def test_wavetrend_against_reference(data: pd.DataFrame) -> None:
    print("\n[3] WaveTrend: ベクトル化 vs 素朴なループ")
    subset = data.iloc[:2000]
    fast = wavetrend(subset)
    slow = reference_wavetrend(subset["high"].tolist(), subset["low"].tolist(),
                               subset["close"].tolist())
    for column in ("hlc3", "esa", "d", "ci", "wt1", "wt2"):
        compare(f"WaveTrend {column} が参照実装と一致",
                fast[column].to_numpy(), np.array(slow[column], dtype=float))
    for column in ("cross_up", "cross_down"):
        check(f"WaveTrend {column} が参照実装と一致",
              (fast[column].to_numpy() == np.array(slow[column])).all(),
              f"不一致 {int((fast[column].to_numpy() != np.array(slow[column])).sum())} 本")
    print(f"       ゼロ除算 (d==0) のバー: {fast.attrs['zero_division_bars']} 本")


def test_adx_against_reference(data: pd.DataFrame) -> None:
    print("\n[4] ADX: ベクトル化 vs 素朴なループ")
    subset = data.iloc[:2000]
    fast = adx(subset)
    slow = reference_adx(subset["high"].tolist(), subset["low"].tolist(),
                         subset["close"].tolist())
    for column in ("true_range", "plus_dm", "minus_dm", "plus_di", "minus_di", "dx", "adx"):
        compare(f"ADX {column} が参照実装と一致",
                fast[column].to_numpy(), np.array(slow[column], dtype=float))


# --------------------------------------------------------------------------
# 4. 指標としての健全性
# --------------------------------------------------------------------------
def test_sanity(data: pd.DataFrame) -> None:
    print("\n[5] 指標の値域と warmup")
    wt = wavetrend(data).iloc[WARMUP_BARS:]
    ax = adx(data).iloc[WARMUP_BARS:]
    check("ADX は 0..100 の範囲", bool((ax["adx"] >= 0).all() and (ax["adx"] <= 100).all()),
          f"range {ax['adx'].min()}..{ax['adx'].max()}")
    check("DI+ / DI- は非負", bool((ax["plus_di"] >= 0).all() and (ax["minus_di"] >= 0).all()))
    check("warmup 後の wt1 は概ね ±150 に収まる",
          bool(wt["wt1"].abs().max() < 150), f"max |wt1| = {wt['wt1'].abs().max():.1f}")
    check("上抜けと下抜けが同時に立つバーは無い",
          not bool((wt["cross_up"] & wt["cross_down"]).any()))
    print(f"       wt1 range {wt['wt1'].min():.1f}..{wt['wt1'].max():.1f} / "
          f"adx range {ax['adx'].min():.1f}..{ax['adx'].max():.1f}")
    print(f"       cross_up={int(wt['cross_up'].sum())} "
          f"cross_down={int(wt['cross_down'].sum())} "
          f"adx>20 上抜け={int(ax['cross_threshold'].sum())}")


def test_scale_invariance(data: pd.DataFrame) -> None:
    """WaveTrend と ADX はどちらもスケール非依存のはず。

    val (Phase 0) がスケール依存だったのと対照的。
    銘柄横断で同じ閾値を使ってよい根拠になる。
    """
    print("\n[6] スケール非依存性")
    scaled = data.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * 1000.0
    base_wt, scaled_wt = wavetrend(data), wavetrend(scaled)
    base_adx, scaled_adx = adx(data), adx(scaled)
    compare("wt1 は 1000 倍しても不変", base_wt["wt1"].to_numpy(),
            scaled_wt["wt1"].to_numpy(), tolerance=1e-8)
    compare("adx は 1000 倍しても不変", base_adx["adx"].to_numpy(),
            scaled_adx["adx"].to_numpy(), tolerance=1e-8)


# --------------------------------------------------------------------------
# 5. ルックアヘッド検査 (陰性対照を先に通す)
# --------------------------------------------------------------------------
def test_lookahead(data: pd.DataFrame) -> None:
    print("\n[7] ルックアヘッド検査器の陰性対照 (skill 6.4)")
    subset = data.iloc[:1500]
    verify_detector_catches_negative_controls(subset)
    check("陰性対照 6 種すべてを検出できた", True)

    print("\n[8] WaveTrend / ADX に対するルックアヘッド検査")
    wt_columns = ["hlc3", "esa", "d", "ci", "wt1", "wt2",
                  "cross_up", "cross_down", "oversold_zone", "overbought_zone"]
    ax_columns = ["true_range", "plus_dm", "minus_dm", "plus_di", "minus_di",
                  "dx", "adx", "cross_threshold", "cross_secondary", "bullish"]

    assert_no_lookahead(lambda d: wavetrend(d)[wt_columns].astype(float),
                        subset, warmup=WARMUP_BARS // 2)
    check("main_indicators.wavetrend にルックアヘッドなし", True)

    assert_no_lookahead(lambda d: adx(d)[ax_columns].astype(float),
                        subset, warmup=WARMUP_BARS // 2)
    check("main_indicators.adx にルックアヘッドなし", True)


def main() -> None:
    print("データ読み込み: BTCUSDT 1h 2025-01..2025-06")
    data, report = load_klines("BTCUSDT", "1h", "2025-01-01", "2025-06-01", verbose=True)
    print(f"  本数の理論値照合: {report.actual_bars}/{report.expected_bars} "
          f"欠損{report.missing_count} 重複{report.duplicated_times}")

    test_smoothing_definitions()
    test_directional_movement_rule()
    test_wavetrend_against_reference(data)
    test_adx_against_reference(data)
    test_sanity(data)
    test_scale_invariance(data)
    test_lookahead(data)

    print(f"\n=== 全 {len(PASSED)} 項目 合格 ===")


if __name__ == "__main__":
    main()
