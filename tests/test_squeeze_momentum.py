"""Phase 0: 指標の正確性検証とルックアヘッド検査。

実行: python3 tests/test_squeeze_momentum.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fetch_binance_klines import load_klines            # noqa: E402
from lookahead_check import (                                # noqa: E402
    assert_no_lookahead,
    verify_detector_catches_negative_controls,
)
from squeeze_momentum import (                               # noqa: E402
    SqueezeParams,
    compute,
    linreg_value,
    true_range,
)
from tests.reference_impl import reference_squeeze           # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name}: {detail}")
        raise AssertionError(f"{name}: {detail}")


# --------------------------------------------------------------------------
# 1. linreg の解析解テスト
# --------------------------------------------------------------------------
def test_linreg_on_exact_line() -> None:
    """完全な直線を入れたら、最小二乗直線は元の直線と一致するはず。"""
    print("\n[1] linreg: 完全な直線")
    slope, intercept = 3.0, -7.0
    series = pd.Series(intercept + slope * np.arange(500, dtype=float))
    got = linreg_value(series, 20)
    # 窓の右端 (現在バー) での直線の値 = その時点の実際の値
    expected = series.iloc[19:]
    check(
        "直線に対しては linreg == 元の値",
        np.allclose(got.iloc[19:], expected, rtol=1e-10),
        f"最大差 {np.abs(got.iloc[19:] - expected).max()}",
    )
    check("立ち上がりは NaN", got.iloc[:19].isna().all())


def test_linreg_matches_polyfit() -> None:
    """閉じた形の式が numpy.polyfit と一致するか (別経路での検算)。"""
    print("\n[2] linreg: 閉じた形 vs polyfit")
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(size=300).cumsum())
    n = 20
    got = linreg_value(series, n)
    x = np.arange(n, dtype=float)
    expected = [
        np.polyval(np.polyfit(x, series.iloc[i - n + 1: i + 1].to_numpy(), 1), n - 1)
        for i in range(n - 1, len(series))
    ]
    check(
        "閉じた形 == polyfit",
        np.allclose(got.iloc[n - 1:].to_numpy(), expected, rtol=1e-9),
        f"最大差 {np.abs(got.iloc[n-1:].to_numpy() - np.array(expected)).max()}",
    )


# --------------------------------------------------------------------------
# 2. 既知の落とし穴のテスト
# --------------------------------------------------------------------------
def test_stdev_is_population(data: pd.DataFrame) -> None:
    """dev は母標準偏差 (ddof=0)。pandas 既定の ddof=1 だと値が変わる。"""
    print("\n[3] stdev が母標準偏差か")
    result = compute(data)
    close = data["close"]
    population = 2.0 * close.rolling(20).std(ddof=0)
    sample = 2.0 * close.rolling(20).std(ddof=1)
    check("dev == 2*stdev(ddof=0)", np.allclose(result["dev"].dropna(), population.dropna()))
    check(
        "dev != 2*stdev(ddof=1) (取り違えると値が変わることの確認)",
        not np.allclose(result["dev"].dropna(), sample.dropna()),
    )
    ratio = (sample.dropna() / population.dropna()).mean()
    print(f"       ddof=1/ddof=0 の比 = {ratio:.6f} (理論値 sqrt(20/19)={np.sqrt(20/19):.6f})")


def test_rangema_is_sma_not_wilder(data: pd.DataFrame) -> None:
    """rangema は SMA(TR)。Wilder の ATR (RMA) ではない。"""
    print("\n[4] rangema が SMA(TR) か (Wilder ATR ではないか)")
    result = compute(data)
    tr = true_range(data["high"], data["low"], data["close"])
    sma_tr = tr.rolling(20).mean()
    wilder_atr = tr.ewm(alpha=1 / 20, adjust=False).mean()
    check("rangema == SMA(TR, 20)", np.allclose(result["rangema"].dropna(), sma_tr.dropna()))
    overlap = result["rangema"].dropna().index.intersection(wilder_atr.dropna().index)
    difference = (result["rangema"][overlap] - wilder_atr[overlap]).abs().mean()
    check("rangema != Wilder ATR", difference > 1e-6, f"平均差 {difference}")
    print(f"       SMA(TR) と Wilder ATR の平均差 = {difference:.4f} 価格単位")


def test_kc_and_momentum_are_separable(data: pd.DataFrame) -> None:
    """kc_length とモメンタム期間が独立に動くこと (依頼 0.2-4)。"""
    print("\n[5] KC 期間とモメンタム期間の分離")
    base = compute(data, SqueezeParams())
    kc_changed = compute(data, SqueezeParams(kc_length=30))
    momentum_changed = compute(data, SqueezeParams(momentum_length=30))
    check(
        "kc_length だけ変えても val は変わらない",
        np.allclose(base["val"].dropna(), kc_changed["val"].reindex(base["val"].dropna().index)),
    )
    check(
        "kc_length を変えると sqz_on は変わる",
        not base["sqz_on"].equals(kc_changed["sqz_on"]),
    )
    check(
        "momentum_length を変えると val は変わる",
        not np.allclose(
            base["val"].dropna(),
            momentum_changed["val"].reindex(base["val"].dropna().index),
            equal_nan=True,
        ),
    )
    check(
        "momentum_length を変えても sqz_on は変わらない",
        base["sqz_on"].equals(momentum_changed["sqz_on"]),
    )


def test_squeeze_states_are_exclusive(data: pd.DataFrame) -> None:
    print("\n[6] スクイーズ 3 状態の排他性")
    result = compute(data)
    ready = result["upper_bb"].notna() & result["upper_kc"].notna()
    total = (result["sqz_on"] + result["sqz_off"] + result["no_sqz"])[ready]
    check("sqz_on/sqz_off/no_sqz はちょうど 1 つが True", (total == 1).all(),
          f"合計値の分布 {total.value_counts().to_dict()}")
    print(f"       sqz_on={result['sqz_on'][ready].mean():.1%} "
          f"sqz_off={result['sqz_off'][ready].mean():.1%} "
          f"no_sqz={result['no_sqz'][ready].mean():.1%}")


def test_val_norm_is_scale_free(data: pd.DataFrame) -> None:
    """val はスケール依存、val_norm はスケール非依存 (依頼 0.3)。"""
    print("\n[7] val のスケール依存 / val_norm のスケール非依存")
    scaled = data.copy()
    for column in ("open", "high", "low", "close"):
        scaled[column] = scaled[column] * 1000.0
    base, scaled_result = compute(data), compute(scaled)
    check(
        "val は 1000 倍になる (スケール依存)",
        np.allclose(base["val"].dropna() * 1000.0,
                    scaled_result["val"].reindex(base["val"].dropna().index), rtol=1e-8),
    )
    check(
        "val_norm は変わらない (スケール非依存)",
        np.allclose(base["val_norm"].dropna(),
                    scaled_result["val_norm"].reindex(base["val_norm"].dropna().index), rtol=1e-8),
    )
    check(
        "sqz_on は変わらない (スケール非依存)",
        base["sqz_on"].equals(scaled_result["sqz_on"]),
    )


# --------------------------------------------------------------------------
# 3. 独立した 2 実装の突き合わせ (skill 6.5a)
# --------------------------------------------------------------------------
def test_against_reference_implementation(data: pd.DataFrame) -> None:
    print("\n[8] ベクトル化実装 vs 素朴なループ実装 (独立実装の突き合わせ)")
    subset = data.iloc[:1500]
    fast = compute(subset)
    slow = reference_squeeze(
        subset["open"].tolist(), subset["high"].tolist(),
        subset["low"].tolist(), subset["close"].tolist(),
    )
    for column in ("basis", "upper_bb", "lower_bb", "kc_ma", "rangema",
                   "upper_kc", "lower_kc", "true_range", "val"):
        a = fast[column].to_numpy(dtype=float)
        b = np.array(slow[column], dtype=float)
        both = ~np.isnan(a) & ~np.isnan(b)
        max_difference = np.abs(a[both] - b[both]).max()
        scale = max(np.abs(b[both]).max(), 1.0)
        check(f"{column} が参照実装と一致", max_difference / scale < 1e-9,
              f"相対最大差 {max_difference / scale:.3e}")
    for column in ("sqz_on", "sqz_off"):
        a = fast[column].to_numpy()
        b = np.array(slow[column])
        check(f"{column} が参照実装と一致", (a == b).all(),
              f"不一致 {int((a != b).sum())} 本")


# --------------------------------------------------------------------------
# 4. ルックアヘッド検査 (陰性対照を先に通す)
# --------------------------------------------------------------------------
def test_lookahead(data: pd.DataFrame) -> None:
    print("\n[9] ルックアヘッド検査器の陰性対照 (skill 6.4)")
    subset = data.iloc[:1200]
    verify_detector_catches_negative_controls(subset)
    check("陰性対照 6 種すべてを検出できた", True)

    print("\n[10] 本番の指標に対するルックアヘッド検査")
    numeric_columns = ["basis", "dev", "upper_bb", "lower_bb", "kc_ma", "rangema",
                       "upper_kc", "lower_kc", "val", "atr_norm", "val_norm",
                       "sqz_on", "sqz_off", "no_sqz", "val_rising", "squeeze_run"]

    def build(price_data: pd.DataFrame) -> pd.DataFrame:
        return compute(price_data)[numeric_columns].astype(float)

    assert_no_lookahead(build, subset, warmup=SqueezeParams().warmup_bars)
    check("squeeze_momentum.compute にルックアヘッドなし", True)


def main() -> None:
    print("データ読み込み: BTCUSDT 1h 2025-01..2025-04")
    data, report = load_klines("BTCUSDT", "1h", "2025-01-01", "2025-04-01", verbose=True)
    print(f"  本数の理論値照合: {report.actual_bars}/{report.expected_bars} "
          f"欠損{report.missing_count} 重複{report.duplicated_times}")

    test_linreg_on_exact_line()
    test_linreg_matches_polyfit()
    test_stdev_is_population(data)
    test_rangema_is_sma_not_wilder(data)
    test_kc_and_momentum_are_separable(data)
    test_squeeze_states_are_exclusive(data)
    test_val_norm_is_scale_free(data)
    test_against_reference_implementation(data)
    test_lookahead(data)

    print(f"\n=== 全 {len(PASSED)} 項目 合格 ===")


if __name__ == "__main__":
    main()
