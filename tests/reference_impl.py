"""Pine 原典から素朴に書き起こした参照実装 (検算専用・遅い)。

skill 6.5a: バックテストのバグはエラーにならず、それらしい数字が出る。
だから同じ数値を**別々に書いた 2 つの実装**で計算して一致を見る。

こちらは 1 バーずつ Python のリストを回す実装で、
squeeze_momentum.py (numpy ベクトル化) とはコードを一切共有していない。
linreg には numpy.polyfit を使い、閉じた形の式とは別経路で解を求める。
"""

from __future__ import annotations

import math

import numpy as np


def _population_stdev(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def reference_squeeze(
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
    bb_length: int = 20,
    bb_mult: float = 2.0,
    kc_length: int = 20,
    kc_mult: float = 1.5,
    momentum_length: int = 20,
) -> dict[str, list[float]]:
    bar_count = len(close)
    nan = float("nan")

    true_range = [nan] * bar_count
    for i in range(bar_count):
        if i == 0:
            true_range[i] = high[i] - low[i]
        else:
            true_range[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

    basis = [nan] * bar_count
    upper_bb = [nan] * bar_count
    lower_bb = [nan] * bar_count
    kc_ma = [nan] * bar_count
    rangema = [nan] * bar_count
    upper_kc = [nan] * bar_count
    lower_kc = [nan] * bar_count
    sqz_on = [False] * bar_count
    sqz_off = [False] * bar_count
    momentum_source = [nan] * bar_count
    val = [nan] * bar_count

    for i in range(bar_count):
        if i >= bb_length - 1:
            window = close[i - bb_length + 1: i + 1]
            b = sum(window) / bb_length
            d = bb_mult * _population_stdev(window)
            basis[i], upper_bb[i], lower_bb[i] = b, b + d, b - d

        if i >= kc_length - 1:
            window_close = close[i - kc_length + 1: i + 1]
            window_tr = true_range[i - kc_length + 1: i + 1]
            m = sum(window_close) / kc_length
            r = sum(window_tr) / kc_length
            kc_ma[i], rangema[i] = m, r
            upper_kc[i], lower_kc[i] = m + r * kc_mult, m - r * kc_mult

        if not math.isnan(upper_bb[i]) and not math.isnan(upper_kc[i]):
            sqz_on[i] = lower_bb[i] > lower_kc[i] and upper_bb[i] < upper_kc[i]
            sqz_off[i] = lower_bb[i] < lower_kc[i] and upper_bb[i] > upper_kc[i]

        if i >= momentum_length - 1:
            hh = max(high[i - momentum_length + 1: i + 1])
            ll = min(low[i - momentum_length + 1: i + 1])
            sma = sum(close[i - momentum_length + 1: i + 1]) / momentum_length
            momentum_source[i] = close[i] - ((hh + ll) / 2.0 + sma) / 2.0

        if i >= 2 * momentum_length - 2:
            window = momentum_source[i - momentum_length + 1: i + 1]
            x = np.arange(momentum_length, dtype=float)
            slope, intercept = np.polyfit(x, np.array(window, dtype=float), 1)
            val[i] = slope * (momentum_length - 1) + intercept

    return {
        "basis": basis,
        "upper_bb": upper_bb,
        "lower_bb": lower_bb,
        "kc_ma": kc_ma,
        "rangema": rangema,
        "upper_kc": upper_kc,
        "lower_kc": lower_kc,
        "sqz_on": sqz_on,
        "sqz_off": sqz_off,
        "true_range": true_range,
        "val": val,
    }


# ---------------------------------------------------------------------------
# Phase 1b: WaveTrend / ADX の参照実装 (素朴なループ・検算専用)
# ---------------------------------------------------------------------------
def _reference_ema(values: list[float], length: int) -> list[float]:
    """Pine の ema を素朴な漸化式で。

    NaN の扱いは pandas の ewm(adjust=False) に合わせる:
      - 先頭の連続 NaN は結果も NaN。最初の有効値から漸化式を開始する
      - 途中の NaN では更新せず、直前の値をそのまま持ち越す
    """
    alpha = 2.0 / (length + 1.0)
    return _reference_recursive_smooth(values, alpha)


def _reference_wilder(values: list[float], length: int) -> list[float]:
    """Wilder の平滑化 (Pine の rma)。EMA とは alpha が違う。"""
    return _reference_recursive_smooth(values, 1.0 / length)


def _reference_recursive_smooth(values: list[float], alpha: float) -> list[float]:
    out: list[float] = []
    previous = float("nan")
    started = False
    for value in values:
        if math.isnan(value):
            out.append(previous if started else float("nan"))
            continue
        if not started:
            previous, started = value, True
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        out.append(previous)
    return out


def reference_wavetrend(
    high: list[float],
    low: list[float],
    close: list[float],
    channel_length: int = 10,
    average_length: int = 21,
    signal_length: int = 4,
) -> dict[str, list[float]]:
    bar_count = len(close)
    nan = float("nan")

    average_price = [(high[i] + low[i] + close[i]) / 3.0 for i in range(bar_count)]
    esa = _reference_ema(average_price, channel_length)
    absolute_deviation = [abs(average_price[i] - esa[i]) for i in range(bar_count)]
    deviation = _reference_ema(absolute_deviation, channel_length)

    channel_index = [nan] * bar_count
    for i in range(bar_count):
        if deviation[i] == 0.0 or math.isnan(deviation[i]):
            channel_index[i] = nan
        else:
            channel_index[i] = (average_price[i] - esa[i]) / (0.015 * deviation[i])

    wt1 = _reference_ema(channel_index, average_length)

    wt2 = [nan] * bar_count
    for i in range(signal_length - 1, bar_count):
        window = wt1[i - signal_length + 1: i + 1]
        if all(not math.isnan(v) for v in window):
            wt2[i] = sum(window) / signal_length

    cross_up = [False] * bar_count
    cross_down = [False] * bar_count
    for i in range(1, bar_count):
        values = (wt1[i], wt2[i], wt1[i - 1], wt2[i - 1])
        if any(math.isnan(v) for v in values):
            continue
        cross_up[i] = wt1[i] > wt2[i] and wt1[i - 1] <= wt2[i - 1]
        cross_down[i] = wt1[i] < wt2[i] and wt1[i - 1] >= wt2[i - 1]

    return {"hlc3": average_price, "esa": esa, "d": deviation,
            "ci": channel_index, "wt1": wt1, "wt2": wt2,
            "cross_up": cross_up, "cross_down": cross_down}


def reference_adx(
    high: list[float],
    low: list[float],
    close: list[float],
    length: int = 14,
) -> dict[str, list[float]]:
    bar_count = len(close)
    nan = float("nan")

    true_range = [nan] * bar_count
    plus_dm = [0.0] * bar_count
    minus_dm = [0.0] * bar_count
    for i in range(bar_count):
        if i == 0:
            true_range[i] = high[i] - low[i]
            continue
        true_range[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        # 「大きい方だけ採用、小さい方はゼロ」
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    smoothed_tr = _reference_wilder(true_range, length)
    smoothed_plus = _reference_wilder(plus_dm, length)
    smoothed_minus = _reference_wilder(minus_dm, length)

    plus_di = [nan] * bar_count
    minus_di = [nan] * bar_count
    directional_index = [nan] * bar_count
    dx_filled = [0.0] * bar_count
    for i in range(bar_count):
        if smoothed_tr[i] == 0.0 or math.isnan(smoothed_tr[i]):
            continue
        plus_di[i] = 100.0 * smoothed_plus[i] / smoothed_tr[i]
        minus_di[i] = 100.0 * smoothed_minus[i] / smoothed_tr[i]
        total = plus_di[i] + minus_di[i]
        if total != 0.0:
            directional_index[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / total
            dx_filled[i] = directional_index[i]

    adx_line = _reference_wilder(dx_filled, length)

    return {"true_range": true_range, "plus_dm": plus_dm, "minus_dm": minus_dm,
            "plus_di": plus_di, "minus_di": minus_di,
            "dx": directional_index, "adx": adx_line}
