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
