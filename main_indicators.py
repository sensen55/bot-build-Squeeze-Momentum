"""Phase 1b の「主役」候補: WaveTrend Oscillator と ADX。

どちらも指標計算のみ。売買ロジックは含めない (Phase 0 と同じ方針)。

平滑化の使い分け (混同しやすいので明示)
--------------------------------------
指標ごとに「正しい平滑化」が違う。原典がどう書いているかで決まる。

| 場所                          | 正しい平滑化 | Python                          |
|------------------------------|------------|---------------------------------|
| Squeeze の rangema (Phase 0)  | SMA        | TR.rolling(20).mean()           |
| WaveTrend の esa / d / tci    | EMA        | ewm(span=n, adjust=False)       |
| ADX の各平滑化                 | Wilder(RMA)| ewm(alpha=1/n, adjust=False)    |

Phase 0 で「Wilder の ATR を使うな」と書いたのは、Squeeze の原典が SMA だから
であって Wilder が悪いからではない。ADX の原典 (Wilder 本人) は Wilder 平滑化を
指定しているので、そちらでは Wilder が正解になる。

pandas の ewm は既定が adjust=True で、これは Pine の ema / rma と一致しない。
必ず adjust=False を指定すること。

ルックアヘッドについて
----------------------
ewm(adjust=False) は漸化式 y[i] = a*x[i] + (1-a)*y[i-1] であり、
y[i] は x[:i+1] のみの関数。rolling(n) も現在バーを含む過去 n 本のみ。
未来方向のシフト、全期間統計、bfill、interpolate は一切使っていない。
この性質は tests/test_main_indicators.py の切断テストで機械的に検証している。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# EMA の立ち上がりは長く尾を引くため、先頭をまとめて解析対象外にする。
# Phase 1 の warmup (atr_norm の 100 本) より長く取る。
WARMUP_BARS = 500


def ema(source: pd.Series, length: int) -> pd.Series:
    """Pine の ema(source, length)。

    y[i] = alpha*x[i] + (1-alpha)*y[i-1],  alpha = 2/(length+1),  y[0] = x[0]
    pandas 既定の adjust=True は別物なので必ず adjust=False を指定する。
    """
    return source.ewm(span=length, adjust=False).mean()


def wilder_smooth(source: pd.Series, length: int) -> pd.Series:
    """Wilder の平滑化 (Pine の rma)。

    y[i] = x[i]/length + y[i-1]*(length-1)/length,  y[0] = x[0]
    = ewm(alpha=1/length, adjust=False)
    EMA (alpha = 2/(n+1)) とは alpha が違う。混同しないこと。
    """
    return source.ewm(alpha=1.0 / length, adjust=False).mean()


# ---------------------------------------------------------------------------
# WaveTrend Oscillator [LazyBear]
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WaveTrendParams:
    channel_length: int = 10        # n1
    average_length: int = 21        # n2
    signal_length: int = 4          # wt2 の SMA 期間
    oversold: float = -53.0         # 副次的な記述統計にのみ使う
    overbought: float = 53.0


def wavetrend(data: pd.DataFrame, params: WaveTrendParams | None = None) -> pd.DataFrame:
    """WaveTrend Oscillator [LazyBear]。

    ap  = hlc3
    esa = ema(ap, n1)
    d   = ema(|ap - esa|, n1)
    ci  = (ap - esa) / (0.015 * d)
    wt1 = ema(ci, n2)
    wt2 = sma(wt1, 4)

    d == 0 のとき ci は発散する。その場合は NaN にし、本数を
    DataFrame.attrs["zero_division_bars"] に記録する。
    """
    params = params or WaveTrendParams()

    average_price = (data["high"] + data["low"] + data["close"]) / 3.0
    esa = ema(average_price, params.channel_length)
    deviation = ema((average_price - esa).abs(), params.channel_length)

    # ゼロ除算: d == 0 の位置を NaN にする (0 で割った結果を使わない)
    zero_division = deviation == 0.0
    safe_deviation = deviation.where(~zero_division)
    channel_index = (average_price - esa) / (0.015 * safe_deviation)

    wt1 = ema(channel_index, params.average_length)
    wt2 = wt1.rolling(params.signal_length).mean()

    previous_wt1 = wt1.shift(1)         # 過去方向のみ
    previous_wt2 = wt2.shift(1)
    cross_up = (wt1 > wt2) & (previous_wt1 <= previous_wt2)
    cross_down = (wt1 < wt2) & (previous_wt1 >= previous_wt2)

    result = pd.DataFrame(
        {
            "hlc3": average_price,
            "esa": esa,
            "d": deviation,
            "ci": channel_index,
            "wt1": wt1,
            "wt2": wt2,
            "cross_up": cross_up.fillna(False),
            "cross_down": cross_down.fillna(False),
            "oversold_zone": wt1 < params.oversold,
            "overbought_zone": wt1 > params.overbought,
        },
        index=data.index,
    )
    result.attrs["zero_division_bars"] = int(zero_division.sum())
    return result


# ---------------------------------------------------------------------------
# ADX (Wilder, 期間 14)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AdxParams:
    length: int = 14
    threshold: float = 20.0         # 主判定に使う閾値
    secondary_threshold: float = 25.0   # 副次的な記述統計にのみ使う


def adx(data: pd.DataFrame, params: AdxParams | None = None) -> pd.DataFrame:
    """ADX と DI+ / DI-。

    +DM / -DM の「大きい方だけ採用、小さい方はゼロ」という教科書どおりの規則:

        up   = high[t] - high[t-1]
        down = low[t-1] - low[t]
        +DM  = up   if (up > down   and up > 0)   else 0
        -DM  = down if (down > up   and down > 0) else 0

    両方上がった日は「上」だけを数える、という考え方。
    平滑化はすべて Wilder (alpha = 1/length)。EMA ではない。
    """
    params = params or AdxParams()
    n = params.length
    high, low, close = data["high"], data["low"], data["close"]

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    if len(true_range):
        true_range.iloc[0] = high.iloc[0] - low.iloc[0]

    up_move = high.diff()
    down_move = -low.diff()          # low[t-1] - low[t]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=data.index)
    minus_dm = pd.Series(minus_dm, index=data.index)
    plus_dm.iloc[0] = 0.0            # 前バーが無い最初のバーは 0
    minus_dm.iloc[0] = 0.0

    smoothed_tr = wilder_smooth(true_range, n)
    smoothed_plus = wilder_smooth(plus_dm, n)
    smoothed_minus = wilder_smooth(minus_dm, n)

    safe_tr = smoothed_tr.where(smoothed_tr != 0.0)
    plus_di = 100.0 * smoothed_plus / safe_tr
    minus_di = 100.0 * smoothed_minus / safe_tr

    di_sum = plus_di + minus_di
    directional_index = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0.0)
    # DI+ と DI- がともに 0 の日は「方向の偏りが無い」ので DX = 0 として扱う。
    # これは同一バーの値からの決定であり、未来からの補完 (bfill) ではない。
    adx_line = wilder_smooth(directional_index.fillna(0.0), n)

    previous_adx = adx_line.shift(1)
    cross_threshold = (adx_line > params.threshold) & (previous_adx <= params.threshold)
    cross_secondary = (
        (adx_line > params.secondary_threshold)
        & (previous_adx <= params.secondary_threshold)
    )

    result = pd.DataFrame(
        {
            "true_range": true_range,
            "plus_dm": plus_dm,
            "minus_dm": minus_dm,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "dx": directional_index,
            "adx": adx_line,
            "cross_threshold": cross_threshold.fillna(False),
            "cross_secondary": cross_secondary.fillna(False),
            "bullish": plus_di > minus_di,
        },
        index=data.index,
    )
    result.attrs["zero_tr_bars"] = int((smoothed_tr == 0.0).sum())
    return result
