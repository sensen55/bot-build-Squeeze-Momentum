"""Squeeze Momentum Indicator [LazyBear] の指標計算のみ。

売買ロジックはここに書かない (依頼の Phase 0 要件)。

Pine 原典 (LazyBear v2 / pastebin.com/UCpcX8d7) からの移植で、
Python で間違えやすい箇所は以下のとおり明示的に処理している。

1. ``stdev`` は母標準偏差。pandas の既定 ddof=1 ではなく ``ddof=0`` を指定する。
2. ``rangema`` は SMA(TR) であって Wilder の ATR ではない。
   TA-Lib / pandas-ta の ATR を使うと別物になるので自前で TR.rolling(n).mean() を書く。
3. ``linreg(x, N, 0)`` は直近 N 本への最小二乗直線の、現在バー位置の値。
   閉じた形: slope = Σ(xi-x̄)(yi-ȳ) / Σ(xi-x̄)², val = ȳ + slope*(N-1)/2
   (x は 0..N-1、Σ(xi-x̄)² = N(N²-1)/12)。ゼロラグではない。
4. 原典では ``lengthKC`` が KC 期間とモメンタム期間を兼ねているが、
   本実装では ``kc_length`` と ``momentum_length`` に分離した (既定はどちらも 20)。

ルックアヘッドについて
----------------------
本モジュールが使う演算は ``rolling(N)`` (現在バーを含む過去 N 本) と
``shift(+1)`` (過去方向) のみ。未来方向の ``shift(-N)``、全期間統計
(``.mean()`` / ``.std()`` / ``.min()`` / ``.max()`` を列全体に適用)、
``bfill`` / ``interpolate`` は一切使っていない。
したがって時刻 t の出力は t 以前の入力のみの関数である。
この性質は tests/test_lookahead.py の切断テストで機械的に検証している。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class SqueezeParams:
    """指標パラメータ。既定値は LazyBear 原典と同じ。"""

    bb_length: int = 20
    bb_mult: float = 2.0
    kc_length: int = 20
    kc_mult: float = 1.5
    use_true_range: bool = True
    momentum_length: int = 20      # 原典では kc_length と共有されている
    atr_norm_length: int = 100     # val の ATR 正規化に使う SMA(TR) の期間

    def __post_init__(self) -> None:
        for name in ("bb_length", "kc_length", "momentum_length", "atr_norm_length"):
            if getattr(self, name) < 2:
                raise ValueError(f"{name} は 2 以上である必要があります")

    @property
    def warmup_bars(self) -> int:
        """立ち上がりに必要な本数。これ未満のバーの出力は NaN になる。"""
        return max(self.bb_length, self.kc_length, self.momentum_length,
                   self.atr_norm_length) + 1


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Pine の ``tr``。最初のバーは前終値が無いので high-low になる。"""
    previous_close = close.shift(1)
    candidates = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    result = candidates.max(axis=1)
    result.iloc[0] = (high.iloc[0] - low.iloc[0]) if len(high) else np.nan
    return result


def linreg_value(source: pd.Series, length: int) -> pd.Series:
    """Pine の ``linreg(source, length, 0)``。

    直近 ``length`` 本 (現在バーを含む) に最小二乗直線を当て、
    現在バー位置での直線の値を返す。

    x = 0..N-1 (0 が最も古い、N-1 が現在バー) とすると
        x̄ = (N-1)/2
        Σ(xi-x̄)² = N(N²-1)/12
        slope = Σ(xi-x̄)·yi / Σ(xi-x̄)²        ← ȳ の項は Σ(xi-x̄)=0 で消える
        値 = ȳ + slope·((N-1) - x̄) = ȳ + slope·(N-1)/2

    ループ内で pandas を触らないよう、sliding_window_view と行列積で計算する
    (skill 6.1)。窓は「現在バーで終わる過去 N 本」なので未来は含まない。
    """
    values = source.to_numpy(dtype=float)
    n = int(length)
    if len(values) < n:
        return pd.Series(np.full(len(values), np.nan), index=source.index)

    x = np.arange(n, dtype=float)
    centered_x = x - x.mean()
    denominator = n * (n * n - 1) / 12.0

    windows = sliding_window_view(values, n)          # shape: (len-n+1, n)
    slope = windows @ centered_x / denominator
    y_mean = windows.mean(axis=1)
    fitted = y_mean + slope * (n - 1) / 2.0

    out = np.full(len(values), np.nan)
    out[n - 1:] = fitted                              # 窓の右端 = 現在バーに揃える
    return pd.Series(out, index=source.index)


def compute(data: pd.DataFrame, params: SqueezeParams | None = None) -> pd.DataFrame:
    """Squeeze Momentum の全系列を返す。

    引数 ``data`` は open/high/low/close を持ち、index は
    **バー確定時刻** であることを前提とする。

    返す列
    ------
    basis, dev, upper_bb, lower_bb : ボリンジャーバンド
    kc_ma, rangema, upper_kc, lower_kc : ケルトナーチャネル
    sqz_on, sqz_off, no_sqz : スクイーズ 3 状態 (相互排他)
    val : モメンタムヒストグラム (価格単位。スケール依存)
    val_norm : val / SMA(TR, atr_norm_length) (スケール非依存)
    val_rising : val > val[1] (直前バーとの比較)
    color_state : 原典の 4 状態 (下記)
    squeeze_run : その時点までに sqz_on が連続している本数
    """
    params = params or SqueezeParams()

    required = {"open", "high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"必要な列がありません: {sorted(missing)}")

    high, low, close = data["high"], data["low"], data["close"]

    # --- ボリンジャーバンド ---------------------------------------------
    basis = close.rolling(params.bb_length).mean()
    # ddof=0 = 母標準偏差。Pine の stdev と一致させるために必須。
    deviation = params.bb_mult * close.rolling(params.bb_length).std(ddof=0)
    upper_bb = basis + deviation
    lower_bb = basis - deviation

    # --- ケルトナーチャネル ---------------------------------------------
    kc_ma = close.rolling(params.kc_length).mean()
    tr = true_range(high, low, close)
    bar_range = tr if params.use_true_range else (high - low)
    # SMA(TR)。Wilder の平滑化 (ATR) ではない。
    rangema = bar_range.rolling(params.kc_length).mean()
    upper_kc = kc_ma + rangema * params.kc_mult
    lower_kc = kc_ma - rangema * params.kc_mult

    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz = ~sqz_on & ~sqz_off

    # 立ち上がり期間 (バンドが NaN) は「スクイーズ判定不能」として全部 False にする。
    # NaN 比較は False になるので sqz_on/sqz_off は自動的に False だが、
    # no_sqz が誤って True になるのを防ぐ。
    bands_ready = upper_bb.notna() & upper_kc.notna()
    sqz_on = sqz_on & bands_ready
    sqz_off = sqz_off & bands_ready
    no_sqz = no_sqz & bands_ready

    # --- モメンタムヒストグラム -----------------------------------------
    n = params.momentum_length
    highest_high = high.rolling(n).max()
    lowest_low = low.rolling(n).min()
    donchian_mid = (highest_high + lowest_low) / 2.0
    sma_close = close.rolling(n).mean()
    reference = (donchian_mid + sma_close) / 2.0
    val = linreg_value(close - reference, n)

    # --- ATR 正規化 -------------------------------------------------------
    # 全期間統計ではなく rolling を使う (依頼 0.3 / skill 2.2)。
    # 2025年1月の判定に2026年のボラティリティが混ざることを防ぐ。
    atr_norm = bar_range.rolling(params.atr_norm_length).mean()
    val_norm = val / atr_norm.replace(0.0, np.nan)

    val_previous = val.shift(1)                       # 過去方向のみ
    val_rising = val > val_previous

    color_state = pd.Series(pd.NA, index=data.index, dtype="object")
    positive = val > 0
    color_state[positive & val_rising] = "lime"        # 上昇モメンタム加速
    color_state[positive & ~val_rising] = "green"      # 上昇モメンタム減速
    color_state[~positive & (val < val_previous)] = "red"    # 下落モメンタム加速
    color_state[~positive & (val >= val_previous)] = "maroon"  # 下落モメンタム減速
    color_state[val.isna() | val_previous.isna()] = pd.NA

    result = pd.DataFrame(
        {
            "basis": basis,
            "dev": deviation,
            "upper_bb": upper_bb,
            "lower_bb": lower_bb,
            "kc_ma": kc_ma,
            "true_range": tr,
            "rangema": rangema,
            "upper_kc": upper_kc,
            "lower_kc": lower_kc,
            "sqz_on": sqz_on,
            "sqz_off": sqz_off,
            "no_sqz": no_sqz,
            "val": val,
            "atr_norm": atr_norm,
            "val_norm": val_norm,
            "val_rising": val_rising,
            "color_state": color_state,
            "squeeze_run": _consecutive_run(sqz_on.to_numpy()),
        },
        index=data.index,
    )
    return result


def _consecutive_run(flags: np.ndarray) -> np.ndarray:
    """True が現在バーまで何本連続しているかを返す (現在バーを含む)。

    未来を見ないことは自明: 出力 i は flags[:i+1] のみの関数。
    """
    run = np.zeros(len(flags), dtype=np.int32)
    count = 0
    for i, flag in enumerate(flags):
        count = count + 1 if flag else 0
        run[i] = count
    return run
