"""Phase 1c: グリッド設定を評価するエンジン。

Part A / Part B の両方で同じコードを使う。違うのは期間だけ。

ルックアヘッドを構造的に防ぐ設計 (Phase 1 / 1b と同一)
------------------------------------------------------
  判断に使える情報 : シグナルバー t の終値確定時点で既知の値のみ
  エントリー価格   : open[t+1]
  決済価格         : close[t+N]
`entry_index = signal_index + 1` として固定してあり、詰めることができない。

ATR 正規化は全設定で共通の基準を使う
------------------------------------
リターンの正規化に使う ATR は、設定によらず固定の SqueezeParams()
(SMA(TR, 100)) から取る。設定ごとに正規化の分母が変わると、
順位の比較が「リターンの差」なのか「分母の差」なのか分からなくなる。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data.fetch_binance_klines import INTERVAL_MINUTES, load_klines
from main_indicators import AdxParams, WaveTrendParams, adx, wavetrend
from phase1c_grid import HORIZONS, WARMUP_BARS, Setting
from squeeze_momentum import SqueezeParams, compute


@dataclass
class MarketContext:
    """1 つの (銘柄, 時間足, 期間) について、設定に依存しない部分を先に作っておく。

    データ読み込みと将来リターンの計算は設定ごとに変わらないので、
    65 設定ぶん繰り返さずに使い回す。
    """

    symbol: str
    interval: str
    data: pd.DataFrame
    atr: np.ndarray
    forward_atr: dict[int, np.ndarray]
    forward_bps: dict[int, np.ndarray]
    valid: np.ndarray
    bar_count: int
    integrity: object = None


def build_context(symbol: str, interval: str, start: str, end: str,
                  verbose: bool = False) -> MarketContext:
    data, report = load_klines(symbol, interval, start, end, verbose=verbose)
    step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    gaps = data.index.to_series().diff().dropna()
    if len(gaps[gaps != step]):
        raise ValueError(f"{symbol} {interval}: バーが等間隔ではありません")

    squeeze = compute(data, SqueezeParams())
    atr = squeeze["atr_norm"].to_numpy(dtype=float)
    open_prices = data["open"].to_numpy(dtype=float)
    close_prices = data["close"].to_numpy(dtype=float)
    bar_count = len(data)

    forward_bps: dict[int, np.ndarray] = {}
    forward_atr: dict[int, np.ndarray] = {}
    for horizon in HORIZONS:
        bps = np.full(bar_count, np.nan)
        in_atr = np.full(bar_count, np.nan)
        last = bar_count - horizon - 1
        if last > 0:
            entry = open_prices[1: last + 2]
            exit_ = close_prices[horizon: last + horizon + 1]
            bps[: last + 1] = (exit_ / entry - 1.0) * 10_000.0
            in_atr[: last + 1] = (exit_ - entry) / atr[: last + 1]
        forward_bps[horizon] = bps
        forward_atr[horizon] = in_atr

    valid = np.zeros(bar_count, dtype=bool)
    valid[WARMUP_BARS:] = True
    valid &= ~np.isnan(atr)

    return MarketContext(symbol, interval, data, atr, forward_atr, forward_bps,
                         valid, bar_count, report)


def make_signals(setting: Setting, context: MarketContext) -> tuple[np.ndarray, np.ndarray, int]:
    """設定に応じたシグナルバーと方向を返す。

    戻り値: (シグナルバーの添字, 方向 (+1/-1), 参考情報としての青(no_sqz)の本数)
    """
    data = context.data
    blue_bars = 0

    if setting.family == "adx":
        indicator = adx(data, AdxParams(length=setting.params["length"],
                                        threshold=setting.params["threshold"]))
        is_signal = indicator["cross_threshold"].to_numpy(dtype=bool)
        direction = np.where(indicator["bullish"].to_numpy(dtype=bool), 1.0, -1.0)

    elif setting.family == "wavetrend":
        params = WaveTrendParams(channel_length=setting.params["n1"],
                                 average_length=setting.params["n2"])
        indicator = wavetrend(data, params)
        cross_up = indicator["cross_up"].to_numpy(dtype=bool)
        cross_down = indicator["cross_down"].to_numpy(dtype=bool)
        if setting.params["mode"] == "zone":
            wt1 = indicator["wt1"].to_numpy(dtype=float)
            is_signal = ((cross_up & (wt1 < params.oversold))
                         | (cross_down & (wt1 > params.overbought)))
        else:
            is_signal = cross_up | cross_down
        direction = np.where(cross_up, 1.0, -1.0)

    elif setting.family == "squeeze":
        params = SqueezeParams(bb_length=setting.params["bb_length"],
                               kc_length=setting.params["kc_length"],
                               kc_mult=setting.params["kc_mult"])
        indicator = compute(data, params)
        sqz_on = indicator["sqz_on"].to_numpy(dtype=bool)
        sqz_off = indicator["sqz_off"].to_numpy(dtype=bool)
        blue_bars = int(indicator["no_sqz"].sum())
        # 解放は原典どおり「黒 -> 灰」のみ (Phase 1 追試で確定した定義)
        is_signal = np.zeros(context.bar_count, dtype=bool)
        is_signal[1:] = sqz_off[1:] & sqz_on[:-1]
        val = indicator["val"].to_numpy(dtype=float)
        direction = np.where(val > 0, 1.0, -1.0)
        is_signal &= ~np.isnan(val)

    else:
        raise ValueError(f"未知の族: {setting.family}")

    is_signal = is_signal & context.valid
    return np.flatnonzero(is_signal), direction, blue_bars


def evaluate_setting(setting: Setting, contexts: list[MarketContext]) -> list[dict]:
    """1 設定 x 1 時間足 を 6 ホライズンで評価する (銘柄はプール)。

    Part A では信頼区間を出さない。候補を選ぶだけなので平均と順位で足りるし、
    ブートストラップを 1,170 セルぶん回すのは無駄が大きい。
    """
    interval = contexts[0].interval
    per_symbol: dict[str, dict[int, np.ndarray]] = {}
    blue_total = 0
    for context in contexts:
        index, direction, blue = make_signals(setting, context)
        blue_total += blue
        per_symbol[context.symbol] = {
            horizon: direction[index] * context.forward_atr[horizon][index]
            for horizon in HORIZONS
        }
        per_symbol[context.symbol]["_index"] = index
        per_symbol[context.symbol]["_bps"] = {
            horizon: direction[index] * context.forward_bps[horizon][index]
            for horizon in HORIZONS
        }

    rows = []
    for horizon in HORIZONS:
        pooled_atr, pooled_bps, symbol_means = [], [], {}
        for context in contexts:
            values = per_symbol[context.symbol][horizon]
            values = values[~np.isnan(values)]
            bps_values = per_symbol[context.symbol]["_bps"][horizon]
            bps_values = bps_values[~np.isnan(bps_values)]
            pooled_atr.append(values)
            pooled_bps.append(bps_values)
            symbol_means[context.symbol] = float(values.mean()) if len(values) else np.nan
        pooled_atr = np.concatenate(pooled_atr)
        pooled_bps = np.concatenate(pooled_bps)
        signs = [np.sign(v) for v in symbol_means.values() if not np.isnan(v)]

        rows.append({
            "setting_key": setting.key,
            "family": setting.family,
            "label": setting.label,
            "interval": interval,
            "horizon": horizon,
            "n_signals": len(pooled_atr),
            "mean_atr": float(pooled_atr.mean()) if len(pooled_atr) else np.nan,
            "mean_bps": float(pooled_bps.mean()) if len(pooled_bps) else np.nan,
            "win_rate": float((pooled_atr > 0).mean()) if len(pooled_atr) else np.nan,
            "signs_agree": bool(len(set(signs)) == 1) if len(signs) == 3 else False,
            "blue_bars": blue_total,
            **{f"mean_atr_{s}": v for s, v in symbol_means.items()},
        })
    return rows
