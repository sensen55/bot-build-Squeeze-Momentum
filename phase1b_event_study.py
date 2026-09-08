"""Phase 1b: 「主役 = WaveTrend / ADX、補助 = スクイーズ」仮説の検証。

Phase 1 とは別の仮説であり、Phase 1 の結論を覆すための再挑戦ではない。
Phase 1 の判定・パラメータ・データ期間は一切変更しない。

問 1: 主役 (WaveTrend / ADX) は単体で方向情報を持つか
問 2: スクイーズは主役を改善するか (問 1 が Yes の主役についてのみ)

Phase 1 のレビューを踏まえた変更点
----------------------------------
主検定は **ATR 正規化リターン** で行う。Phase 1 では bps でプールしたため、
値動きの大きい SOL に平均が引っ張られた (1h/N=48 の 35.2 bps は
SOL 単独で 78.8 bps だった)。ATR 単位に揃えれば、
「その銘柄の普段の値動きの何倍動いたか」で公平に比較できる。
コスト判定のときだけ bps に戻す。

ルックアヘッドを構造的に防ぐ設計 (Phase 1 と同じ)
-------------------------------------------------
シグナルバーの添字を t とすると:

  判断に使える情報 : wt1[t], wt2[t], adx[t], plus_di[t], sqz_on[t], atr_norm[t]
                     ← すべて t の終値が確定した時点で既知
  エントリー価格   : open[t+1]      ← t の終値確定後に到来する価格
  決済価格         : close[t+N]

`entry_index = signal_index + 1` として固定しており、詰めることができない。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.fetch_binance_klines import INTERVAL_MINUTES, load_klines
from main_indicators import WARMUP_BARS, AdxParams, WaveTrendParams, adx, wavetrend
from squeeze_momentum import SqueezeParams, compute

# ---------------------------------------------------------------------------
# 事前確定した設定 (実行後に変更しないこと)
# ---------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["5m", "15m", "1h"]
HORIZONS = [1, 3, 6, 12, 24, 48]

STUDY_START = "2025-01-01"
STUDY_END = "2026-06-01"          # Hold-out (2026-06-01..2026-08-31) は封印

LEADERS = ["wavetrend", "adx"]

# 問 2 で「解放直後」とみなす本数。事前確定。3 や 12 を後から選び直さない。
SQUEEZE_LOOKBACK_K = 6
DESCRIPTIVE_LOOKBACKS = [3, 6, 12]   # 記述統計としてのみ並べる。判定には使わない

# COSTS.md より。判定に使うのは最良シナリオ F。
COST_FLOOR_BPS = 1.0

# 問 1: 2 主役 x 3 時間足 x 6 ホライズン = 36 セル
Q1_CELL_COUNT = len(LEADERS) * len(INTERVALS) * len(HORIZONS)
ALPHA = 0.05
ALPHA_Q1 = ALPHA / Q1_CELL_COUNT
# 問 2: 1 主役あたり 3 時間足 x 6 ホライズン = 18 セル
Q2_CELL_COUNT = len(INTERVALS) * len(HORIZONS)
ALPHA_Q2 = ALPHA / Q2_CELL_COUNT

BOOTSTRAP_ROUNDS = 10_000
RANDOM_SEED = 20260908


@dataclass
class SignalTable:
    """1 つの (主役, 銘柄, 時間足) から取り出したシグナルとリターン。"""

    leader: str
    symbol: str
    interval: str
    variant: str                # "primary" (判定に使う) / "secondary" (記述統計のみ)
    signals: pd.DataFrame       # 1 行 = 1 シグナル
    bar_count: int
    zero_division_bars: int


# ---------------------------------------------------------------------------
# ブートストラップ (Phase 1 と同じ推定量を、ブロック平均経由で高速に計算する)
# ---------------------------------------------------------------------------
def block_bootstrap_mean_ci(
    values: np.ndarray,
    block_length: int,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """移動ブロックブートストラップで平均の 95% 信頼区間を出す。

    なぜ iid ブートストラップではダメか:
    シグナルの将来リターン窓は互いに重なるし、ボラティリティには自己相関がある。
    iid で復元抽出すると独立性を仮定してしまい、信頼区間が実際より狭く出る
    (= 有意に見えやすくなる)。時間順に並べたまま連続ブロックで抜き出す。

    高速化: 等長ブロックを繋げた標本の平均は「ブロック平均の平均」に等しい。
    先に全開始位置のブロック平均を rolling で求めておけば、
    1 ラウンドあたり block_count 個を引くだけで済む。
    Phase 1 の実装 (要素を毎回並べる版) と同じ推定量で、結果も一致する
    (tests/test_phase1b.py で確認)。
    """
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2 * block_length or n == 0:
        return (np.nan, np.nan)

    block_means = sliding_window_view(values, block_length).mean(axis=1)
    block_count = max(1, n // block_length)

    rng = np.random.default_rng(seed)
    means = np.empty(rounds)
    chunk = max(1, int(5_000_000 / block_count))
    done = 0
    while done < rounds:
        size = min(chunk, rounds - done)
        picked = rng.integers(0, len(block_means), size=(size, block_count))
        means[done: done + size] = block_means[picked].mean(axis=1)
        done += size
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def block_bootstrap_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """2 群の平均の差 (a - b) の 95% 信頼区間。各群を独立にブロック再標本化する。"""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 20 or len(b) < 20:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    samples = []
    for values in (a, b):
        block = max(10, int(round(len(values) ** (1 / 3))))
        block_means = sliding_window_view(values, block).mean(axis=1)
        block_count = max(1, len(values) // block)
        picked = rng.integers(0, len(block_means), size=(rounds, block_count))
        samples.append(block_means[picked].mean(axis=1))
    difference = samples[0] - samples[1]
    return (float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5)))


def select_non_overlapping(signal_index: np.ndarray, horizon: int) -> np.ndarray:
    """将来リターン窓が重ならないシグナルだけを貪欲に選ぶ。"""
    keep: list[int] = []
    next_allowed = -1
    for position, index in enumerate(signal_index):
        if index >= next_allowed:
            keep.append(position)
            next_allowed = index + horizon + 1
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# シグナル表の構築
# ---------------------------------------------------------------------------
def _assert_contiguous(index: pd.DatetimeIndex, interval: str) -> None:
    step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    gaps = index.to_series().diff().dropna()
    bad = gaps[gaps != step]
    if len(bad):
        raise ValueError(f"バーが等間隔ではありません ({interval}): {len(bad)} 箇所")


def build_signal_table(
    leader: str,
    symbol: str,
    interval: str,
    variant: str = "primary",
) -> SignalTable:
    """シグナル表を作る。

    variant:
      "primary"   - 事前確定した主シグナル定義。問 1 / 問 2 の判定に使う
      "secondary" - 依頼書の「副次（記述統計のみ）」。判定には使わない
                      WaveTrend: クロスが売られすぎ/買われすぎ帯で起きたものに限定
                      ADX      : 閾値 25 の上抜け（20 とは別のバーなので別集合）
    """
    data, report = load_klines(symbol, interval, STUDY_START, STUDY_END, verbose=False)
    if report.missing_count or report.duplicated_times:
        raise ValueError(f"{symbol} {interval}: 欠損/重複あり")
    _assert_contiguous(data.index, interval)

    squeeze = compute(data, SqueezeParams())
    open_prices = data["open"].to_numpy(dtype=float)
    close_prices = data["close"].to_numpy(dtype=float)
    atr = squeeze["atr_norm"].to_numpy(dtype=float)
    sqz_on = squeeze["sqz_on"].to_numpy(dtype=bool)
    sqz_off = squeeze["sqz_off"].to_numpy(dtype=bool)
    bar_count = len(data)

    # --- 主役ごとのシグナルと方向 ------------------------------------------
    if leader == "wavetrend":
        indicator = wavetrend(data, WaveTrendParams())
        cross_up = indicator["cross_up"].to_numpy(dtype=bool)
        cross_down = indicator["cross_down"].to_numpy(dtype=bool)
        wt1 = indicator["wt1"].to_numpy(dtype=float)
        # 副次: 売られすぎ帯での上抜け / 買われすぎ帯での下抜け。
        # これは主シグナルの部分集合になる。
        in_zone = (
            (cross_up & (wt1 < WaveTrendParams().oversold))
            | (cross_down & (wt1 > WaveTrendParams().overbought))
        )
        is_signal = in_zone if variant == "secondary" else (cross_up | cross_down)
        direction = np.where(cross_up, 1.0, -1.0)
        secondary = in_zone
        zero_division_bars = indicator.attrs["zero_division_bars"]
    elif leader == "adx":
        indicator = adx(data, AdxParams())
        bullish = indicator["bullish"].to_numpy(dtype=bool)
        direction = np.where(bullish, 1.0, -1.0)
        # 副次: 閾値 25 の上抜け。20 を上抜けるバーとは別のバーなので、
        # 主シグナルの部分集合ではなく独立した集合になる。
        secondary = indicator["cross_secondary"].to_numpy(dtype=bool)
        is_signal = (
            secondary if variant == "secondary"
            else indicator["cross_threshold"].to_numpy(dtype=bool)
        )
        zero_division_bars = indicator.attrs["zero_tr_bars"]
    else:
        raise ValueError(f"未知の主役: {leader}")

    # --- 将来リターン (目的変数) -------------------------------------------
    # エントリー = open[t+1], 決済 = close[t+N]。どちらも t より必ず未来。
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

    # --- スクイーズによる群分け (問 2 用) -----------------------------------
    # 解放バー: 直前が黒 (sqz_on) で当該バーが灰 (sqz_off)。Phase 1 追試の定義。
    release = np.zeros(bar_count, dtype=bool)
    release[1:] = sqz_off[1:] & sqz_on[:-1]

    recent_release: dict[int, np.ndarray] = {}
    for k in DESCRIPTIVE_LOOKBACKS:
        # 「直近 k 本以内に解放があった」= release[t-k+1 .. t] のどれかが True。
        # 現在バーを含む過去 k 本だけを見るので未来は参照しない。
        flags = np.zeros(bar_count, dtype=bool)
        cumulative = np.concatenate([[0], np.cumsum(release)])
        start = np.maximum(np.arange(bar_count) - k + 1, 0)
        flags = (cumulative[np.arange(bar_count) + 1] - cumulative[start]) > 0
        recent_release[k] = flags

    # --- シグナル抽出 --------------------------------------------------------
    valid = np.zeros(bar_count, dtype=bool)
    valid[WARMUP_BARS:] = True                 # EMA の立ち上がりを丸ごと除外
    valid &= ~np.isnan(atr)
    is_signal = is_signal & valid

    signal_index = np.flatnonzero(is_signal)
    signals = pd.DataFrame({"index": signal_index})
    signals["time"] = data.index[signal_index]
    signals["utc_hour"] = signals["time"].dt.hour
    signals["direction"] = direction[signal_index]
    signals["secondary"] = secondary[signal_index]
    signals["sqz_on"] = sqz_on[signal_index]
    for k in DESCRIPTIVE_LOOKBACKS:
        signals[f"recent_release_{k}"] = recent_release[k][signal_index]
    for horizon in HORIZONS:
        signals[f"ret_{horizon}_bps"] = forward_bps[horizon][signal_index]
        signals[f"ret_{horizon}_atr"] = forward_atr[horizon][signal_index]

    # 群分け: S1 (解放直後) > S2 (スクイーズ中) > S3 (それ以外) の優先順
    group = np.full(len(signals), "S3", dtype=object)
    group[signals["sqz_on"].to_numpy()] = "S2"
    group[signals[f"recent_release_{SQUEEZE_LOOKBACK_K}"].to_numpy()] = "S1"
    signals["group"] = group

    return SignalTable(leader, symbol, interval, variant, signals,
                       bar_count, zero_division_bars)


# ---------------------------------------------------------------------------
# 問 1: 主役単体の方向情報
# ---------------------------------------------------------------------------
def analyse_leader(tables: list[SignalTable], horizon: int) -> dict:
    """符号方向リターンが 0 より大きいか。主検定は ATR 単位。"""
    atr_parts, bps_parts, index_parts, symbol_means = [], [], [], {}
    for table in tables:
        selected = table.signals.dropna(subset=[f"ret_{horizon}_atr"])
        signed_atr = (selected["direction"] * selected[f"ret_{horizon}_atr"]).to_numpy()
        signed_bps = (selected["direction"] * selected[f"ret_{horizon}_bps"]).to_numpy()
        atr_parts.append(signed_atr)
        bps_parts.append(signed_bps)
        index_parts.append(selected["index"].to_numpy())
        symbol_means[table.symbol] = float(signed_atr.mean()) if len(signed_atr) else np.nan

    signed_atr = np.concatenate(atr_parts)
    signed_bps = np.concatenate(bps_parts)
    n = len(signed_atr)
    if n < 2:
        return {"horizon": horizon, "n": n}

    t_atr = stats.ttest_1samp(signed_atr, 0.0)
    block = max(10, int(round(n ** (1 / 3))))
    ci_low_atr, ci_high_atr = block_bootstrap_mean_ci(signed_atr, block)
    ci_low_bps, ci_high_bps = block_bootstrap_mean_ci(signed_bps, block)

    # 窓が重ならない部分集合での再検定 (独立標本仮定を満たすため)
    non_overlap = []
    for table, index_array in zip(tables, index_parts):
        keep = select_non_overlapping(index_array, horizon)
        subset = table.signals.set_index("index").loc[index_array[keep]]
        non_overlap.append(
            (subset["direction"] * subset[f"ret_{horizon}_atr"]).to_numpy())
    non_overlap = np.concatenate(non_overlap)
    non_overlap = non_overlap[~np.isnan(non_overlap)]
    if len(non_overlap) >= 2:
        t_no = stats.ttest_1samp(non_overlap, 0.0)
        no_mean, no_p, no_n = float(non_overlap.mean()), float(t_no.pvalue), len(non_overlap)
    else:
        no_mean, no_p, no_n = np.nan, np.nan, 0

    signs = [np.sign(v) for v in symbol_means.values() if not np.isnan(v)]
    return {
        "horizon": horizon,
        "n": n,
        "mean_atr": float(signed_atr.mean()),
        "median_atr": float(np.median(signed_atr)),
        "t_stat": float(t_atr.statistic),
        "p_value": float(t_atr.pvalue),
        "ci_low_atr": ci_low_atr,
        "ci_high_atr": ci_high_atr,
        "mean_bps": float(signed_bps.mean()),
        "ci_low_bps": ci_low_bps,
        "ci_high_bps": ci_high_bps,
        "win_rate": float((signed_atr > 0).mean()),
        **{f"mean_atr_{symbol}": value for symbol, value in symbol_means.items()},
        "signs_agree": bool(len(set(signs)) == 1) if signs else False,
        "n_non_overlap": no_n,
        "non_overlap_mean_atr": no_mean,
        "non_overlap_p": no_p,
    }


# ---------------------------------------------------------------------------
# 問 2: スクイーズ群による層別
# ---------------------------------------------------------------------------
def analyse_squeeze_groups(tables: list[SignalTable], horizon: int, k: int) -> dict:
    """S1 (解放直後) と S3 (それ以外) の符号方向リターンの差を検定する。"""
    column_atr = f"ret_{horizon}_atr"
    column_bps = f"ret_{horizon}_bps"
    combined = pd.concat([t.signals for t in tables], ignore_index=True)
    combined = combined.dropna(subset=[column_atr])

    group = np.full(len(combined), "S3", dtype=object)
    group[combined["sqz_on"].to_numpy()] = "S2"
    group[combined[f"recent_release_{k}"].to_numpy()] = "S1"
    combined["group_k"] = group
    combined["signed_atr"] = combined["direction"] * combined[column_atr]
    combined["signed_bps"] = combined["direction"] * combined[column_bps]

    parts = {name: combined[combined["group_k"] == name] for name in ("S1", "S2", "S3")}
    result = {"horizon": horizon, "k": k}
    for name, part in parts.items():
        result[f"n_{name}"] = len(part)
        result[f"mean_atr_{name}"] = float(part["signed_atr"].mean()) if len(part) else np.nan
        result[f"mean_bps_{name}"] = float(part["signed_bps"].mean()) if len(part) else np.nan

    s1, s3 = parts["S1"]["signed_atr"].to_numpy(), parts["S3"]["signed_atr"].to_numpy()
    if len(s1) < 20 or len(s3) < 20:
        result["difference_atr"] = np.nan
        result["welch_p"] = np.nan
        result["diff_ci_low"] = np.nan
        result["diff_ci_high"] = np.nan
        return result

    welch = stats.ttest_ind(s1, s3, equal_var=False)
    low, high = block_bootstrap_difference_ci(s1, s3)
    result["difference_atr"] = float(s1.mean() - s3.mean())
    result["welch_t"] = float(welch.statistic)
    result["welch_p"] = float(welch.pvalue)
    result["diff_ci_low"] = low
    result["diff_ci_high"] = high
    # S1 の絶対水準もコストを超えているか (S3 が大きくマイナスなだけ、を弾く)
    s1_bps = parts["S1"]["signed_bps"].to_numpy()
    block = max(10, int(round(len(s1_bps) ** (1 / 3))))
    result["s1_ci_low_bps"], result["s1_ci_high_bps"] = block_bootstrap_mean_ci(
        s1_bps, block)
    return result
