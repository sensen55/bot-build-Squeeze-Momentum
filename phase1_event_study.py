"""Phase 1: スクイーズ解放イベントのイベントスタディ。

目的
----
バックテストエンジンを書く**前に**、生のリターン分布だけで
エッジの有無を判定する。売買ロジックを作り込むと自由度が増え、
「たまたま勝つ組み合わせ」が必ず見つかってしまうため。

検証する帰無仮説 (依頼 0.4)
---------------------------
「スクイーズ解放時点の val の符号は、その後のリターンの符号について
  情報を持つか？」

ルックアヘッドを構造的に防ぐ設計
--------------------------------
イベント (解放バー) の添字を t とすると:

  判断に使える情報 : sqz_on[t], val[t], atr_norm[t], squeeze_run[t-1]
                     ← すべて t の終値が確定した時点で既知
  エントリー価格   : open[t+1]      ← t の終値確定後に到来する価格
  決済価格         : close[t+N]

判断時刻 (t の終値) と約定時刻 (t+1 の始値) が必ず 1 バー以上離れる。
この分離はコード上で `entry_index = event_index + 1` として固定されており、
オフバイワンで詰めることができない形にしてある。

将来リターンの生成にのみ未来方向の参照を使う。これは目的変数であって
特徴量ではないので問題ない (skill 2.5)。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.fetch_binance_klines import load_klines, INTERVAL_MINUTES
from squeeze_momentum import SqueezeParams, compute

# ---------------------------------------------------------------------------
# 事前確定した設定 (実行後に変更しないこと)
# ---------------------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["5m", "15m", "1h"]

STUDY_START = "2025-01-01"
STUDY_END = "2026-06-01"          # Hold-out (2026-06-01..2026-08-31) は封印

HORIZONS = [1, 3, 6, 12, 24, 48]
EXTRA_HORIZONS = [36, 60, 72]  # 近傍ホライズンの連続性確認用
RUN_BUCKETS = [1, 3, 5, 10, 20]

# 解放イベントの定義。既定は原典どおりの "gray" (黒 -> 灰)。
RELEASE_DEFINITIONS = {
    "any": "黒 -> 黒以外 (灰 + 青)。旧定義",
    "gray": "黒 -> 灰 (sqz_off)。原典 Carter / LazyBear の推奨",
    "blue": "黒 -> 青 (no_sqz)。gray から除外された側",
}

# COSTS.md より。判定に使うのは最良シナリオ F。
COST_FLOOR_BPS = 1.0              # Lighter 0% + 穏やか局面の実測スリッページ
COST_BASE_BPS = 2.0
COST_HIGH_BPS = 5.0
COST_HYPERLIQUID_BPS = 11.0

# 主検定は 3 時間足 x 6 ホライズン = 18 セル。Bonferroni で補正する。
PRIMARY_CELL_COUNT = len(INTERVALS) * len(HORIZONS)
ALPHA = 0.05
ALPHA_BONFERRONI = ALPHA / PRIMARY_CELL_COUNT

BOOTSTRAP_ROUNDS = 10_000
RANDOM_SEED = 20260908


@dataclass
class EventTable:
    """1 つの (銘柄, 時間足) から取り出したイベントとリターン。"""

    symbol: str
    interval: str
    definition: str                 # 解放イベントの定義 ("gray" / "blue" / "any")
    events: pd.DataFrame            # 1 行 = 1 イベント
    baseline: pd.DataFrame          # 1 行 = 全バー (ベースライン分布 A)
    bar_move_bps: float             # 1 バーの |リターン| 中央値 (skill 3.7)


def _assert_contiguous(index: pd.DatetimeIndex, interval: str) -> None:
    """バーが等間隔で欠損なく並んでいることを確認する。

    添字演算 (t+1, t+N) が実時間の +1 バー / +N バーと一致することの前提。
    欠損があると「48 本後」が実際には 3 時間後だったりする。
    """
    step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    gaps = index.to_series().diff().dropna()
    bad = gaps[gaps != step]
    if len(bad):
        raise ValueError(
            f"バーが等間隔ではありません ({interval}): 異常な間隔 {len(bad)} 箇所。"
            f" 例: {bad.head(3).to_dict()}"
        )


def build_event_table(
    symbol: str,
    interval: str,
    definition: str = "gray",
) -> EventTable:
    data, report = load_klines(symbol, interval, STUDY_START, STUDY_END, verbose=False)
    if report.missing_count or report.duplicated_times:
        raise ValueError(f"{symbol} {interval}: 欠損/重複あり。先に調査すること。")
    _assert_contiguous(data.index, interval)

    indicator = compute(data, SqueezeParams())
    open_prices = data["open"].to_numpy(dtype=float)
    close_prices = data["close"].to_numpy(dtype=float)
    val = indicator["val"].to_numpy(dtype=float)
    val_norm = indicator["val_norm"].to_numpy(dtype=float)
    atr = indicator["atr_norm"].to_numpy(dtype=float)
    sqz_on = indicator["sqz_on"].to_numpy(dtype=bool)
    sqz_off = indicator["sqz_off"].to_numpy(dtype=bool)
    no_sqz = indicator["no_sqz"].to_numpy(dtype=bool)
    squeeze_run = indicator["squeeze_run"].to_numpy(dtype=int)
    val_rising = indicator["val_rising"].to_numpy(dtype=bool)
    bar_count = len(data)

    # --- 将来リターン (目的変数) -------------------------------------------
    # エントリー = open[t+1], 決済 = close[t+N]。
    # t+1 も t+N も t より必ず未来なので、これは目的変数であり特徴量ではない。
    forward_bps: dict[int, np.ndarray] = {}
    forward_atr: dict[int, np.ndarray] = {}
    for horizon in HORIZONS + EXTRA_HORIZONS:
        bps = np.full(bar_count, np.nan)
        in_atr = np.full(bar_count, np.nan)
        last = bar_count - horizon - 1        # t+horizon <= bar_count-1 かつ t+1 が存在
        if last > 0:
            entry = open_prices[1: last + 2]
            exit_ = close_prices[horizon: last + horizon + 1]
            bps[: last + 1] = (exit_ / entry - 1.0) * 10_000.0
            in_atr[: last + 1] = (exit_ - entry) / atr[: last + 1]
        forward_bps[horizon] = bps
        forward_atr[horizon] = in_atr

    # 1 バーの値動きとコストの比 (skill 3.7)
    one_bar_bps = np.abs((close_prices[1:] / close_prices[:-1] - 1.0) * 10_000.0)
    bar_move_bps = float(np.nanmedian(one_bar_bps))

    # --- ベースライン分布 (A) ------------------------------------------------
    # 依頼書は「全バーからランダム抽出」だが、全バーをそのまま使う。
    # 全バーは母集団そのものであり、ランダム抽出はその部分集合にすぎない。
    # 抽出しない方が推定精度が高く、抽出の乱数次第で結論が動く余地も無い。
    warmup = SqueezeParams().warmup_bars
    baseline_mask = np.zeros(bar_count, dtype=bool)
    baseline_mask[warmup:] = True
    baseline_mask &= ~np.isnan(val) & ~np.isnan(atr)

    baseline = pd.DataFrame({"index": np.flatnonzero(baseline_mask)})
    baseline["val"] = val[baseline["index"]]
    baseline["val_sign"] = np.sign(baseline["val"]).replace(0.0, 1.0)
    for horizon in HORIZONS:
        baseline[f"ret_{horizon}_bps"] = forward_bps[horizon][baseline["index"]]
        baseline[f"ret_{horizon}_atr"] = forward_atr[horizon][baseline["index"]]

    # --- イベント抽出 --------------------------------------------------------
    # 解放バー t は、直前バーが黒 (sqz_on) で、当該バーで黒でなくなったバー。
    # 「黒でなくなった」の中身を 3 通りに分けられるようにしてある:
    #
    #   "gray" : sqz_off[t] == True   黒 -> 灰。原典 (Carter / LazyBear) の推奨。
    #                                 「黒の後、最初の灰でエントリー」
    #   "blue" : no_sqz[t] == True    黒 -> 青。gray から除外された側
    #   "any"  : ~sqz_on[t]           黒 -> 黒以外。gray + blue (旧定義)
    #
    # 継続本数は t-1 時点の連続本数 (= 解放直前までの収縮の長さ)。
    if definition == "gray":
        became = sqz_off
    elif definition == "blue":
        became = no_sqz
    elif definition == "any":
        became = ~sqz_on
    else:
        raise ValueError(f"未知の定義: {definition}")

    release = np.zeros(bar_count, dtype=bool)
    release[1:] = became[1:] & sqz_on[:-1]
    release &= baseline_mask

    event_index = np.flatnonzero(release)
    events = pd.DataFrame({"index": event_index})
    events["time"] = data.index[event_index]
    events["utc_hour"] = events["time"].dt.hour
    events["squeeze_run"] = squeeze_run[event_index - 1]     # 解放直前の収縮本数
    events["val"] = val[event_index]
    events["val_norm"] = val_norm[event_index]
    events["val_sign"] = np.sign(events["val"]).replace(0.0, 1.0)
    events["val_rising"] = val_rising[event_index]
    for horizon in HORIZONS + EXTRA_HORIZONS:
        events[f"ret_{horizon}_bps"] = forward_bps[horizon][event_index]
        events[f"ret_{horizon}_atr"] = forward_atr[horizon][event_index]

    return EventTable(symbol, interval, definition, events, baseline, bar_move_bps)


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def moving_block_bootstrap_mean_ci(
    values: np.ndarray,
    block_length: int,
    rounds: int = BOOTSTRAP_ROUNDS,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """移動ブロックブートストラップで平均の 95% 信頼区間を出す。

    なぜ iid ブートストラップではダメか (skill Phase 5 の NG パターン):
    イベントの将来リターン窓は互いに重なりうるし、ボラティリティには
    自己相関がある。iid で復元抽出すると独立性を仮定してしまい、
    信頼区間が実際より狭く出る (= 有意に見えやすくなる)。
    時間順に並べたまま連続ブロックで抜き出すことで自己相関を保つ。
    """
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 2 * block_length or n == 0:
        return (np.nan, np.nan)

    rng = np.random.default_rng(seed)
    block_count = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    means = np.empty(rounds)

    # メモリを抑えるため rounds を分割して処理する (n が大きいと一括では乗らない)
    chunk = max(1, int(5_000_000 / max(block_count * block_length, 1)))
    done = 0
    while done < rounds:
        size = min(chunk, rounds - done)
        starts = rng.integers(0, n - block_length + 1, size=(size, block_count))
        picked = (starts[:, :, None] + offsets).reshape(size, -1)[:, :n]
        means[done: done + size] = values[picked].mean(axis=1)
        done += size
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def select_non_overlapping(event_index: np.ndarray, horizon: int) -> np.ndarray:
    """将来リターン窓が重ならないイベントだけを貪欲に選ぶ。

    重なったイベント同士のリターンは同じ値動きを共有するため、
    t 検定の「独立標本」仮定が壊れる。窓を重ねない部分集合なら
    その仮定が概ね成り立つ。
    """
    keep: list[int] = []
    next_allowed = -1
    for position, index in enumerate(event_index):
        if index >= next_allowed:
            keep.append(position)
            next_allowed = index + horizon + 1
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------
def analyse_volatility_expansion(tables: list[EventTable], horizon: int) -> dict:
    """(B) vs (A): スクイーズ解放は「値幅の拡大」を予測するか。

    方向ではなく大きさを見る。分散比と、Brown-Forsythe 検定
    (中央値を中心とした Levene 検定) を使う。
    暗号資産のリターンは裾が厚く正規分布から遠いので、
    正規性を仮定する F 検定は使わない。
    """
    event_returns = np.concatenate(
        [t.events[f"ret_{horizon}_bps"].dropna().to_numpy() for t in tables])
    baseline_returns = np.concatenate(
        [t.baseline[f"ret_{horizon}_bps"].dropna().to_numpy() for t in tables])

    statistic, p_value = stats.levene(event_returns, baseline_returns, center="median")
    return {
        "horizon": horizon,
        "n_events": len(event_returns),
        "n_baseline": len(baseline_returns),
        "event_std_bps": float(event_returns.std(ddof=1)),
        "baseline_std_bps": float(baseline_returns.std(ddof=1)),
        "std_ratio": float(event_returns.std(ddof=1) / baseline_returns.std(ddof=1)),
        "event_median_abs_bps": float(np.median(np.abs(event_returns))),
        "baseline_median_abs_bps": float(np.median(np.abs(baseline_returns))),
        "median_abs_ratio": float(
            np.median(np.abs(event_returns)) / np.median(np.abs(baseline_returns))),
        "levene_stat": float(statistic),
        "levene_p": float(p_value),
    }


def analyse_direction(
    tables: list[EventTable],
    horizon: int,
    condition: str = "sign",
) -> dict:
    """(C): val の符号方向のリターンが 0 より大きいか。

    condition:
      "sign"        - val の符号だけで方向を決める (主検定)
      "sign_accel"  - 符号 + 加速 (val>0 かつ上昇 / val<=0 かつ下降) のみ採用 (副次)
    """
    parts_returns: list[np.ndarray] = []
    parts_index: list[np.ndarray] = []
    for table in tables:
        events = table.events
        mask = events[f"ret_{horizon}_bps"].notna()
        if condition == "sign_accel":
            accelerating = (
                ((events["val"] > 0) & events["val_rising"])
                | ((events["val"] <= 0) & ~events["val_rising"])
            )
            mask &= accelerating
        selected = events[mask]
        parts_returns.append(
            (selected["val_sign"] * selected[f"ret_{horizon}_bps"]).to_numpy())
        parts_index.append(selected["index"].to_numpy())

    signed = np.concatenate(parts_returns)
    n = len(signed)
    if n < 2:
        return {"horizon": horizon, "condition": condition, "n": n}

    mean = float(signed.mean())
    std = float(signed.std(ddof=1))
    t_stat, p_value = stats.ttest_1samp(signed, 0.0)

    block = max(10, int(round(n ** (1 / 3))))
    ci_low, ci_high = moving_block_bootstrap_mean_ci(signed, block)

    # 窓が重ならない部分集合での再検定 (独立標本仮定を満たすため)
    non_overlap_parts = []
    for table, index_array in zip(tables, parts_index):
        keep = select_non_overlapping(index_array, horizon)
        events = table.events.set_index("index")
        subset = events.loc[index_array[keep]]
        non_overlap_parts.append(
            (subset["val_sign"] * subset[f"ret_{horizon}_bps"]).to_numpy())
    non_overlap = np.concatenate(non_overlap_parts)
    non_overlap = non_overlap[~np.isnan(non_overlap)]
    if len(non_overlap) >= 2:
        no_t, no_p = stats.ttest_1samp(non_overlap, 0.0)
        no_mean = float(non_overlap.mean())
    else:
        no_t, no_p, no_mean = np.nan, np.nan, np.nan

    return {
        "horizon": horizon,
        "condition": condition,
        "n": n,
        "mean_bps": mean,
        "std_bps": std,
        "median_bps": float(np.median(signed)),
        "stderr_bps": float(std / np.sqrt(n)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_low_bps": ci_low,
        "ci_high_bps": ci_high,
        "win_rate": float((signed > 0).mean()),
        "n_non_overlap": int(len(non_overlap)),
        "non_overlap_mean_bps": no_mean,
        "non_overlap_t": float(no_t),
        "non_overlap_p": float(no_p),
    }


def analyse_baseline_direction(tables: list[EventTable], horizon: int) -> dict:
    """対照: 「解放バーに限らず全バーで」val の符号方向リターンを取ったらどうか。

    これが解放バーと同程度なら、「スクイーズ解放」というイベントには
    何の付加価値も無く、単に val の符号を見ているだけということになる。
    """
    signed = np.concatenate([
        (t.baseline["val_sign"] * t.baseline[f"ret_{horizon}_bps"]).dropna().to_numpy()
        for t in tables
    ])
    t_stat, p_value = stats.ttest_1samp(signed, 0.0)
    return {
        "horizon": horizon,
        "n": len(signed),
        "mean_bps": float(signed.mean()),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def analyse_run_buckets(tables: list[EventTable], horizon: int) -> pd.DataFrame:
    """継続本数 n ごとの集計 (副次的な記述統計)。

    「長い収縮ほど値幅が大きい」という仮説を見るためのもの。
    ここから合格を主張しないこと (多重比較になる)。
    """
    rows = []
    combined = pd.concat([t.events for t in tables], ignore_index=True)
    for minimum_run in RUN_BUCKETS:
        subset = combined[combined["squeeze_run"] >= minimum_run]
        returns = subset[f"ret_{horizon}_bps"].dropna()
        signed = (subset["val_sign"] * subset[f"ret_{horizon}_bps"]).dropna()
        if len(signed) < 2:
            continue
        rows.append({
            "min_run": minimum_run,
            "n": len(signed),
            "median_abs_bps": float(np.median(np.abs(returns))),
            "signed_mean_bps": float(signed.mean()),
            "signed_t": float(stats.ttest_1samp(signed, 0.0).statistic),
            "signed_p": float(stats.ttest_1samp(signed, 0.0).pvalue),
        })
    return pd.DataFrame(rows)


def analyse_hour_of_day(tables: list[EventTable], horizon: int) -> pd.DataFrame:
    """UTC 時間帯別のイベント発生数とリターン。

    暗号資産は 24/365 でボラティリティに日内周期がある。
    特定の時間帯にイベントが偏っていたら、
    「スクイーズを検出した」のではなく「静かな時間帯を検出した」だけの可能性がある。
    """
    combined = pd.concat([t.events for t in tables], ignore_index=True)
    combined["signed"] = combined["val_sign"] * combined[f"ret_{horizon}_bps"]
    grouped = combined.groupby("utc_hour").agg(
        n=("signed", "size"),
        signed_mean_bps=("signed", "mean"),
        median_abs_bps=(f"ret_{horizon}_bps", lambda s: float(np.median(np.abs(s.dropna())))),
    )
    grouped["share"] = grouped["n"] / grouped["n"].sum()
    uniform = 1.0 / 24.0
    grouped["vs_uniform"] = grouped["share"] / uniform
    return grouped.reset_index()
