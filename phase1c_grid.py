"""Phase 1c: パラメータグリッドの定義と、候補選定ルール。

**このファイルはグリッドを回す前に確定・コミットする。**
選定ルールを結果を見てから決めると、選定ルール自体が探索対象になってしまう。

データの分離 (最重要)
---------------------
Part A (探索): 2025-01-01 .. 2026-05-31。既に何度も見ている期間なので、
               ここで出た数字に統計的な意味は無い。候補を選ぶためだけに使う。
Part B (確認): 2023-01-01 .. 2024-12-31。一度も取得していない期間。
               エッジの有無を判定するのはここだけ。
Hold-out     : 2026-06-01 .. 2026-08-31。今回も開封しない。

なぜ近傍安定性を必須条件にするのか
----------------------------------
65 設定 x 3 時間足 x 6 ホライズン = 1,170 セルを見る。
全設定が本当に無価値でも、偶然だけで p < 0.05 のセルが約 59 個出る規模である。
「一番良かった設定」を選ぶだけでは、その 59 個から一番運の良いものを拾うだけになる。

本物の効果なら、パラメータを少し動かしても急には消えないはず。
逆に運で当たった設定は「隣が全部悪いのに 1 点だけ良い」孤立した山になる。
そこで **自分のスコアではなく隣のスコア** を見る。
こうすると自分が偶然良かっただけの設定は落ちる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 期間 (事前確定)
# ---------------------------------------------------------------------------
PART_A_START = "2025-01-01"
PART_A_END = "2026-06-01"      # 探索用。既に何度も見ている
PART_B_START = "2023-01-01"
PART_B_END = "2025-01-01"      # 確認用。未使用
HOLDOUT_START = "2026-06-01"   # 今回も開封しない
HOLDOUT_END = "2026-09-01"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVALS = ["5m", "15m", "1h"]
HORIZONS = [1, 3, 6, 12, 24, 48]

# EMA の立ち上がりを避けるため、全族で同じ本数を除外する。
# 族ごとに変えると、比較しているバー集合が違ってしまいランクが公平でなくなる。
WARMUP_BARS = 500

COST_FLOOR_BPS = 1.0           # COSTS.md シナリオ F (最良)

# 選定ルールの定数 (事前確定)
NEIGHBOUR_TOP_FRACTION = 0.30  # 隣のスコア中央値がこの上位割合に入ること
MAX_CANDIDATES_FROM_GRID = 3   # グリッドから選ぶ候補の上限 (族をまたいで合計)

# Phase 1b の副次系列。探索とは独立に発生したので選定ルールを経由せず無条件で追加する。
UNCONDITIONAL_CANDIDATE = {
    "family": "adx",
    "params": {"length": 14, "threshold": 25.0},
    "interval": "15m",
    "origin": "Phase 1b の副次系列 (選定ルールを経由しない無条件追加)",
}


# ---------------------------------------------------------------------------
# グリッド定義
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    """1 つのパラメータ設定。

    axis_index は「グリッド上の座標」。近傍 (±1 段階) の判定に使う。
    """

    family: str
    params: dict
    axis_index: tuple[int, ...]

    @property
    def label(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.params.items())

    @property
    def key(self) -> str:
        return f"{self.family}[{self.label}]"


ADX_LENGTHS = [7, 10, 14, 20, 28]
ADX_THRESHOLDS = [15.0, 20.0, 25.0, 30.0]

WT_CHANNEL_LENGTHS = [6, 10, 15]
WT_AVERAGE_LENGTHS = [12, 21, 34]
WT_SIGNAL_MODES: list[Literal["all", "zone"]] = ["all", "zone"]

SQZ_BB_LENGTHS = [14, 20, 30]
SQZ_KC_LENGTHS = [14, 20, 30]
SQZ_KC_MULTS = [1.0, 1.5, 2.0]


def build_grid() -> list[Setting]:
    """65 設定 (ADX 20 + WaveTrend 18 + Squeeze 27)。"""
    settings: list[Setting] = []

    for i, length in enumerate(ADX_LENGTHS):
        for j, threshold in enumerate(ADX_THRESHOLDS):
            settings.append(Setting("adx",
                                    {"length": length, "threshold": threshold},
                                    (i, j)))

    for i, n1 in enumerate(WT_CHANNEL_LENGTHS):
        for j, n2 in enumerate(WT_AVERAGE_LENGTHS):
            for k, mode in enumerate(WT_SIGNAL_MODES):
                settings.append(Setting("wavetrend",
                                        {"n1": n1, "n2": n2, "mode": mode},
                                        (i, j, k)))

    for i, bb in enumerate(SQZ_BB_LENGTHS):
        for j, kc in enumerate(SQZ_KC_LENGTHS):
            for k, mult in enumerate(SQZ_KC_MULTS):
                settings.append(Setting("squeeze",
                                        {"bb_length": bb, "kc_length": kc,
                                         "kc_mult": mult},
                                        (i, j, k)))

    return settings


def neighbours(setting: Setting, grid: list[Setting]) -> list[Setting]:
    """グリッド上で「各パラメータ ±1 段階」だけ離れた設定を返す。

    同じ族の中で、軸の添字がちょうど 1 つだけ 1 だけ違うものが隣。
    (斜め = 2 軸同時に動いたものは隣に含めない)
    """
    result = []
    for other in grid:
        if other.family != setting.family or other.key == setting.key:
            continue
        differences = [abs(a - b) for a, b in zip(setting.axis_index, other.axis_index)]
        if sum(d > 0 for d in differences) == 1 and max(differences) == 1:
            result.append(other)
    return result


# ---------------------------------------------------------------------------
# スコアリングと選定ルール (グリッドを回す前に確定させる)
# ---------------------------------------------------------------------------
def compute_scores(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """各 (設定, 時間足) のスコア = 6 ホライズンの順位の平均。

    results は 1 行 = (setting_key, family, interval, horizon, mean_atr, ...)。

    順位は「同じ時間足・同じホライズンの中で、65 設定を符号方向リターン
    (ATR 単位) の降順に並べたときの順位」。1 位が最良。
    ホライズンごとにスケールが違うので、生の値ではなく順位で揃える。
    スコアは小さいほど良い。
    """
    scored = results.copy()
    scored["rank_in_cell"] = (
        scored.groupby(["interval", "horizon"])["mean_atr"]
        .rank(ascending=False, method="average")
    )
    score = (
        scored.groupby(["setting_key", "family", "interval"])["rank_in_cell"]
        .mean()
        .reset_index(name="score")
    )
    # スコアの順位パーセンタイル (時間足内で 65 設定を比較)。0 に近いほど良い。
    score["score_percentile"] = (
        score.groupby("interval")["score"].rank(pct=True, ascending=True)
    )
    return score, scored


def best_horizon(results: pd.DataFrame, setting_key: str, interval: str) -> int:
    """その (設定, 時間足) で順位が最も良かったホライズンを機械的に選ぶ。

    Part B に持ち込むホライズンはここで決め、Part B の結果を見て変えない。
    """
    scoped = results[(results["setting_key"] == setting_key)
                     & (results["interval"] == interval)]
    return int(scoped.loc[scoped["rank_in_cell"].idxmin(), "horizon"])


def apply_selection_rules(
    results: pd.DataFrame,
    scores: pd.DataFrame,
    grid: list[Setting],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """選定ルールを適用し、(通過した候補, 全設定の判定内訳) を返す。

    必須条件:
      (2) 近傍安定性 — 隣接設定のスコアの中央値が、その時間足の上位 30% に入る
      (3) 3 銘柄で符号が一致 (判定に使うホライズンにおいて)

    条件を満たしたものをスコア順に並べ、上位 MAX_CANDIDATES_FROM_GRID を採る。
    """
    grid_by_key = {setting.key: setting for setting in grid}
    score_lookup = {(row.setting_key, row.interval): row.score
                    for row in scores.itertuples()}

    rows = []
    for row in scores.itertuples():
        setting = grid_by_key[row.setting_key]
        neighbour_scores = [
            score_lookup[(other.key, row.interval)]
            for other in neighbours(setting, grid)
            if (other.key, row.interval) in score_lookup
        ]
        neighbour_median = float(np.median(neighbour_scores)) if neighbour_scores else np.nan

        horizon = best_horizon(results, row.setting_key, row.interval)
        cell = results[(results["setting_key"] == row.setting_key)
                       & (results["interval"] == row.interval)
                       & (results["horizon"] == horizon)].iloc[0]

        rows.append({
            "setting_key": row.setting_key,
            "family": row.family,
            "interval": row.interval,
            "score": row.score,
            "score_percentile": row.score_percentile,
            "n_neighbours": len(neighbour_scores),
            "neighbour_median_score": neighbour_median,
            "best_horizon": horizon,
            "mean_atr": cell["mean_atr"],
            "mean_bps": cell["mean_bps"],
            "n_signals": cell["n_signals"],
            "signs_agree": bool(cell["signs_agree"]),
        })

    breakdown = pd.DataFrame(rows)
    # 近傍中央値を時間足内でパーセンタイル化してから 30% で切る
    breakdown["neighbour_percentile"] = (
        breakdown.groupby("interval")["neighbour_median_score"]
        .rank(pct=True, ascending=True)
    )
    breakdown["passes_neighbour"] = breakdown["neighbour_percentile"] <= NEIGHBOUR_TOP_FRACTION
    breakdown["passes_signs"] = breakdown["signs_agree"]
    breakdown["passes_all"] = breakdown["passes_neighbour"] & breakdown["passes_signs"]

    qualified = (
        breakdown[breakdown["passes_all"]]
        .sort_values("score")
        .head(MAX_CANDIDATES_FROM_GRID)
        .reset_index(drop=True)
    )
    return qualified, breakdown


def plateau_metric(scores: pd.DataFrame, grid: list[Setting]) -> pd.DataFrame:
    """族ごとに「平らな丘か、孤立した山か」を数値化する。

    自分のスコアと隣のスコア中央値の Spearman 相関を見る。
    1 に近い = 表面がなめらか (平らな丘がある)。
    0 に近い = 隣と無関係 (ノイズ。良い設定は孤立した山)。
    """
    from scipy import stats as scipy_stats

    grid_by_key = {setting.key: setting for setting in grid}
    score_lookup = {(row.setting_key, row.interval): row.score
                    for row in scores.itertuples()}

    rows = []
    for family in ("adx", "wavetrend", "squeeze"):
        for interval in INTERVALS:
            own, neighbour = [], []
            for row in scores.itertuples():
                if row.family != family or row.interval != interval:
                    continue
                setting = grid_by_key[row.setting_key]
                values = [score_lookup[(o.key, interval)]
                          for o in neighbours(setting, grid)
                          if (o.key, interval) in score_lookup]
                if values:
                    own.append(row.score)
                    neighbour.append(float(np.median(values)))
            correlation, p_value = scipy_stats.spearmanr(own, neighbour)
            rows.append({"family": family, "interval": interval,
                         "n_settings": len(own),
                         "spearman": float(correlation), "p_value": float(p_value)})
    return pd.DataFrame(rows)
