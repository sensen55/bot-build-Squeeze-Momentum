"""特徴量計算にルックアヘッドが無いことを機械的に検査する。

原理
----
ルックアヘッドが無い特徴量は「時刻 t の値が t 以前の入力だけの関数」である。
したがって、価格データを位置 c で切り捨てて計算し直しても、
位置 c 以前の出力は一切変わらないはずである。変わったなら、
元の計算は c より後 (= 未来) のデータを参照していたことになる。

切断位置の設計 (skill 6.4)
--------------------------
切断位置の置き方が検出力を決める。

| ルックアヘッドの種類       | 影響範囲             | 必要な切断位置        |
|--------------------------|--------------------|--------------------|
| 全期間統計 (.mean() 等)   | 全期間              | どこで切っても検出可能 |
| shift(-1)                | 常に 1 バー先        | どこで切っても検出可能 |
| bfill (疎な欠損)          | 欠損位置の 1 バー手前 | その 1 点に当たる必要 |

そこで「連続した切断位置のまとまり (run)」と「全体に散らした切断位置」を
両方使う。連続 run の長さを CONSECUTIVE_RUN_LENGTH とすると、
影響範囲の周期がこれ以下のパターンは必ずどれかに当たる。

検出できる範囲の限界 (重要)
---------------------------
「影響範囲の周期が CONSECUTIVE_RUN_LENGTH より長く、かつ極端に疎な」
ルックアヘッドは、散らした切断位置に偶然当たらなければ捕まらない。
つまり本検査は確率的な保証しか与えない。
**「検査に通った = 安全」と読まないこと。** これは
「よくある形のルックアヘッドは入っていない」を意味するに過ぎない。
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

CONSECUTIVE_RUN_LENGTH = 64
SCATTERED_CUT_COUNT = 40
TOLERANCE = 1e-9

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]


class LookaheadDetectedError(AssertionError):
    """切断テストで出力が変化した = 未来を参照している。"""


def _cut_positions(total_bars: int, warmup: int) -> list[int]:
    """切断位置の一覧。warmup 直後から末尾手前までを対象にする。"""
    first = max(warmup + 5, 10)
    last = total_bars - 1
    if last <= first:
        raise ValueError("データが短すぎて切断テストができません")

    positions: set[int] = set()

    # (1) 連続 run: 影響範囲が短いルックアヘッドを確実に捕まえる
    run_start = max(first, last - CONSECUTIVE_RUN_LENGTH)
    positions.update(range(run_start, last))

    # (2) 中間にもう一つ連続 run を置く (末尾付近だけだと偏るため)
    mid = (first + last) // 2
    positions.update(range(mid, min(mid + CONSECUTIVE_RUN_LENGTH, last)))

    # (3) 散らし: 影響範囲が全期間に及ぶルックアヘッドを捕まえる
    positions.update(np.linspace(first, last - 1, SCATTERED_CUT_COUNT).astype(int).tolist())

    return sorted(positions)


def _compare(full: pd.DataFrame, truncated: pd.DataFrame, cut: int) -> list[str]:
    """切断版と全体版の、位置 cut 未満の領域を突き合わせる。"""
    problems: list[str] = []
    for column in full.columns:
        a = full[column].iloc[:cut]
        b = truncated[column].iloc[:cut]
        if len(a) != len(b):
            problems.append(f"{column}: 長さ不一致 {len(a)} vs {len(b)}")
            continue

        a_values, b_values = a.to_numpy(), b.to_numpy()
        if a.dtype.kind in "fiu" and b.dtype.kind in "fiu":
            a_float = a_values.astype(float)
            b_float = b_values.astype(float)
            nan_mismatch = np.isnan(a_float) != np.isnan(b_float)
            both_valid = ~np.isnan(a_float) & ~np.isnan(b_float)
            value_mismatch = np.zeros_like(nan_mismatch)
            value_mismatch[both_valid] = ~np.isclose(
                a_float[both_valid], b_float[both_valid], rtol=TOLERANCE, atol=TOLERANCE
            )
            bad = nan_mismatch | value_mismatch
        else:
            bad = np.array([
                not (pd.isna(x) and pd.isna(y)) and x != y
                for x, y in zip(a_values, b_values)
            ])

        if bad.any():
            first_bad = int(np.argmax(bad))
            problems.append(
                f"{column}: 切断位置 {cut} で {int(bad.sum())} 箇所が変化 "
                f"(最初は index {first_bad}, "
                f"全体版={a_values[first_bad]!r} 切断版={b_values[first_bad]!r})"
            )
    return problems


def assert_no_lookahead(
    build_features: FeatureBuilder,
    price_data: pd.DataFrame,
    warmup: int = 0,
) -> None:
    """ルックアヘッドがあれば LookaheadDetectedError を投げる。"""
    full = build_features(price_data)
    if not isinstance(full, pd.DataFrame):
        full = full.to_frame()

    for cut in _cut_positions(len(price_data), warmup):
        truncated = build_features(price_data.iloc[:cut])
        if not isinstance(truncated, pd.DataFrame):
            truncated = truncated.to_frame()
        problems = _compare(full, truncated, cut)
        if problems:
            raise LookaheadDetectedError(
                f"ルックアヘッドを検出しました (切断位置 {cut}):\n  "
                + "\n  ".join(problems)
            )


# --------------------------------------------------------------------------
# 陰性対照: わざとルックアヘッドを仕込んだ特徴量 (skill 6.4)
# 検査器がこれら全部を捕まえられることを、本番の検査より先に確認する。
# --------------------------------------------------------------------------

def build_features_with_full_period_statistics(price_data: pd.DataFrame) -> pd.DataFrame:
    """【わざと不正】全期間の平均と標準偏差で正規化する。"""
    close = price_data["close"]
    return pd.DataFrame({"bad": (close - close.mean()) / close.std()})


def build_features_with_backward_fill(price_data: pd.DataFrame) -> pd.DataFrame:
    """【わざと不正】bfill で未来の値を過去の欠損に埋める。"""
    close = price_data["close"].copy()
    close.iloc[::50] = np.nan
    return pd.DataFrame({"bad": close.bfill()})


def build_features_with_negative_shift(price_data: pd.DataFrame) -> pd.DataFrame:
    """【わざと不正】次のバーの終値を特徴量にする。"""
    return pd.DataFrame({"bad": price_data["close"].shift(-1)})


def build_features_with_centered_rolling(price_data: pd.DataFrame) -> pd.DataFrame:
    """【わざと不正】center=True の移動平均 (窓の半分が未来)。"""
    return pd.DataFrame({"bad": price_data["close"].rolling(21, center=True).mean()})


def build_features_with_full_period_minmax(price_data: pd.DataFrame) -> pd.DataFrame:
    """【わざと不正】全期間の min-max スケーリング。"""
    close = price_data["close"]
    return pd.DataFrame({"bad": (close - close.min()) / (close.max() - close.min())})


def build_features_with_reversed_expanding(price_data: pd.DataFrame) -> pd.DataFrame:
    """【わざと不正】逆向き expanding。t 以降すべての最大値を見る。

    影響範囲が「その先の全部」なので、切断位置がどこでも変化する。
    """
    reversed_max = price_data["close"][::-1].expanding().max()[::-1]
    return pd.DataFrame({"bad": reversed_max})


NEGATIVE_CONTROLS: dict[str, FeatureBuilder] = {
    "全期間の mean/std で正規化": build_features_with_full_period_statistics,
    "bfill で未来の値を埋める": build_features_with_backward_fill,
    "shift(-1) で次バーの終値を使う": build_features_with_negative_shift,
    "center=True の移動平均": build_features_with_centered_rolling,
    "全期間 min-max スケーリング": build_features_with_full_period_minmax,
    "逆向き expanding max": build_features_with_reversed_expanding,
}


def verify_detector_catches_negative_controls(price_data: pd.DataFrame) -> None:
    """検査器そのものを検査する。全部を捕まえられなければ例外。"""
    for name, bad_builder in NEGATIVE_CONTROLS.items():
        try:
            assert_no_lookahead(bad_builder, price_data)
        except LookaheadDetectedError:
            continue                      # 期待どおり検出できた
        raise AssertionError(
            f"検査器が陰性対照『{name}』のルックアヘッドを検出できませんでした。"
            " 切断位置の設計を見直すこと。"
        )
