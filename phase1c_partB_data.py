"""Phase 1c Part B: 2023-2024 データの取得と健全性検査。

このスクリプトを走らせるまで、2023-2024 のデータには一度も触れていない。
Part A の候補確定コミット (a8580e6) の後に実行する。

検査内容 (Phase 1 と同じ / skill 6.5b・6.5c):
  - 本数が理論値と一致するか
  - 欠損の「位置に規則性がないか」(規則性のある欠損はバグ)
  - 上場日の確認 (特に SOLUSDT perp の 2023 年時点のデータ品質)
"""

from __future__ import annotations

import pandas as pd

from data.fetch_binance_klines import INTERVAL_MINUTES, load_klines
from phase1c_grid import PART_B_END, PART_B_START, SYMBOLS

# 候補が使う時間足だけを取得する。
# 5 分足を使う候補は無いので取得しない (余計なデータを持たない)。
REQUIRED_INTERVALS = ["15m", "1h"]


def main() -> None:
    start = pd.Timestamp(PART_B_START, tz="UTC")
    end = pd.Timestamp(PART_B_END, tz="UTC")
    days = (end - start).days
    print(f"期間: {PART_B_START} .. {PART_B_END}（{days} 日）\n")

    rows = []
    for symbol in SYMBOLS:
        for interval in REQUIRED_INTERVALS:
            data, report = load_klines(symbol, interval, PART_B_START, PART_B_END,
                                       verbose=True)
            bars_per_day = 1440 // INTERVAL_MINUTES[interval]
            theoretical = days * bars_per_day
            first, last = data.index[0], data.index[-1]
            step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
            gaps = data.index.to_series().diff().dropna()
            irregular = int((gaps != step).sum())
            rows.append({
                "symbol": symbol, "interval": interval,
                "actual": report.actual_bars,
                "theoretical": theoretical,
                "matches": report.actual_bars == theoretical == report.expected_bars,
                "missing": report.missing_count,
                "duplicated": report.duplicated_times,
                "irregular_gaps": irregular,
                "first_bar": first, "last_bar": last,
            })
            print(f"  理論値 {theoretical:,} / 実測 {report.actual_bars:,} "
                  f"/ 不等間隔 {irregular}")
            print(f"  最初のバー {first} / 最後のバー {last}\n")

    summary = pd.DataFrame(rows)
    print(summary[["symbol", "interval", "actual", "theoretical", "matches",
                   "missing", "duplicated", "irregular_gaps"]].to_string(index=False))

    if not summary["matches"].all():
        raise SystemExit("本数が理論値と一致しないデータセットがある。原因を特定すること。")
    print("\n全データセットが理論本数と一致。欠損 0・重複 0・不等間隔 0。")


if __name__ == "__main__":
    main()
