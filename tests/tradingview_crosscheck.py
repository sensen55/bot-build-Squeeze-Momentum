"""TradingView 手動照合用の数値ダンプ。

この環境から TradingView へはアクセスできない (要認証・JS 描画) ため、
自動照合はできない。代わりに全中間値を出力し、利用者が
TradingView 上の BINANCE:BTCUSDT.P 1h に
Squeeze Momentum Indicator [LazyBear] (既定パラメータ 20/2.0/20/1.5, useTrueRange=true)
を載せて、同じ時刻のバーと目視照合できるようにする。

TradingView 側の注意:
  - チャートのタイムゾーンを UTC にする
  - 表示される時刻は「バーの開始時刻」。下表の open_time と突き合わせる
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.fetch_binance_klines import load_klines   # noqa: E402
from squeeze_momentum import compute                # noqa: E402

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def main() -> None:
    data, _ = load_klines("BTCUSDT", "1h", "2025-01-01", "2025-04-01", verbose=False)
    result = compute(data)
    joined = data[["open", "high", "low", "close"]].join(result)
    joined.insert(0, "open_time_utc", joined.index - pd.Timedelta(hours=1) + pd.Timedelta(milliseconds=1))

    sample = joined.iloc[500:510]
    columns = ["open_time_utc", "open", "high", "low", "close",
               "basis", "upper_bb", "lower_bb", "rangema", "upper_kc", "lower_kc",
               "sqz_on", "sqz_off", "val", "color_state"]
    print("BINANCE:BTCUSDT.P / 1h / UTC / LazyBear 既定パラメータ (20, 2.0, 20, 1.5)")
    print(sample[columns].to_string(float_format=lambda v: f"{v:,.4f}"))


if __name__ == "__main__":
    main()
