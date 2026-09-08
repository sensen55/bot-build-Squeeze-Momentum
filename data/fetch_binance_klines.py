"""Binance USD-M Futures の kline を公開データダンプから取得する。

なぜ REST API ではなく data.binance.vision を使うか
----------------------------------------------------
実行環境から fapi.binance.com は地域制限 (HTTP 451) で拒否される。
data.binance.vision は Binance 公式が配布している同一の約定由来データの
月次アーカイブであり、内容は API の kline と同じ。

ルックアヘッドに関する注意
--------------------------
kline の open_time は「バーの開始時刻」である。バーが確定するのは close_time。
本プロジェクトでは DataFrame の index を **close_time（バー確定時刻）** に置く。
こうしておくと「index の時刻に立っていれば、その行の値はすべて既知」という
不変条件が成り立ち、うっかり未確定バーを参照する事故を防げる。
"""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

INTERVAL_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}


@dataclass(frozen=True)
class IntegrityReport:
    """データ健全性の検算結果 (skill 6.5b / 6.5c)。"""

    symbol: str
    interval: str
    start: pd.Timestamp
    end: pd.Timestamp
    expected_bars: int
    actual_bars: int
    missing_times: pd.DatetimeIndex
    duplicated_times: int

    @property
    def missing_count(self) -> int:
        return len(self.missing_times)

    def describe(self) -> str:
        lines = [
            f"{self.symbol} {self.interval}: {self.actual_bars} / {self.expected_bars} 本 "
            f"(欠損 {self.missing_count}, 重複 {self.duplicated_times})"
        ]
        if self.missing_count:
            lines.append("  欠損時刻の規則性チェック:")
            lines.extend(f"    {line}" for line in _describe_missing_pattern(self.missing_times))
        return "\n".join(lines)


def _describe_missing_pattern(missing: pd.DatetimeIndex) -> list[str]:
    """欠損位置に規則性がないかを要約する。

    skill 6.5c: 規則性のある欠損は自然現象ではなくバグ。
    「毎月1日の00:05」のような並びが出たら読み込み処理を疑う。
    """
    out: list[str] = []
    by_time_of_day = missing.to_series().groupby(missing.time).size().sort_values(ascending=False)
    out.append(f"時刻別 top5: {dict(list(by_time_of_day.items())[:5])}")
    by_day_of_month = missing.to_series().groupby(missing.day).size().sort_values(ascending=False)
    out.append(f"日別 top5: {dict(list(by_day_of_month.items())[:5])}")
    concentration = by_time_of_day.iloc[0] / len(missing) if len(missing) else 0.0
    if concentration > 0.30 and len(missing) >= 10:
        out.append(
            f"  !! 警告: 欠損の {concentration:.0%} が単一の時刻に集中している。"
            " 自然な欠損ではなく読み込みバグの可能性が高い。"
        )
    else:
        out.append("  規則性は検出されず (単一時刻への集中は3割未満)")
    return out


def _month_range(start: str, end: str) -> list[str]:
    months = pd.date_range(pd.Timestamp(start).normalize().replace(day=1),
                           pd.Timestamp(end), freq="MS")
    return [m.strftime("%Y-%m") for m in months]


def _download_month(symbol: str, interval: str, month: str) -> bytes:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{symbol}-{interval}-{month}.zip"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()

    url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            cache_path.write_bytes(payload)
            return payload
        except Exception as exc:  # noqa: BLE001 - ネットワーク起因は全部リトライ対象
            last_error = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ダウンロード失敗: {url}") from last_error


def _parse_month(payload: bytes, symbol: str, interval: str, month: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        raw = archive.read(name)

    first_line = raw.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    # ヘッダー行の有無をファイルごとに判定する。
    # skill 6.5c の事例: header=0 と skiprows=1 を同時に渡すと、各ファイルの
    # 先頭1本が静かに消える。ここでは片方だけを使い、両方は決して渡さない。
    has_header = first_line.lower().startswith("open_time")
    frame = pd.read_csv(
        io.BytesIO(raw),
        header=0 if has_header else None,
        names=None if has_header else RAW_COLUMNS,
    )
    frame = frame[RAW_COLUMNS[:9]]

    # 一部の期間で open_time がマイクロ秒になっている。桁数で判定する。
    unit = "us" if frame["open_time"].iloc[0] > 1e14 else "ms"
    frame["open_time"] = pd.to_datetime(frame["open_time"], unit=unit, utc=True)
    frame["close_time"] = pd.to_datetime(frame["close_time"], unit=unit, utc=True)

    for column in ("open", "high", "low", "close", "volume", "quote_volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    frame.attrs["source_month"] = month
    return frame


def load_klines(
    symbol: str,
    interval: str,
    start: str,
    end: str,
    verbose: bool = True,
) -> tuple[pd.DataFrame, IntegrityReport]:
    """[start, end) の kline を取得する。index は close_time (バー確定時刻)。"""
    if interval not in INTERVAL_MINUTES:
        raise ValueError(f"未対応の interval: {interval}")

    frames = [
        _parse_month(_download_month(symbol, interval, month), symbol, interval, month)
        for month in _month_range(start, end)
    ]
    data = pd.concat(frames, ignore_index=True)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    data = data[(data["open_time"] >= start_ts) & (data["open_time"] < end_ts)]

    duplicated = int(data["open_time"].duplicated().sum())
    data = data.drop_duplicates(subset="open_time", keep="first")
    data = data.sort_values("open_time").reset_index(drop=True)

    step = pd.Timedelta(minutes=INTERVAL_MINUTES[interval])
    expected_open_times = pd.date_range(start_ts, end_ts - step, freq=step)
    missing = expected_open_times.difference(pd.DatetimeIndex(data["open_time"]))

    report = IntegrityReport(
        symbol=symbol,
        interval=interval,
        start=start_ts,
        end=end_ts,
        expected_bars=len(expected_open_times),
        actual_bars=len(data),
        missing_times=missing,
        duplicated_times=duplicated,
    )
    if verbose:
        print(report.describe())

    # index は「バー確定時刻」。欠損は埋めない (ffill も bfill もしない)。
    # 欠損を埋めると存在しないバーを取引可能に見せてしまうため、
    # 指標側で連続性が崩れる箇所は NaN のまま扱う。
    data = data.set_index("close_time")
    data.index.name = "close_time"
    return data, report


def load_panel(
    symbols: list[str],
    intervals: list[str],
    start: str,
    end: str,
    verbose: bool = True,
) -> dict[tuple[str, str], pd.DataFrame]:
    panel: dict[tuple[str, str], pd.DataFrame] = {}
    for symbol in symbols:
        for interval in intervals:
            frame, _ = load_klines(symbol, interval, start, end, verbose=verbose)
            panel[(symbol, interval)] = frame
    return panel


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "1h"])
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-06-01")
    args = parser.parse_args()

    for sym in args.symbols:
        for iv in args.intervals:
            load_klines(sym, iv, args.start, args.end)
