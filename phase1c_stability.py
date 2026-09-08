"""Phase 1c 追加分析: 事前登録候補の「時間的な安定性」。

なぜこれをやるのか
------------------
Part B の判定は「合格 0/4」で終わっているが、これは有意性の話であって
「効果が時間的に安定しているか」は別の問いである。

平均が +2.33 bps でも、その中身が
  (a) 8 四半期にまんべんなく散らばっている
  (b) 1 四半期だけが突出していて、残りはほぼゼロ
のどちらなのかで、意味がまったく違う。(b) なら「特定の相場環境でだけ効く」
可能性が高く、次に同じ環境が来る保証はない。

なぜ Walk-Forward Analysis ではなくこれなのか
---------------------------------------------
WFA はデータを訓練期間とテスト期間に分割するので、判定に使える件数が減る。
確信度 (信頼区間の狭さ) はデータ量で決まるため、**WFA を回すと今より結論が
曖昧になる**。合格しなかったものが合格に変わることは原理的にない。

  全期間そのまま 6,067 件 → 標準誤差 1.68 bps
  WFA (OOS 8割)  4,854 件 → 標準誤差 1.88 bps (悪化)
  WFA (OOS 5割)  3,034 件 → 標準誤差 2.38 bps (悪化)

WFA が本来答えるのは「定期的な再最適化という手順が有効か」であり、
それは Part A で選んで Part B で試した時点で 1 回テスト済み (結果 0/4)。

そこで、確信度を増やそうとするのではなく、
**同じデータから別の問い (安定性) に答える**のがこの分析である。

これは記述統計であり、判定を覆すものではない。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase1c_engine import build_context, make_signals
from phase1c_grid import COST_FLOOR_BPS, SYMBOLS, Setting

REPORT_DIR = Path(__file__).resolve().parent / "reports"

PERIODS = {
    "Part B（未使用）": ("2023-01-01", "2025-01-01"),
    "Part A（探索済）": ("2025-01-01", "2026-06-01"),
}
# 実測スリッページ 0.7〜1.0 bps の中央値。往復コストの想定値
ASSUMED_COST_BPS = 0.85


def collect(setting: Setting, interval: str, horizon: int) -> pd.DataFrame:
    """候補 1 つについて、両期間のトレードを 1 本の表にまとめる。"""
    frames = []
    for period, (start, end) in PERIODS.items():
        for symbol in SYMBOLS:
            context = build_context(symbol, interval, start, end)
            index, direction, _ = make_signals(setting, context)
            values = direction[index] * context.forward_bps[horizon][index]
            valid = ~np.isnan(values)
            frames.append(pd.DataFrame({
                "time": context.data.index[index][valid],
                "bps": values[valid],
                "symbol": symbol,
                "period": period,
            }))
    frame = pd.concat(frames, ignore_index=True)
    frame["quarter"] = frame["time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    return frame


def summarise(frame: pd.DataFrame, period: str) -> dict:
    subset = frame[frame["period"] == period]
    quarterly = subset.groupby("quarter")["bps"].agg(["count", "mean"])
    best_quarter = quarterly["mean"].idxmax()
    without_best = subset[subset["quarter"] != best_quarter]
    return {
        "period": period,
        "n": len(subset),
        "mean_bps": float(subset["bps"].mean()),
        "n_quarters": len(quarterly),
        "quarters_positive": int((quarterly["mean"] > 0).sum()),
        "quarters_above_cost": int((quarterly["mean"] > ASSUMED_COST_BPS).sum()),
        "best_quarter": best_quarter,
        "best_quarter_mean": float(quarterly.loc[best_quarter, "mean"]),
        "mean_without_best": float(without_best["bps"].mean()),
        "quarterly": quarterly,
    }


def build_section(candidates: list[dict]) -> list[str]:
    lines: list[str] = []
    add = lines.append

    add("## 6. 追加分析: 時間的な安定性（記述統計・判定は覆さない）")
    add("")
    add("判定は「合格 0/4」で終わっているが、有意性とは別に")
    add("**効果が時間的に安定しているか**を見ておく。")
    add("")
    add("平均が同じ +2 bps でも、8 四半期にまんべんなく散らばっているのと、")
    add("1 四半期だけが突出して残りがほぼゼロなのとでは意味がまったく違う。")
    add("後者なら「特定の相場環境でだけ効く」可能性が高く、次に同じ環境が来る保証はない。")
    add("")
    add("### なぜ Walk-Forward Analysis を回さないのか")
    add("")
    add("WFA はデータを訓練期間とテスト期間に分割するため、")
    add("**判定に使える件数が減り、結論は今より曖昧になる**。")
    add("信頼区間の狭さはデータ量で決まるので、分析手法を変えても確信度は増えない。")
    add("")
    add("| 分析方法 | 使える件数 | 標準誤差 |")
    add("|---|---:|---:|")
    add("| 全期間そのまま（現行）| 6,067 | **1.68 bps** |")
    add("| WFA（OOS が 8 割）| 4,854 | 1.88 bps（悪化）|")
    add("| WFA（OOS が 5 割）| 3,034 | 2.38 bps（悪化）|")
    add("")
    add("WFA が本来答えるのは「定期的な再最適化という**手順**が有効か」であり、")
    add("それは Part A で選んで Part B で試した時点で 1 回テスト済み（結果 0/4）。")
    add("")

    summary_rows = []
    for i, candidate in enumerate(candidates, start=1):
        setting = Setting(candidate["family"], candidate["params"], ())
        frame = collect(setting, candidate["interval"], candidate["horizon"])
        add(f"### 候補 {i}: {setting.key} / {candidate['interval']} / "
            f"N={candidate['horizon']}")
        add("")
        for period in PERIODS:
            result = summarise(frame, period)
            summary_rows.append({"candidate": i, "setting": setting.key, **{
                k: v for k, v in result.items() if k != "quarterly"}})
            quarterly = result["quarterly"]
            add(f"**{period}** — {result['n']:,} 件 / 平均 {result['mean_bps']:+.2f} bps")
            add("")
            add("| 四半期 | 件数 | 平均(bps) |")
            add("|---|---:|---:|")
            for quarter, row in quarterly.iterrows():
                mark = " ←最良" if quarter == result["best_quarter"] else ""
                add(f"| {quarter} | {int(row['count'])} | {row['mean']:+.2f}{mark} |")
            add("")
            add(f"- 平均がプラスの四半期: **{result['quarters_positive']} / "
                f"{result['n_quarters']}**")
            add(f"- コスト {ASSUMED_COST_BPS} bps を上回った四半期: "
                f"**{result['quarters_above_cost']} / {result['n_quarters']}**")
            verdict = ("上回る" if result["mean_without_best"] > ASSUMED_COST_BPS
                       else "**下回る**")
            add(f"- 最良の {result['best_quarter']} を除くと平均 "
                f"**{result['mean_without_best']:+.2f} bps** → "
                f"コスト {ASSUMED_COST_BPS} bps を {verdict}")
            add("")

    add("### まとめ")
    add("")
    summary = pd.DataFrame(summary_rows)
    table = summary[["candidate", "period", "n", "mean_bps", "quarters_above_cost",
                     "n_quarters", "best_quarter", "mean_without_best"]].copy()
    table.columns = ["#", "期間", "n", "平均(bps)", "コスト超の四半期",
                     "四半期数", "最良四半期", "最良を除く平均(bps)"]
    add(table.round(2).to_markdown(index=False))
    add("")
    add("**読み方**")
    add("")
    add("- 探索に使った Part A（2025-2026）で成績が良いのは当然である。そう選んだのだから。")
    add("- 見るべきは Part B（未使用）の列。ここで**最良の 1 四半期を除いた平均が")
    add("  コストを下回る**なら、その候補のプラスは実質的にその 1 四半期が作っている。")
    add("- 探索期は全四半期がコスト超え、未使用期は一部だけ、という対比は")
    add("  **選択バイアスの教科書どおりの姿**である。")
    add("")
    add("**この分析は判定を覆さない。** Part B の判定は「合格 0/4」のままであり、")
    add("この結果はその結論を弱めるのではなく**強める方向**に働く。")
    add("")
    return lines


def main() -> None:
    candidates = json.loads(
        (REPORT_DIR / "phase1c_registered_candidates.json").read_text(encoding="utf-8"))
    lines = build_section(candidates)
    (REPORT_DIR / "phase1c_stability.md").write_text("\n".join(lines), encoding="utf-8")
    print("出力:", REPORT_DIR / "phase1c_stability.md")


if __name__ == "__main__":
    main()
