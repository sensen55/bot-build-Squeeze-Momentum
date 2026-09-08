# Squeeze Momentum Indicator [LazyBear] — エッジ検証

TradingView の Squeeze Momentum Indicator [LazyBear] に、PerpDEX の perp 取引で
コストを超えるエッジが存在するかを検証したプロジェクト。

**目的は bot 化ではなくエッジの有無の判定。**

## 結論

**エッジは確認できなかった。Phase 1（イベントスタディ）で中止基準に到達し、
Phase 2 以降には進んでいない。**

事前に確定した合格条件を満たしたセルは **0 / 18**。
詳細は [`reports/phase1_event_study.md`](reports/phase1_event_study.md)。

追試として解放イベントの定義を原典どおり（黒 → 灰）に修正したが、
このデータでは青（no_sqz）が 1 本も発生しないため**イベント集合は完全に同一**で、
18 セルの数値も完全一致した。結論は変わらない。
詳細は [`reports/phase1_release_definition.md`](reports/phase1_release_definition.md)。

主要な数字:

| 検定 | 結果 |
|---|---|
| 解放時 `val` の符号方向リターン | 5 分足は全ホライズンでマイナス。1 時間足はプラスだが Bonferroni 補正後に有意なセル 0 |
| スクイーズ解放 → 値幅拡大 | std 比 0.92〜1.15。有意なのは N=1 のみで、効果は +4〜15%。N=3 以降で消える |
| 勝率 | 0.46〜0.52（ほぼコイン投げ） |
| 対照（全バーで `val` 符号） | 解放バーに絞っても改善しない。時間足ごとに符号が逆転する |

## Phase 1b: 「主役 = WaveTrend / ADX、補助 = スクイーズ」仮説

作者 LazyBear が「ADX や WaveTrend のような追加指標が必要だった」と述べていることから、
**スクイーズを主役ではなく補助として使った場合に価値があるか**を検証した（Phase 1 とは別の仮説）。

**問 1（主役は単体で方向情報を持つか）で合格セル 0 / 36。** 事前の中止基準により
問 2（補助の価値）は実施せず、Phase 1b をここで終了した。

| 主役 | 合格セル | Bonferroni 補正後に有意 | CI 下限 > 1.0 bps |
|---|---|---|---|
| WaveTrend（10/21/4、標準設定） | 0 / 18 | 0 / 18 | 0 / 18 |
| ADX（14、閾値 20） | 0 / 18 | 0 / 18 | 0 / 18 |

主検定は ATR 正規化リターンで実施（Phase 1 のレビューを受けた改善。bps でプールすると
値動きの大きい SOL に平均が引っ張られるため）。勝率はいずれも 0.45〜0.51。

なお、**判定に使わないと事前宣言した副次族**（ADX 閾値 25 / 15 分足）に
コストを超える数字が出ている。判定は変えていないが、レポートに事実を記載している。
詳細は [`reports/phase1b_leader_and_squeeze.md`](reports/phase1b_leader_and_squeeze.md)。

## Phase 1c: パラメータ探索と、未使用データでの確認

標準設定に固定してきたパラメータを探索した。ただし探索は必ず「見かけ上良い設定」を生むため、
**Part A（探索）と Part B（確認）でデータを完全に分離**した。

- **Part A**: 2025-01..2026-05（既に何度も見ている期間）で 65 設定 × 3 時間足 × 6 ホライズン
  = 1,170 セルを探索。p 値も信頼区間も計算していない（多重比較の産物が「発見」に見えるため）。
  選定ルール（近傍安定性・3 銘柄符号一致）は**グリッドを回す前にコミット**した。
- **Part B**: 2023-01..2024-12（**初アクセス**）で、事前登録した 4 候補のみを検定。

**Part B の合格候補は 0 / 4。** 参考の 24 セル全体を見ても合格条件を満たすセルは 0 件で、
ルールを破ってスヌーピングしても結論は変わらない。

| # | 設定 | 時間足 | N | Part A 平均ATR | Part B 平均ATR | 残存率 | p値 | CI下限(bps) |
|---|---|---|---|---:|---:|---:|---:|---:|
| 1 | ADX length=28 thr=25 | 15m | 12 | +0.375 | +0.141 | +38% | 0.106 | −6.16 |
| 2 | Squeeze bb=20 kc=30 mult=2.0 | 1h | 3 | +0.346 | **−0.127** | −37% | 0.219 | −22.32 |
| 3 | ADX length=20 thr=30 | 15m | 12 | +0.321 | +0.157 | +49% | 0.044 | −3.70 |
| 4 | ADX length=14 thr=25 | 15m | 12 | +0.184 | +0.091 | +49% | 0.064 | −1.63 |

族の地形（自分のスコアと隣接設定スコア中央値の Spearman 相関）:
ADX +0.66 / WaveTrend +0.79 は「平らな丘」だが、**Squeeze は +0.17 で「孤立した山ばかり」**。

依頼書に定めたとおり、**Squeeze / WaveTrend / ADX の 3 指標については検証を完全終了**した。
詳細は [`reports/phase1c_exploration.md`](reports/phase1c_exploration.md)（Part A）と
[`reports/phase1c_confirmation.md`](reports/phase1c_confirmation.md)（Part B）。

## 構成

| ファイル | 内容 |
|---|---|
| `squeeze_momentum.py` | 指標計算のみ（売買ロジックなし）。Phase 0 |
| `data/fetch_binance_klines.py` | Binance USD-M Futures kline 取得 + 健全性検査 |
| `lookahead_check.py` | 切断法によるルックアヘッド検査 + 陰性対照 6 種 |
| `tests/reference_impl.py` | Pine から独立に書き起こした素朴なループ実装（検算用） |
| `tests/test_squeeze_momentum.py` | Phase 0 の正確性検証（28 項目） |
| `tests/tradingview_crosscheck.py` | TradingView 手動照合用の数値ダンプ |
| `phase1_event_study.py` | Phase 1 のイベント抽出と統計 |
| `phase1_nearmiss.py` | 最良セルの精査（銘柄別 / 期間分割 / 隣接ホライズン） |
| `phase1_release_definition.py` | 追試: 解放イベント定義（黒→灰 / 黒→青 / 黒→黒以外）の比較 |
| `main_indicators.py` | Phase 1b の主役: WaveTrend Oscillator と ADX（指標計算のみ）|
| `phase1b_event_study.py` | Phase 1b のシグナル抽出と統計 |
| `phase1b_run.py` | Phase 1b 実行とレポート生成（`--report-only` で再集計なしの再出力）|
| `tests/test_main_indicators.py` | WaveTrend / ADX の正確性検証（30 項目）|
| `tests/test_phase1b.py` | Phase 1b の集計ロジックの検算（13 項目）|
| `phase1c_grid.py` | Phase 1c のグリッド定義と候補選定ルール（実行前に確定）|
| `phase1c_engine.py` | Phase 1c の評価エンジン（Part A / B で共通）|
| `phase1c_partA.py` | Part A: 探索とレポート生成 |
| `phase1c_partB_data.py` | Part B: 2023-2024 データの取得と健全性検査 |
| `phase1c_partB.py` | Part B: 事前登録した 4 候補のみの検定 |
| `phase1_run.py` | Phase 1 実行とレポート生成 |
| `COSTS.md` | コスト前提と中止基準（実行前に確定） |

## 実行

```bash
pip install pandas numpy scipy tabulate
python3 tests/test_squeeze_momentum.py   # Phase 0:  指標の正確性 (28 項目)
python3 phase1_run.py                    # Phase 1:  イベントスタディ
python3 phase1_release_definition.py     # 追試:      解放イベント定義の比較
python3 tests/test_main_indicators.py    # Phase 1b: WaveTrend / ADX の正確性 (30 項目)
python3 tests/test_phase1b.py            # Phase 1b: 集計ロジックの検算 (13 項目)
python3 phase1b_run.py                   # Phase 1b: 主役 + 補助の検証
python3 phase1c_partA.py                 # Phase 1c: Part A 探索
python3 phase1c_partB_data.py            # Phase 1c: Part B データ取得と健全性検査
python3 phase1c_partB.py                 # Phase 1c: Part B 確認 (判定はここだけ)
```

## データ

- Binance USD-M Futures kline（`data.binance.vision` の月次アーカイブ）
- 実行環境から `fapi.binance.com` は地域制限（HTTP 451）のため公開ダンプを使用
- BTCUSDT / ETHUSDT / SOLUSDT × 5m / 15m / 1h
- 検証期間: 2025-01-01 .. 2026-05-31（全 9 データセットが理論本数と一致、欠損 0 本）
- Phase 1c Part B: 2023-01-01 .. 2024-12-31 の 15m / 1h（全 6 データセットが理論本数と一致、欠損 0 本）
- **Hold-out: 2026-06-01 .. 2026-08-31 は封印・未使用**（`data/cache/` は 2026-05 で止まる）

## 未解決事項

`COSTS.md` に記載。特に以下は本番検討時に必ず埋めること。

- 両取引所の公式ドキュメントは実行環境の egress proxy で遮断され、手数料は二次情報
- ボラティリティ拡大局面での往復スリッページが未実測

ただし本結論は**手数料 0%・往復 1 bps という最も甘いコスト前提**で出しているため、
これらの精度は結論を変えない。
