# ai_trading_mt5

MT5 (XMTrading) 向けの AI 自動売買システムです。

## 概要
- 実行足: M15
- 上位足フィルター: H1
- 監視銘柄: GOLD, USDJPY, EURUSD, US100Cash, OILCash
- AI判定: OpenAI Responses API
- 通知: Discord Webhook
- ログ: SQLite

## 主な機能
- H1/M15 のマルチタイムフレーム判定
- 相関リスク制御（通貨グループ上限）
- ゴールド対応の厳密ロット計算（contract size 動的取得）
- 2段判定（一次 + 最終承認モデル）
- エントリー時TP設定（AI提案 + 最低R倍率ガード）
- 保有中にエントリー根拠が崩れた場合の強制EXIT
- 利確優先のポジション管理方針
- 機械式緊急エグジット
- 利益保護（建値移動・利益ロック）
- 市場クローズ中のAI判定スキップ
- SQLite 自動メンテナンス

## セットアップ
1. Python 3.11+ を準備
2. 依存関係をインストール

```powershell
pip install -r requirements.txt
```

3. `.env.example` を参考に `.env` を作成
4. MT5 (XMTrading) を起動し、口座ログイン

## 実行

```powershell
python main.py
```

## 主要設定
- `OPENAI_MODEL`
- `OPENAI_FINAL_APPROVAL_MODEL`
- `RISK_PER_TRADE`
- `MAX_LOT`
- `EXIT_MONITOR_TF`
- `EXIT_MIN_CONFIDENCE`
- `FORCE_EXIT_ON_PREMISE_BREAK`
- `ENTRY_TP_R`
- `ENTRY_MIN_TP_R`
- `MARKET_DATA_STALE_SEC`
- `EMERGENCY_EXIT_*`
- `PROFIT_PROTECTION_*`

## デモ環境から別環境への引き継ぎ

学習済みの confidence 閾値（`adaptive_params.json`）を新環境にコピーすることで、デモ環境のパラメータをそのままの状態で引き継げます。

### 手順

1. デモ環境の `analytics/adaptive_params.json` を新環境の同ディレクトリにコピー
2. 必要に応じて `trades.db` も同様にコピー（履歴があると Lookback 期間の統計が即使用可能）
3. `.env` は環境ごとに新規作成（MT5 ログイン情報・APIキー等が異なるため）

```
# コピーが必要なファイル
analytics/adaptive_params.json   ← 学習済み閾値（必須）
trades.db                         ← トレード履歴（任意）

# 環境ごとに作り直すファイル
.env                              ← APIキー・MT5接続情報
```

### コピーしない場合の初期値

`adaptive_params.json` がない場合、起動時に自動でデフォルト値が使われます。

```
global_confidence_threshold = ADAPTIVE_CONF_MIN (デフォルト: 70)
buckets = {} (空)
```

その後、`ADAPTIVE_MIN_SAMPLES`（デフォルト: 10件）のトレードが溜まると、週次で自動更新されます。

### 引き継ぎ後の挙動

コピーした `adaptive_params.json` の閾値は初期値として使われますが、新環境でのトレード実績が蓄積されると次の週次更新サイクルで上書きされ、新環境の実績に基づいて独立して育っていきます。

## 注意
- `.env` は機密情報を含むため Git 管理しない
- 実運用前にデモ口座で十分に検証する
- 高ボラ銘柄（GOLD/US100Cash/OILCash）は設定を保守的に運用する
