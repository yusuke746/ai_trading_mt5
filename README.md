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
- `MARKET_DATA_STALE_SEC`
- `EMERGENCY_EXIT_*`
- `PROFIT_PROTECTION_*`

## 注意
- `.env` は機密情報を含むため Git 管理しない
- 実運用前にデモ口座で十分に検証する
- 高ボラ銘柄（GOLD/US100Cash/OILCash）は設定を保守的に運用する
