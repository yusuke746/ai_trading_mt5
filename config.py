import os
from dotenv import load_dotenv

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_CURRENT_DIR, ".env"))

# ──────────────────────────────────────
# MT5 接続設定
# ──────────────────────────────────────
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\XMTrading MT5\terminal64.exe")
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "XMTrading-MT5")

# ──────────────────────────────────────
# OpenAI API
# ──────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")
OPENAI_FINAL_APPROVAL_MODEL = os.getenv("OPENAI_FINAL_APPROVAL_MODEL", "gpt-5.2")
OPENAI_FINAL_APPROVAL_ENABLED = os.getenv("OPENAI_FINAL_APPROVAL_ENABLED", "true").lower() == "true"
OPENAI_FINAL_APPROVAL_REASONING_EFFORT = os.getenv("OPENAI_FINAL_APPROVAL_REASONING_EFFORT", "medium")

# 最終承認は高ボラ銘柄 or 高confidence時だけ実行
FINAL_APPROVAL_SYMBOLS = {
    symbol.strip() for symbol in os.getenv("FINAL_APPROVAL_SYMBOLS", "GOLD,US100Cash").split(",") if symbol.strip()
}
FINAL_APPROVAL_MIN_CONFIDENCE = int(os.getenv("FINAL_APPROVAL_MIN_CONFIDENCE", "75"))

# ──────────────────────────────────────
# Discord Webhook
# ──────────────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ──────────────────────────────────────
# 監視銘柄リスト (MT5シンボル名)
# XMTrading のシンボル名に合わせて変更してください
# ──────────────────────────────────────
SYMBOLS = [
    "GOLD",       # ゴールド (XAUUSD)
    "USDJPY",     # ドル円
    "EURUSD",     # ユロドル
    "US100Cash",  # NASDAQ100 (XMTradingでの名称)
    "OILCash",    # WTI原油 (XMTradingでの名称)
]

# ──────────────────────────────────────
# 通貨グループ定義 (相関リスク制御用)
# 各銘柄がどの通貨に「関連する」かを定義
# ──────────────────────────────────────
CURRENCY_GROUPS = {
    "GOLD":      ["USD"],
    "USDJPY":    ["USD", "JPY"],
    "EURUSD":    ["USD", "EUR"],
    "US100Cash": ["USD"],
    "OILCash":   ["USD"],
}

# ──────────────────────────────────────
# 資金管理パラメータ
# ──────────────────────────────────────
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.02"))   # 1トレード損切り = 残高の 2%
MAX_LOT = float(os.getenv("MAX_LOT", "1.0"))                  # ロット上限
MAX_CORRELATED_POSITIONS = int(os.getenv("MAX_CORRELATED_POSITIONS", "2"))

# ──────────────────────────────────────
# タイムフレーム
# ──────────────────────────────────────
EXECUTION_TF = "M15"   # 執行足
TREND_TF = "H1"        # トレンドフィルター足
EXIT_MONITOR_TF = os.getenv("EXIT_MONITOR_TF", "M15")

# ──────────────────────────────────────
# ATR 設定
# ──────────────────────────────────────
ATR_PERIOD = 14
MA_PERIOD = 20

# ──────────────────────────────────────
# 機械式緊急エグジット設定
# ──────────────────────────────────────
EMERGENCY_EXIT_ENABLED = os.getenv("EMERGENCY_EXIT_ENABLED", "true").lower() == "true"
EMERGENCY_EXIT_ADVERSE_ATR = float(os.getenv("EMERGENCY_EXIT_ADVERSE_ATR", "1.8"))
EMERGENCY_EXIT_ATR_SPIKE_MULTIPLIER = float(os.getenv("EMERGENCY_EXIT_ATR_SPIKE_MULTIPLIER", "1.8"))
EMERGENCY_EXIT_ATR_SPIKE_LOOKBACK = int(os.getenv("EMERGENCY_EXIT_ATR_SPIKE_LOOKBACK", "20"))
EMERGENCY_EXIT_ADVERSE_ATR_BY_SYMBOL = {
    "GOLD": 2.4,
    "US100Cash": 2.0,
    "OILCash": 2.2,
    "USDJPY": 1.8,
    "EURUSD": 1.8,
}

# ──────────────────────────────────────
# 利益保護設定
# ──────────────────────────────────────
PROFIT_PROTECTION_ENABLED = os.getenv("PROFIT_PROTECTION_ENABLED", "true").lower() == "true"
BREAKEVEN_R = float(os.getenv("BREAKEVEN_R", "1.0"))
BREAKEVEN_BUFFER_R = float(os.getenv("BREAKEVEN_BUFFER_R", "0.10"))
LOCK_PROFIT_1_TRIGGER_R = float(os.getenv("LOCK_PROFIT_1_TRIGGER_R", "1.5"))
LOCK_PROFIT_1_R = float(os.getenv("LOCK_PROFIT_1_R", "0.50"))
LOCK_PROFIT_2_TRIGGER_R = float(os.getenv("LOCK_PROFIT_2_TRIGGER_R", "2.0"))

# 利確優先のTP設定
ENTRY_TP_R = float(os.getenv("ENTRY_TP_R", "1.2"))
ENTRY_MIN_TP_R = float(os.getenv("ENTRY_MIN_TP_R", "1.0"))
EXIT_MIN_CONFIDENCE = int(os.getenv("EXIT_MIN_CONFIDENCE", "45"))
FORCE_EXIT_ON_PREMISE_BREAK = os.getenv("FORCE_EXIT_ON_PREMISE_BREAK", "true").lower() == "true"

# ──────────────────────────────────────
# SMC (Smart Money Concepts) フィルタ設定
# ──────────────────────────────────────
# SMCフィルタを有効にすると、エントリー条件にLiquidity Sweepの確認が追加される
SMC_FILTER_ENABLED = os.getenv("SMC_FILTER_ENABLED", "true").lower() == "true"
# Liquidity Sweepと判定するための最小侵食幅 (ATR比率)
# 例: 0.3 → 高安値をATRの30%以上超えた場合にSweep認定
SMC_SWEEP_ATR_MULT = float(os.getenv("SMC_SWEEP_ATR_MULT", "0.3"))
# 機械的SMCゲート: AI呼び出し前に数値条件でフィルタリング
# Falseにすると機械ゲートをスキップしてAIのみで判断
SMC_MECHANICAL_GATE_ENABLED = os.getenv("SMC_MECHANICAL_GATE_ENABLED", "true").lower() == "true"
# Sweepを探す遡り期間 (H1バー数)
SMC_SWEEP_LOOKBACK_BARS = int(os.getenv("SMC_SWEEP_LOOKBACK_BARS", "10"))
# 順張り (Continuation BOS) エントリーを有効にする
SMC_CONTINUATION_ENABLED = os.getenv("SMC_CONTINUATION_ENABLED", "true").lower() == "true"
# 順張りBOS判定: MAのスロープを見る遡り期間 (H1バー数)
SMC_CONTINUATION_BOS_LOOKBACK_BARS = int(os.getenv("SMC_CONTINUATION_BOS_LOOKBACK_BARS", "5"))
# 順張りBOS判定: MA傾きの最小値 (ATR比率) — この値未満のMA傾きはトレンドなしとみなす
SMC_CONTINUATION_MA_SLOPE_ATR_MULT = float(os.getenv("SMC_CONTINUATION_MA_SLOPE_ATR_MULT", "0.3"))

# 市場クローズ時の無駄なAI判定を防ぐため、ティックが古い銘柄は停止中とみなす
MARKET_DATA_STALE_SEC = int(os.getenv("MARKET_DATA_STALE_SEC", "1800"))
LOCK_PROFIT_2_R = float(os.getenv("LOCK_PROFIT_2_R", "1.00"))

# ──────────────────────────────────────
# チャート画像設定
# ──────────────────────────────────────
CHART_BARS = 100        # 生成チャートに表示するバー数
CHART_WIDTH = 1200      # px
CHART_HEIGHT = 700      # px

# ──────────────────────────────────────
# ループ間隔
# ──────────────────────────────────────
HEARTBEAT_INTERVAL_SEC = 3600     # Heartbeat: 1時間
MAIN_LOOP_SLEEP_SEC = 10          # メインループスリープ
CANDLE_WAIT_SEC = 15              # 足確定後の待ち時間

# ──────────────────────────────────────
# SQLite メンテナンス設定
# ──────────────────────────────────────
DB_MAINTENANCE_INTERVAL_SEC = int(os.getenv("DB_MAINTENANCE_INTERVAL_SEC", "3600"))
DB_FULL_VACUUM_INTERVAL_SEC = int(os.getenv("DB_FULL_VACUUM_INTERVAL_SEC", "86400"))
DB_RETENTION_DAYS_AI_LOGS = int(os.getenv("DB_RETENTION_DAYS_AI_LOGS", "14"))
DB_RETENTION_DAYS_HEARTBEATS = int(os.getenv("DB_RETENTION_DAYS_HEARTBEATS", "30"))
DB_RETENTION_DAYS_CLOSED_TRADES = int(os.getenv("DB_RETENTION_DAYS_CLOSED_TRADES", "365"))

# ──────────────────────────────────────
# パス
# ──────────────────────────────────────
BASE_DIR = _CURRENT_DIR
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "trades.db")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")

# ディレクトリ作成
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
