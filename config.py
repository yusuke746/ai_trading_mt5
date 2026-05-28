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
OPENAI_ENTRY_MODEL = os.getenv("OPENAI_ENTRY_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini"))
OPENAI_EXIT_MODEL = os.getenv("OPENAI_EXIT_MODEL", "gpt-5-nano")
# 互換性維持: 既存コード/外部スクリプト向け
OPENAI_MODEL = OPENAI_ENTRY_MODEL
OPENAI_FINAL_APPROVAL_MODEL = os.getenv("OPENAI_FINAL_APPROVAL_MODEL", "gpt-5.4")
# デバッグ完了まで無効化 (.envで OPENAI_FINAL_APPROVAL_ENABLED=true にすると再有効化可)
OPENAI_FINAL_APPROVAL_ENABLED = os.getenv("OPENAI_FINAL_APPROVAL_ENABLED", "false").lower() == "true"
OPENAI_FINAL_APPROVAL_REASONING_EFFORT = os.getenv("OPENAI_FINAL_APPROVAL_REASONING_EFFORT", "medium")

# 最終承認は高ボラ銘柄 or 高confidence時だけ実行
FINAL_APPROVAL_SYMBOLS = {
    symbol.strip() for symbol in os.getenv("FINAL_APPROVAL_SYMBOLS", "GOLD,USDJPY,EURUSD,US100Cash,OILCash").split(",") if symbol.strip()
}
FINAL_APPROVAL_MIN_CONFIDENCE = int(os.getenv("FINAL_APPROVAL_MIN_CONFIDENCE", "65"))

# ──────────────────────────────────────
# OpenRouter (Qwen-VL等マルチモデル対応)
# ──────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_ENTRY_MODEL = os.getenv("OPENROUTER_ENTRY_MODEL", "qwen/qwen2.5-vl-72b-instruct")
# true にするとエントリー分析をOpenRouter経由に切り替え (エグジットは引き続きOpenAI)
USE_OPENROUTER_FOR_ENTRY = os.getenv("USE_OPENROUTER_FOR_ENTRY", "false").lower() == "true"

# ──────────────────────────────────────
# Discord Webhook
# ──────────────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# ──────────────────────────────────────
# 監視銘柄リスト (MT5シンボル名)
# .env の SYMBOLS にカンマ区切りで指定: SYMBOLS=GOLD,USDJPY,EURUSD
# ──────────────────────────────────────
SYMBOLS = [
    s.strip()
    for s in os.getenv("SYMBOLS", "GOLD,USDJPY,EURUSD,US100Cash,OILCash").split(",")
    if s.strip()
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
EMERGENCY_EXIT_ATR_SPIKE_MIN_ADVERSE_ATR = float(os.getenv("EMERGENCY_EXIT_ATR_SPIKE_MIN_ADVERSE_ATR", "0.8"))
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
FLAT_BEFORE_MARKET_CLOSE_ENABLED = os.getenv("FLAT_BEFORE_MARKET_CLOSE_ENABLED", "true").lower() == "true"
FLAT_BEFORE_MARKET_CLOSE_HOUR = int(os.getenv("FLAT_BEFORE_MARKET_CLOSE_HOUR", "0"))
FLAT_BEFORE_MARKET_CLOSE_MINUTE = int(os.getenv("FLAT_BEFORE_MARKET_CLOSE_MINUTE", "0"))
FLAT_BEFORE_MARKET_CLOSE_LEAD_MINUTES = max(0, int(os.getenv("FLAT_BEFORE_MARKET_CLOSE_LEAD_MINUTES", "15")))
# 週末持ち越し回避: XMTradingサーバー時刻(GMT+3)で金曜23:59クローズ前に強制手仕舞い
# WEEKDAY/HOUR/MINUTEは不要 (サーバー時刻で金曜23:59を自動参照)
FLAT_BEFORE_WEEKEND_CLOSE_ENABLED = os.getenv("FLAT_BEFORE_WEEKEND_CLOSE_ENABLED", "true").lower() == "true"
FLAT_BEFORE_WEEKEND_CLOSE_LEAD_MINUTES = max(0, int(os.getenv("FLAT_BEFORE_WEEKEND_CLOSE_LEAD_MINUTES", "30")))

# 利確優先のTP設定
ENTRY_TP_R = float(os.getenv("ENTRY_TP_R", "1.2"))
ENTRY_MIN_TP_R = float(os.getenv("ENTRY_MIN_TP_R", "1.0"))

# ──────────────────────────────────────
# エントリーSL設定（固定値 vs ATR倍率）
# ──────────────────────────────────────
# 銘柄別の固定SL幅設定
# None: ATR倍率を使用 | 数値: その値を固定SLとして使用（機械ゲートの structural_sl_dist を無視）
ENTRY_FIXED_SL_BY_SYMBOL = {
    "GOLD": None,       # ATR/構造SLモード使用（固定0.50は狭すぎるため廃止）
    "USDJPY": None,     # ATR倍率を使用
    "EURUSD": None,
    "US100Cash": None,
    "OILCash": None,
}

# ATR倍率ベースのSL計算時の最小幅（ENTRY_FIXED_SL_BY_SYMBOL で None の銘柄のみ使用）
ENTRY_MIN_SL_ATR_MULT = max(0.1, float(os.getenv("ENTRY_MIN_SL_ATR_MULT", "0.8")))
# 銘柄別の最小SL幅上書き（ATR倍率使用時のみ）
ENTRY_MIN_SL_ATR_MULT_BY_SYMBOL = {
    "GOLD": 1.5,        # GOLD#: ATR×1.5を最小SL下限（ボラ大のため広め）
    "USDJPY": 1.0,
    "EURUSD": 0.8,
}
# ATR倍率ベースのSL計算時の最大幅（上限キャップ: swept_levelが遠い場合のロット計算崩壊防止）
ENTRY_MAX_SL_ATR_MULT = float(os.getenv("ENTRY_MAX_SL_ATR_MULT", "6.0"))
# 銘柄別の最大SL幅上書き（ATR倍率使用時のみ）
ENTRY_MAX_SL_ATR_MULT_BY_SYMBOL: dict = {
    "GOLD": 5.0,        # GOLD# ATR M15≈$20-40 → 上限$100-200
}
EXIT_MIN_CONFIDENCE = int(os.getenv("EXIT_MIN_CONFIDENCE", "45"))
FORCE_EXIT_ON_PREMISE_BREAK = os.getenv("FORCE_EXIT_ON_PREMISE_BREAK", "true").lower() == "true"
MIN_HOLD_MINUTES_BEFORE_FORCE_PREMISE_BREAK = int(os.getenv("MIN_HOLD_MINUTES_BEFORE_FORCE_PREMISE_BREAK", "30"))
EXIT_EARLY_WINDOW_MINUTES = int(os.getenv("EXIT_EARLY_WINDOW_MINUTES", "30"))
EXIT_MIN_CONFIDENCE_EARLY = int(os.getenv("EXIT_MIN_CONFIDENCE_EARLY", "65"))
# 価格変化がこの%未満ならExit AIチェックをスキップ (0.0 で常時実行)
EXIT_AI_SKIP_MIN_MOVE_PCT = float(os.getenv("EXIT_AI_SKIP_MIN_MOVE_PCT", "0.0"))
# 一時的に導入した微小変化強制チェックは廃止 (機械EXIT優先化のため常時無効)
EXIT_AI_FORCE_CHECK_EVERY_CYCLES = 0
EXIT_AI_NO_SKIP_WHEN_LOSS = False
# TP目前の機械利確判定 (AI前に実行)
EXIT_TP_NEAR_ENABLED = os.getenv("EXIT_TP_NEAR_ENABLED", "true").lower() == "true"
# TPまでの残距離 <= max(ATR * この倍率, 1R * この倍率) なら「目前」
EXIT_TP_NEAR_ATR_MULT = float(os.getenv("EXIT_TP_NEAR_ATR_MULT", "0.25"))
EXIT_TP_NEAR_R_MULT = float(os.getenv("EXIT_TP_NEAR_R_MULT", "0.15"))

# ──────────────────────────────────────
# SMC (Smart Money Concepts) フィルタ設定
# ──────────────────────────────────────
# SMCフィルタを有効にすると、エントリー条件にLiquidity Sweepの確認が追加される
SMC_FILTER_ENABLED = os.getenv("SMC_FILTER_ENABLED", "true").lower() == "true"
# Liquidity Sweepと判定するための最小侵食幅 (ATR比率)
# 例: 0.25 → 高安値をATRの25%以上超えた場合にSweep認定 (浅すぎるノイズを排除)
SMC_SWEEP_ATR_MULT = float(os.getenv("SMC_SWEEP_ATR_MULT", "0.1"))
# Sweep侵食深さの上限係数 (深すぎる=ブレイクアウトとして除外)
# デフォルト: ATR×0.5未満 (USDJPY/EURUSD等ボラ安定銘柄向け)
SMC_SWEEP_MAX_ATR_MULT = float(os.getenv("SMC_SWEEP_MAX_ATR_MULT", "0.5"))
# ボラ大銘柄は侵食上限を緩和 (大きなヒゲが出やすいため)
SMC_SWEEP_MAX_ATR_MULT_BY_SYMBOL: dict = {
    "GOLD":      1.5,   # GOLD: ボラ大、深いSweepも有効
    "US100Cash": 1.5,   # US100: 同上
    "OILCash":   1.2,   # OIL: 中程度に緩和
}
# SweepローソクのヒゲがTotal Range(高値〜安値)に占める最低割合
# 0.50: 上/下ヒゲが足全体の50%以上 = 明確なピンバー要件
SMC_SWEEP_MIN_WICK_RATIO = float(os.getenv("SMC_SWEEP_MIN_WICK_RATIO", "0.50"))
# ボラティリティフィルター: 現在ATRをATR SMAと比較するためのSMA期間
SMC_SWEEP_ATR_SMA_PERIOD = int(os.getenv("SMC_SWEEP_ATR_SMA_PERIOD", "50"))
# ボラティリティフィルター有効化 (False=閑散相場フィルタ無効)
SMC_SWEEP_VOLATILITY_FILTER_ENABLED = os.getenv("SMC_SWEEP_VOLATILITY_FILTER_ENABLED", "true").lower() == "true"
# ADXフィルター: 方向感のない閑散相場 (ADX < 閾値) でのエントリーを全面ブロック
ADX_FILTER_ENABLED = os.getenv("ADX_FILTER_ENABLED", "true").lower() == "true"
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
# デフォルト閾値: ADX < 20 は方向感なし → エントリーしない
ADX_MIN_THRESHOLD = float(os.getenv("ADX_MIN_THRESHOLD", "20.0"))
# 銘柄別ADX閾値オーバーライド (デフォルトは ADX_MIN_THRESHOLD)
ADX_MIN_THRESHOLD_BY_SYMBOL: dict = {
    "GOLD":      20.0,
    "US100Cash": 20.0,
    "OILCash":   20.0,
    "USDJPY":    20.0,
    "EURUSD":    20.0,
}
# 機械的SMCゲート: AI呼び出し前に数値条件でフィルタリング
# Falseにすると機械ゲートをスキップしてAIのみで判断
SMC_MECHANICAL_GATE_ENABLED = os.getenv("SMC_MECHANICAL_GATE_ENABLED", "true").lower() == "true"
# Sweepを探す遡り期間 (H1バー数)
SMC_SWEEP_LOOKBACK_BARS = int(os.getenv("SMC_SWEEP_LOOKBACK_BARS", "12"))
# 順張り (Continuation BOS) エントリーを有効にする
SMC_CONTINUATION_ENABLED = os.getenv("SMC_CONTINUATION_ENABLED", "true").lower() == "true"
# 逆張り (Reversal Sweep) エントリーを有効にする
SMC_REVERSAL_ENABLED = os.getenv("SMC_REVERSAL_ENABLED", "true").lower() == "true"
# 順張りBOS判定: MAのスロープを見る遡り期間 (H1バー数)
SMC_CONTINUATION_BOS_LOOKBACK_BARS = int(os.getenv("SMC_CONTINUATION_BOS_LOOKBACK_BARS", "5"))
# 順張りBOS判定: MA傾きの最小値 (ATR比率) — この値未満のMA傾きはトレンドなしとみなす
# 0.15: 以前の0.3から緩和、緩やかなトレンドでもCONTINUATION_BOSが発動するように
SMC_CONTINUATION_MA_SLOPE_ATR_MULT = float(os.getenv("SMC_CONTINUATION_MA_SLOPE_ATR_MULT", "0.15"))
# 順張りBOS判定: MAから現在価格までの最大距離 (ATR倍率)
# 0.5 = 旧来の厳格設定(押し目必須), 5.0 = 緩和(トレンド継続中でもAIに渡す)
SMC_CONTINUATION_PRICE_MAX_ATR_MULT = float(os.getenv("SMC_CONTINUATION_PRICE_MAX_ATR_MULT", "5.0"))
# 機械ゲートRR判定の緩和係数 (1.0未満で緩和、0.5〜1.0の範囲推奨)
try:
    SMC_MECHANICAL_RR_RELAX_FACTOR = max(0.5, min(1.0, float(os.getenv("SMC_MECHANICAL_RR_RELAX_FACTOR", "0.75"))))
except ValueError:
    SMC_MECHANICAL_RR_RELAX_FACTOR = 0.75

# 市場クローズ時の無駄なAI判定を防ぐため、ティックが古い銘柄は停止中とみなす
MARKET_DATA_STALE_SEC = int(os.getenv("MARKET_DATA_STALE_SEC", "1800"))
LOCK_PROFIT_2_R = float(os.getenv("LOCK_PROFIT_2_R", "1.00"))
SYMBOL_LOSS_STREAK_PAUSE_TRIGGER = int(os.getenv("SYMBOL_LOSS_STREAK_PAUSE_TRIGGER", "3"))
SYMBOL_LOSS_STREAK_COOLDOWN_MINUTES = int(os.getenv("SYMBOL_LOSS_STREAK_COOLDOWN_MINUTES", "180"))
# 同銘柄クールダウン: 直近クローズ後、一定本数は再エントリー禁止
SYMBOL_REENTRY_COOLDOWN_ALL_EXITS_ENABLED = os.getenv("SYMBOL_REENTRY_COOLDOWN_ALL_EXITS_ENABLED", "true").lower() == "true"
SYMBOL_REENTRY_COOLDOWN_ALL_EXITS_BARS = max(0, int(os.getenv("SYMBOL_REENTRY_COOLDOWN_ALL_EXITS_BARS", "3")))
# 勝ちトレード後の再エントリー抑制 (ALL_EXITS_BARS=3本で勝ち・負け共通CDが機能するため無効化)
SYMBOL_REENTRY_COOLDOWN_AFTER_WIN_ENABLED = os.getenv("SYMBOL_REENTRY_COOLDOWN_AFTER_WIN_ENABLED", "false").lower() == "true"
SYMBOL_REENTRY_COOLDOWN_AFTER_WIN_BARS = max(0, int(os.getenv("SYMBOL_REENTRY_COOLDOWN_AFTER_WIN_BARS", "4")))
PREMISE_BREAK_REENTRY_BLOCK_ENABLED = os.getenv("PREMISE_BREAK_REENTRY_BLOCK_ENABLED", "true").lower() == "true"
PREMISE_BREAK_REENTRY_BLOCK_BARS = max(0, int(os.getenv("PREMISE_BREAK_REENTRY_BLOCK_BARS", "4")))  # ALL_EXITS_BARS(3本=45分)より長くして同方向ブロックを有効化

# ──────────────────────────────────────
# ニュース監視設定
# ──────────────────────────────────────
# ForexFactory 経済カレンダーによるエントリーブロック
NEWS_CALENDAR_ENABLED = os.getenv("NEWS_CALENDAR_ENABLED", "true").lower() == "true"
# カレンダー取得間隔 (分)
NEWS_CALENDAR_INTERVAL_MINUTES = max(10, int(os.getenv("NEWS_CALENDAR_INTERVAL_MINUTES", "60")))
# 高インパクト指標の前後何分をブロックするか
NEWS_EVENT_BLOCK_MINUTES = max(0, int(os.getenv("NEWS_EVENT_BLOCK_MINUTES", "30")))

# gpt-5-nano 突発ニュースポーリング
NEWS_MONITOR_ENABLED = os.getenv("NEWS_MONITOR_ENABLED", "true").lower() == "true"
# ポーリング間隔 (分) — 短すぎるとAPIコストが増えるため最低15分
NEWS_MONITOR_INTERVAL_MINUTES = max(15, int(os.getenv("NEWS_MONITOR_INTERVAL_MINUTES", "30")))
# ニュース確認に使うモデル (nanoで十分)
NEWS_MONITOR_MODEL = os.getenv("NEWS_MONITOR_MODEL", "gpt-5-nano")
# キャッシュ有効期限 (分) — ポーリング間隔より少し長めに設定
NEWS_CACHE_EXPIRE_MINUTES = max(15, int(os.getenv("NEWS_CACHE_EXPIRE_MINUTES", "45")))
# MEDIUMリスクもエントリーブロックするか (デフォルト=False でHIGHのみブロック)
NEWS_BLOCK_ON_MEDIUM = os.getenv("NEWS_BLOCK_ON_MEDIUM", "false").lower() == "true"

# ──────────────────────────────────────
# Finnhub ニュースAPI
# ──────────────────────────────────────
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
# Finnhubキーワードスコアがこの値以上の場合のみ gpt-5-nano で詳細判断 (0=常にAI, 大=AI呼び出し激減)
NEWS_AI_ESCALATION_SCORE = max(0, int(os.getenv("NEWS_AI_ESCALATION_SCORE", "6")))

# ──────────────────────────────────────
# 市場ストレス検知 (スプレッド/ATR急変)
# ──────────────────────────────────────
# スプレッドが平常時の何倍以上でストレス検知してエントリーブロックするか
MARKET_STRESS_SPREAD_RATIO = float(os.getenv("MARKET_STRESS_SPREAD_RATIO", "3.0"))
# スプレッドがこの倍率以上 かつ GPTがHIGH → 既存ポジションもクローズ (より高い閾値)
MARKET_STRESS_SPREAD_CLOSE_RATIO = float(os.getenv("MARKET_STRESS_SPREAD_CLOSE_RATIO", "5.0"))
# ATRが直近20本平均の何倍以上でストレス検知するか
MARKET_STRESS_ATR_RATIO    = float(os.getenv("MARKET_STRESS_ATR_RATIO", "2.5"))
# ストレス検知後に gpt-5-nano を呼び出すか (False=固定TTLのみ)
MARKET_STRESS_AI_ENABLED   = os.getenv("MARKET_STRESS_AI_ENABLED", "true").lower() == "true"
# GPTが提案するhold_minutesの最小・最大クランプ (分)
MARKET_STRESS_HOLD_MIN_MIN = int(os.getenv("MARKET_STRESS_HOLD_MIN_MIN", "10"))  # 30→10分に短縮
MARKET_STRESS_HOLD_MAX_MIN = int(os.getenv("MARKET_STRESS_HOLD_MAX_MIN", "480"))
# hold_until を過ぎてからこの分数後は、スプレッド/ATR高止まりでも強制解除する絶対期限
MARKET_STRESS_FORCE_CLEAR_GRACE_MIN = int(os.getenv("MARKET_STRESS_FORCE_CLEAR_GRACE_MIN", "30"))
# スプレッドが平常比 1.5 倍未満に戻ったら解除条件を満たすとみなす
MARKET_STRESS_SPREAD_CLEAR_RATIO = float(os.getenv("MARKET_STRESS_SPREAD_CLEAR_RATIO", "1.5"))
# ATRが平常比 1.3 倍未満に戻ったら解除条件を満たすとみなす
MARKET_STRESS_ATR_CLEAR_RATIO = float(os.getenv("MARKET_STRESS_ATR_CLEAR_RATIO", "1.3"))
# 平常スプレッドのベースライン計算に使う過去サンプル数 (銘柄ごと)
MARKET_STRESS_SPREAD_BASELINE_N = int(os.getenv("MARKET_STRESS_SPREAD_BASELINE_N", "50"))
# ストレス解除直後の慎重モード: 最初 N 回のエントリーはロットを縮小する
POST_RECOVERY_TRADE_COUNT      = int(os.getenv("POST_RECOVERY_TRADE_COUNT", "1"))
POST_RECOVERY_LOT_MULTIPLIER   = float(os.getenv("POST_RECOVERY_LOT_MULTIPLIER", "0.5"))

# ──────────────────────────────────────
# チャート画像設定
# ──────────────────────────────────────
CHART_BARS = 100        # 生成チャートに表示するバー数
CHART_WIDTH = 1200      # px
CHART_HEIGHT = 750      # px (Volumeパネル廃止でメインパネル高さを拡張)
# 描画密度制御: 現在価格に近い順で表示する最大本数
SMC_DRAW_MAX_LIQUIDITY_PER_SIDE = max(1, int(os.getenv("SMC_DRAW_MAX_LIQUIDITY_PER_SIDE", "3")))
SMC_DRAW_MAX_SWING_PER_SIDE = max(1, int(os.getenv("SMC_DRAW_MAX_SWING_PER_SIDE", "6")))

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
DB_MAX_AI_LOG_ROWS = int(os.getenv("DB_MAX_AI_LOG_ROWS", "5000"))
DB_MAX_HEARTBEAT_ROWS = int(os.getenv("DB_MAX_HEARTBEAT_ROWS", "2000"))

# ──────────────────────────────────────
# Expectancy Dashboard
# ──────────────────────────────────────
DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() == "true"
DASHBOARD_LOOKBACK_DAYS = int(os.getenv("DASHBOARD_LOOKBACK_DAYS", "90"))

# ──────────────────────────────────────
# Adaptive Learning (ローリング窓 閾値自動更新)
# ──────────────────────────────────────
ADAPTIVE_ENABLED             = os.getenv("ADAPTIVE_ENABLED", "true").lower() == "true"
ADAPTIVE_LOOKBACK_DAYS       = int(os.getenv("ADAPTIVE_LOOKBACK_DAYS", "7"))
ADAPTIVE_MIN_SAMPLES         = int(os.getenv("ADAPTIVE_MIN_SAMPLES", "10"))
ADAPTIVE_CONF_STEP           = int(os.getenv("ADAPTIVE_CONF_STEP", "3"))
ADAPTIVE_CONF_MIN            = int(os.getenv("ADAPTIVE_CONF_MIN", "65"))
ADAPTIVE_CONF_MAX            = int(os.getenv("ADAPTIVE_CONF_MAX", "78"))
ADAPTIVE_CONF_MAX_WEEKLY_DELTA = int(os.getenv("ADAPTIVE_CONF_MAX_WEEKLY_DELTA", "6"))
ADAPTIVE_LLM_ENABLED         = os.getenv("ADAPTIVE_LLM_ENABLED", "true").lower() == "true"
ADAPTIVE_LLM_MODEL           = os.getenv("ADAPTIVE_LLM_MODEL", "gpt-5.4")
ADAPTIVE_LLM_INTERVAL_SEC    = int(os.getenv("ADAPTIVE_LLM_INTERVAL_SEC", "604800"))

# ──────────────────────────────────────
# パス
# ──────────────────────────────────────
BASE_DIR = _CURRENT_DIR
LOG_DIR = os.path.join(BASE_DIR, "logs")
DB_PATH = os.path.join(BASE_DIR, "trades.db")
SCREENSHOT_DIR = os.path.join(BASE_DIR, "screenshots")
ANALYTICS_DIR = os.path.join(BASE_DIR, "analytics")

# ディレクトリ作成
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(ANALYTICS_DIR, exist_ok=True)
