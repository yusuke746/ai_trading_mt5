"""相関リスク制御モジュール

ルール: USD関連、またはJPY関連のポジションは同時に「最大2つ」まで。
例: USDJPYとGOLD（対ドル）を保有している場合、EURUSDの新規エントリーはスキップ。
"""

import logging

import config
import mt5_connector

logger = logging.getLogger(__name__)


def can_open_position(symbol: str) -> tuple[bool, str]:
    """新規ポジションを建ててよいかチェックする。

    Returns:
        (可否, 理由メッセージ)
    """
    # 対象銘柄の通貨グループ取得
    groups = config.CURRENCY_GROUPS.get(symbol, [])
    if not groups:
        logger.warning("CURRENCY_GROUPS未定義: %s → チェックスキップ", symbol)
        return True, ""

    # 現在の保有ポジションのシンボル一覧
    open_symbols = mt5_connector.get_all_open_symbols()

    # 同一銘柄に既にポジションがあればスキップ
    if symbol in open_symbols:
        return False, f"{symbol} は既にポジション保有中"

    # 各通貨グループについて、保有ポジション数をカウント
    for currency in groups:
        count = _count_positions_in_group(currency, open_symbols)
        if count >= config.MAX_CORRELATED_POSITIONS:
            return (
                False,
                f"{currency}グループ: 既に{count}ポジション保有中 "
                f"(上限{config.MAX_CORRELATED_POSITIONS})",
            )

    return True, ""


def _count_positions_in_group(currency: str, open_symbols: list[str]) -> int:
    """指定通貨に関連する保有ポジション数をカウント"""
    count = 0
    for sym in open_symbols:
        sym_groups = config.CURRENCY_GROUPS.get(sym, [])
        if currency in sym_groups:
            count += 1
    return count


def get_exposure_summary() -> dict[str, list[str]]:
    """各通貨グループの現在のエクスポージャーを返す (通知・デバッグ用)"""
    open_symbols = mt5_connector.get_all_open_symbols()
    summary: dict[str, list[str]] = {}

    for sym in open_symbols:
        for currency in config.CURRENCY_GROUPS.get(sym, []):
            summary.setdefault(currency, []).append(sym)

    return summary
