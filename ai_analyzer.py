"""OpenAI Responses API 連携モジュール

GPT-5 mini + web_search_preview を使用して:
  1. H1 + M15 チャート画像からエントリー判断
  2. 保有ポジションのエグジット判断
  3. ニュース検索による市場環境評価
"""

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

import config
import chart_capture

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


# ── レスポンス構造体 ────────────────────

@dataclass
class EntrySignal:
    decision: str       # "BUY", "SELL", "SKIP"
    confidence: int     # 0-100
    h1_trend: str       # "UP", "DOWN", "SIDEWAYS"
    m15_signal: str
    alignment: bool     # H1とM15のトレンド一致
    reasoning: str
    news_impact: str
    sl_distance: float  # 推奨SL幅 (price単位)
    raw_response: str   # AI生テキスト
    approved_by_final_model: bool = False
    final_model_name: str = ""


@dataclass
class ExitSignal:
    decision: str       # "HOLD", "EXIT"
    confidence: int
    reasoning: str
    news_impact: str
    raw_response: str


# ── エントリー分析 ──────────────────────

def analyze_entry(symbol: str, current_price: float,
                  atr_h1: float, atr_m15: float,
                  h1_image: bytes, m15_image: bytes,
                  balance: float) -> EntrySignal:
    """H1 + M15 のチャート画像を送り、エントリー判断を取得する"""

    h1_b64 = chart_capture.chart_to_base64(h1_image)
    m15_b64 = chart_capture.chart_to_base64(m15_image)

    prompt = f"""あなたはプロのテクニカルアナリストです。
以下の2枚のチャート画像（1時間足と15分足）を分析し、{symbol}のエントリー判断をしてください。

【現在の市場データ】
- 銘柄: {symbol}
- 現在価格: {current_price}
- M15 ATR(14): {atr_m15:.5f}
- H1 ATR(14): {atr_h1:.5f}
- 口座残高: ¥{balance:,.0f}

【分析手順】
1. H1足でトレンド方向を確認
2. M15足で20MA付近への押し目・戻りを確認
3. H1とM15の方向性が「一致」しているか判定
4. web検索で{symbol}に関連する直近のニュースや経済イベントを確認

【売買哲学】
- 損切り貧乏を避け、利大損小を最優先してください
- エントリーは「伸びる余地が十分にある局面」に限定し、迷う局面は SKIP を選んでください
- 単なる小さな押しや戻りを反転と誤認しないでください
- SLは近すぎて通常ノイズで刈られないよう、M15の構造とATRに対して妥当な余裕を持たせてください
- 利幅が取りにくいレンジ気味の局面や、イベント前で不確実性が高い局面は SKIP を優先してください

【回答フォーマット (JSON)】
{{
    "decision": "BUY" or "SELL" or "SKIP",
    "confidence": 0-100,
    "h1_trend": "UP" or "DOWN" or "SIDEWAYS",
    "m15_signal": "押し目/戻り等の説明",
    "alignment": true or false,
    "reasoning": "判断理由の詳細",
    "news_impact": "関連ニュースの要約",
    "sl_distance": SL幅の数値(price単位、ATRの1.5倍を目安)
}}

重要: 上位足と下位足のトレンドが一致しない場合は "SKIP" にしてください。
自信度が60未満の場合も "SKIP" にしてください。
期待値が低く、損切りだけが増えやすい局面も "SKIP" にしてください。
必ずJSON形式のみで回答してください。"""

    try:
        client = _get_client()
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{h1_b64}",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{m15_b64}",
                        },
                    ],
                }
            ],
            tools=[{"type": "web_search_preview"}],
        )

        raw_text = response.output_text
        logger.info("[AI Entry] %s raw response length: %d", symbol, len(raw_text))

        primary_signal = _parse_entry_response(raw_text, atr_m15)

        if _should_run_final_approval(symbol, primary_signal):
            final_signal = _run_entry_final_approval(
                symbol=symbol,
                current_price=current_price,
                atr_h1=atr_h1,
                atr_m15=atr_m15,
                balance=balance,
                h1_b64=h1_b64,
                m15_b64=m15_b64,
                primary_signal=primary_signal,
            )
            logger.info(
                "[AI Entry] %s final approval=%s by=%s",
                symbol, final_signal.decision, final_signal.final_model_name,
            )
            return final_signal

        return primary_signal

    except Exception as e:
        logger.error("[AI Entry] API呼び出しエラー: %s", e)
        return EntrySignal(
            decision="SKIP", confidence=0,
            h1_trend="UNKNOWN", m15_signal="API Error",
            alignment=False, reasoning=str(e),
            news_impact="", sl_distance=atr_m15 * 1.5,
            raw_response=str(e),
        )


# ── エグジット分析 ──────────────────────

def analyze_exit(symbol: str, direction: str, entry_price: float,
                 current_price: float, unrealized_pnl: float,
                 hold_minutes: int, m15_image: bytes) -> ExitSignal:
    """保有ポジションのエグジット判断"""

    m15_b64 = chart_capture.chart_to_base64(m15_image)

    prompt = f"""あなたはプロのテクニカルアナリストです。
以下の保有ポジションについてエグジットすべきか判断してください。

【ポジション情報】
- 銘柄: {symbol}
- 方向: {direction}
- エントリー価格: {entry_price}
- 現在価格: {current_price}
- 含み損益: ¥{unrealized_pnl:,.0f}
- 保有時間: {hold_minutes}分

15分足のチャート画像を添付しています。

【分析手順】
1. 現在のトレンドがポジション方向と一致しているか
2. 反転シグナルの有無
3. web検索で直近のニュースを確認
4. 利確/損切りの判断

【ポジション管理方針】
- 損切り貧乏を避けるため、軽微な押し戻しや一時的なノイズだけでは EXIT しないでください
- 利大損小を重視し、利益が伸びる余地があるなら HOLD を優先してください
- EXIT は「M15構造でトレンド継続が崩れた」「ニュースで前提が壊れた」「利益保護の明確な必要がある」場合に限定してください
- 単に含み益がある、あるいは数本停滞しただけでは EXIT しないでください
- 明確な高値切り下げ、安値切り上げ失敗、MAの傾き悪化、構造崩れを重視してください

【回答フォーマット (JSON)】
{{
    "decision": "HOLD" or "EXIT",
    "confidence": 0-100,
    "reasoning": "判断理由の詳細",
    "news_impact": "関連ニュースの要約"
}}

基本方針は HOLD 寄りで、明確な根拠がある場合のみ EXIT にしてください。
必ずJSON形式のみで回答してください。"""

    try:
        client = _get_client()
        response = client.responses.create(
            model=config.OPENAI_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{m15_b64}",
                        },
                    ],
                }
            ],
            tools=[{"type": "web_search_preview"}],
        )

        raw_text = response.output_text
        logger.info("[AI Exit] %s raw response length: %d", symbol, len(raw_text))

        return _parse_exit_response(raw_text)

    except Exception as e:
        logger.error("[AI Exit] API呼び出しエラー: %s", e)
        return ExitSignal(
            decision="HOLD", confidence=0,
            reasoning=str(e), news_impact="",
            raw_response=str(e),
        )


def _should_run_final_approval(symbol: str, signal: EntrySignal) -> bool:
    if not config.OPENAI_FINAL_APPROVAL_ENABLED:
        return False
    if signal.decision not in {"BUY", "SELL"}:
        return False
    if not signal.alignment:
        return False
    return symbol in config.FINAL_APPROVAL_SYMBOLS or signal.confidence >= config.FINAL_APPROVAL_MIN_CONFIDENCE


def _run_entry_final_approval(symbol: str, current_price: float,
                              atr_h1: float, atr_m15: float,
                              balance: float, h1_b64: str, m15_b64: str,
                              primary_signal: EntrySignal) -> EntrySignal:
    prompt = f"""あなたは最終承認を担当するシニア市場アナリストです。
一次判定モデルが {symbol} に対して以下の判断を出しました。

【一次判定】
- decision: {primary_signal.decision}
- confidence: {primary_signal.confidence}
- h1_trend: {primary_signal.h1_trend}
- m15_signal: {primary_signal.m15_signal}
- alignment: {primary_signal.alignment}
- reasoning: {primary_signal.reasoning}
- news_impact: {primary_signal.news_impact}
- sl_distance: {primary_signal.sl_distance}

【現在の市場データ】
- 現在価格: {current_price}
- M15 ATR(14): {atr_m15:.5f}
- H1 ATR(14): {atr_h1:.5f}
- 口座残高: ¥{balance:,.0f}

【あなたの役割】
一次判定を盲信せず、ダマシ・イベント前・短期過熱・トレンド継続性を再評価してください。
特に「今この足で本当に入る価値があるか」を厳しく判定してください。

【最終承認ポリシー】
- 損切り貧乏を避けるため、少しでも期待値が怪しければ SKIP にしてください
- 「当たるか」より「利大損小の形になっているか」を優先してください
- 近すぎるSLでノイズに刈られそうなエントリーは承認しないでください
- 上位足が素直に後押しし、利益の伸び代が大きいときだけ承認してください

【承認ルール】
- 承認するなら一次判定と同じ方向を返す
- 少しでも根拠が弱いなら SKIP
- JSONのみで回答

【回答フォーマット】
{{
    "decision": "{primary_signal.decision}" or "SKIP",
    "confidence": 0-100,
    "h1_trend": "UP" or "DOWN" or "SIDEWAYS",
    "m15_signal": "短期構造の要約",
    "alignment": true or false,
    "reasoning": "最終承認の理由",
    "news_impact": "ニュースとイベントの評価",
    "sl_distance": 数値
}}"""

    try:
        client = _get_client()
        response = client.responses.create(
            model=config.OPENAI_FINAL_APPROVAL_MODEL,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{h1_b64}",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{m15_b64}",
                        },
                    ],
                }
            ],
            tools=[{"type": "web_search_preview"}],
            reasoning={"effort": config.OPENAI_FINAL_APPROVAL_REASONING_EFFORT},
        )

        raw_text = response.output_text
        final_signal = _parse_entry_response(raw_text, atr_m15)
        final_signal.raw_response = primary_signal.raw_response + "\n\n--- FINAL APPROVAL ---\n\n" + raw_text
        final_signal.approved_by_final_model = final_signal.decision in {"BUY", "SELL"}
        final_signal.final_model_name = config.OPENAI_FINAL_APPROVAL_MODEL
        return final_signal
    except Exception as e:
        logger.error("[AI Entry Final] API呼び出しエラー: %s", e)
        return EntrySignal(
            decision="SKIP",
            confidence=0,
            h1_trend=primary_signal.h1_trend,
            m15_signal=primary_signal.m15_signal,
            alignment=False,
            reasoning=f"Final approval failed: {e}",
            news_impact=primary_signal.news_impact,
            sl_distance=primary_signal.sl_distance,
            raw_response=primary_signal.raw_response + "\n\n--- FINAL APPROVAL ERROR ---\n\n" + str(e),
            approved_by_final_model=False,
            final_model_name=config.OPENAI_FINAL_APPROVAL_MODEL,
        )


# ── レスポンスパーサ ────────────────────

def _parse_entry_response(raw_text: str, fallback_atr: float) -> EntrySignal:
    """AIレスポンスからJSONを抽出してEntrySignalに変換"""
    data = _extract_json(raw_text)
    if data is None:
        logger.warning("EntryレスポンスのJSON解析失敗 → SKIP")
        return EntrySignal(
            decision="SKIP", confidence=0,
            h1_trend="UNKNOWN", m15_signal="JSON parse error",
            alignment=False, reasoning=raw_text[:300],
            news_impact="", sl_distance=fallback_atr * 1.5,
            raw_response=raw_text,
        )

    return EntrySignal(
        decision=data.get("decision", "SKIP").upper(),
        confidence=int(data.get("confidence", 0)),
        h1_trend=data.get("h1_trend", "UNKNOWN").upper(),
        m15_signal=data.get("m15_signal", ""),
        alignment=bool(data.get("alignment", False)),
        reasoning=data.get("reasoning", ""),
        news_impact=data.get("news_impact", ""),
        sl_distance=float(data.get("sl_distance", fallback_atr * 1.5)),
        raw_response=raw_text,
    )


def _parse_exit_response(raw_text: str) -> ExitSignal:
    """AIレスポンスからJSONを抽出してExitSignalに変換"""
    data = _extract_json(raw_text)
    if data is None:
        logger.warning("ExitレスポンスのJSON解析失敗 → HOLD")
        return ExitSignal(
            decision="HOLD", confidence=0,
            reasoning=raw_text[:300], news_impact="",
            raw_response=raw_text,
        )

    return ExitSignal(
        decision=data.get("decision", "HOLD").upper(),
        confidence=int(data.get("confidence", 0)),
        reasoning=data.get("reasoning", ""),
        news_impact=data.get("news_impact", ""),
        raw_response=raw_text,
    )


def _extract_json(text: str) -> dict | None:
    """テキストからJSON部分を抽出する (```json ... ``` にも対応)"""
    # まず直接パース
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # ```json ... ``` ブロックを探す
    import re
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # { ... } ブロックを探す
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
