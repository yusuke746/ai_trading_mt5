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
    tp_distance: float  # 推奨TP幅 (price単位)
    raw_response: str   # AI生テキスト
    # SMC分析結果
    smc_liquidity_sweep: bool = False   # Liquidity Sweep発生
    smc_sweep_direction: str = "NONE"   # "HIGH" / "LOW" / "NONE"
    smc_ob_confirmed: bool = False      # Order Block再テスト確認
    smc_fvg_present: bool = False       # Fair Value Gap存在
    approved_by_final_model: bool = False
    final_model_name: str = ""


@dataclass
class ExitSignal:
    decision: str       # "HOLD", "EXIT"
    confidence: int
    entry_premise_valid: bool
    reasoning: str
    news_impact: str
    raw_response: str


# ── エントリー分析 ──────────────────────

def analyze_entry(symbol: str, current_price: float,
                  atr_h1: float, atr_m15: float,
                  h1_image: bytes, m15_image: bytes,
                  balance: float,
                  smc_data: dict | None = None,
                  mech_gate: dict | None = None) -> EntrySignal:
    """H1 + M15 のチャート画像を送り、エントリー判断を取得する

    smc_data (optional): mt5_connector.get_price_levels() の戻り値
        pdh, pdl, pwh, pwl, swing_highs, swing_lows
    mech_gate (optional): _mechanical_smc_gate() の結果
        sweep_pass, bos_pass, rr_pass, sweep_type
    """

    h1_b64 = chart_capture.chart_to_base64(h1_image)
    m15_b64 = chart_capture.chart_to_base64(m15_image)

    # エントリータイプを先に特定しておく
    _mech_entry_type = mech_gate.get("entry_type", "REVERSAL_SWEEP") if mech_gate else "REVERSAL_SWEEP"

    # ── 機械ゲート結果をプロンプトに組み込む ──
    if mech_gate:
        _entry_type_label = {
            "REVERSAL_SWEEP":  "逆張り (Liquidity Sweep後の反転)",
            "CONTINUATION_BOS": "順張り (BOS後の押し目/戻し)",
        }.get(_mech_entry_type, "不明")
        mech_section = f"""
【機械判定ゲート結果 (Python数値計算済・確定事実)】
- エントリータイプ: {_mech_entry_type} ({_entry_type_label})
- 価格レベルSweep検出: {mech_gate.get('sweep_pass', 'N/A')} (方向: {mech_gate.get('sweep_type', 'N/A')})
- BOS/トレンド確認 (MA方向): {mech_gate.get('bos_pass', 'N/A')}
- RR充足 (最低{config.ENTRY_MIN_TP_R:.1f}R以上): {mech_gate.get('rr_pass', 'N/A')}
※ これらは客観的数値判定です。チャートと照合して最終評価してください。
"""
    else:
        mech_section = ""

    # ── SMC価格レベル情報をプロンプトに組み込む ──
    if smc_data and config.SMC_FILTER_ENABLED:
        smc_section = f"""
【SMC価格レベル (Python計算済み)】
- 前日高値 (PDH): {smc_data.get('pdh', 'N/A')}
- 前日安値 (PDL): {smc_data.get('pdl', 'N/A')}
- 前週高値 (PWH): {smc_data.get('pwh', 'N/A')}
- 前週安値 (PWL): {smc_data.get('pwl', 'N/A')}
- H1スウィング高値 (直近5個): {smc_data.get('swing_highs', [])}
- H1スウィング安値 (直近5個): {smc_data.get('swing_lows', [])}
"""
        if _mech_entry_type == "CONTINUATION_BOS":
            smc_filter_instruction = f"""
【SMCフィルタ — 順張り必須条件 (すべてNOの場合はSKIP)】
機械ゲートが「順張り (BOS後の押し目/戻し)」を検出しています。以下の3条件をANDで評価してください:
  必須①: H1トレンド方向 と M15のオーダーフロー方向が一致
  必須②: 直近のBOS（構造的突破）が明確に確認でき、トレンド方向が確立されているか
  必須③: 現在の押し目/戻しがOrder Block (OB) またはFair Value Gap (FVG) に到達し、
          そこで反転シグナル（長ヒゲ・Engulfing等）が出ているか

【SMC加点条件 (1つ以上あると信頼度UP)】
  加点A: 複数回のBOSによるトレンド継続確認 (高値切り上げ / 安値切り下げ)
  加点B: OBまたはFVGでの明確な反応確認
  加点C: 上位足 (H4/D1) でも同方向のトレンドが継続中
※ このセットアップではLiquidity Sweepは必須ではありません。smc_ob_confirmed を重視してください。
"""
        else:
            smc_filter_instruction = f"""
【SMCフィルタ — 逆張り必須条件 (すべてNOの場合はSKIP)】
機械ゲートが「逆張り (Liquidity Sweep後の反転)」を検出しています。以下の3条件をANDで評価してください:
  必須①: H1トレンド方向 と M15のオーダーフロー方向が一致
  必須②: 上記のSMC価格レベル（PDH/PDL/PWH/PWLまたはスウィング高安値）を
          チャート上で価格が侵食・反転した「Liquidity Sweep」が直近で発生しているか
          ※ ATR({atr_h1:.5f})の{config.SMC_SWEEP_ATR_MULT}倍以上の侵食 → Sweep認定
  必須③: Sweep後に明確な反転アクション（長ヒゲ・Engulfing・BOS）が出ているか

【SMC加点条件 (1つ以上あると信頼度UP)】
  加点A: Sweepで形成されたOrder Block (OB) への再テスト
  加点B: Sweep後に生じたFair Value Gap (FVG) を埋める動き
"""
    else:
        smc_section = ""
        smc_filter_instruction = ""

    prompt = f"""あなたはプロのSMC(Smart Money Concepts)アナリストです。
以下の2枚のチャート画像（1時間足と15分足）を分析し、{symbol}のエントリー判断をしてください。
{mech_section}{smc_section}
【現在の市場データ】
- 銘柄: {symbol}
- 現在価格: {current_price}
- M15 ATR(14): {atr_m15:.5f}
- H1 ATR(14): {atr_h1:.5f}
- 口座残高: ¥{balance:,.0f}
{smc_filter_instruction}
【分析手順】
1. H1足でトレンド方向とオーダーフローを確認
2. 上記SMC価格レベルとチャートを照合し、Liquidity Sweepの有無を判定
3. Sweep後の反転アクション (BOS/OB/FVG) を確認
4. M15足でエントリータイミングを精査
5. web検索で{symbol}に関連する直近のニュースや経済イベントを確認

【売買哲学】
- 損切り貧乏を避け、利大損小を最優先してください
- 逆張りエントリーは「Sweepによる流動性回収後の反転局面」に限定し、SLはSweep起点の直外側に置く
- 順張りエントリーは「BOS後の押し目/戻しでOB/FVGが確認できる局面」に限定し、SLは直近スウィング安値/高値の外側に置く
- 利幅が取りにくいレンジ局面やイベント前で不確実性が高い局面は SKIP にしてください

【回答フォーマット (JSON)】
{{
    "decision": "BUY" or "SELL" or "SKIP",
    "confidence": 0-100,
    "h1_trend": "UP" or "DOWN" or "SIDEWAYS",
    "m15_signal": "エントリートリガーの説明",
    "alignment": true or false,
    "smc_liquidity_sweep": true or false,
    "smc_sweep_direction": "HIGH" or "LOW" or "NONE",
    "smc_ob_confirmed": true or false,
    "smc_fvg_present": true or false,
    "reasoning": "判断理由の詳細",
    "news_impact": "関連ニュースの要約",
    "sl_distance": SL幅の数値(price単位、Sweep起点外側を目安),
    "tp_distance": TP幅の数値(price単位、SL以上で利確優先)
}}

重要 (逆張りセットアップ): SMCフィルタ有効時は smc_liquidity_sweep が false なら decision を必ず "SKIP" にしてください。
重要 (順張りセットアップ): smc_ob_confirmed が false の場合は decision を必ず "SKIP" にしてください。
上位足と下位足のトレンドが一致しない場合も "SKIP" にしてください。
自信度が60未満の場合も "SKIP" にしてください。
必ずJSON形式のみで回答してください."""

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
                smc_data=smc_data,
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
            tp_distance=atr_m15 * 1.8,
            raw_response=str(e),
            smc_liquidity_sweep=False,
        )


# ── エグジット分析 ──────────────────────

def analyze_exit(symbol: str, direction: str, entry_price: float,
                 current_price: float, unrealized_pnl: float,
                 hold_minutes: int, m15_image: bytes,
                 entry_reasoning: str = "", entry_news_impact: str = "",
                 tp_price: float | None = None, current_sl: float | None = None) -> ExitSignal:
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
- TP価格: {tp_price if tp_price is not None else '未設定'}
- 現在SL価格: {current_sl if current_sl is not None else '未設定'}

【エントリー時の判断根拠】
- entry_reasoning: {entry_reasoning[:600] if entry_reasoning else '記録なし'}
- entry_news_impact: {entry_news_impact[:400] if entry_news_impact else '記録なし'}

15分足のチャート画像を添付しています。

【分析手順】
1. 現在のトレンドがポジション方向と一致しているか
2. 反転シグナルの有無
3. web検索で直近のニュースを確認
4. 利確/損切りの判断

【ポジション管理方針】
- 利確優先で判断し、利益保護できる局面では HOLD より EXIT を優先してください
- エントリー時の前提（方向性・構造・ニュース）が崩れたら、含み益/含み損を問わず EXIT を優先してください
- TP到達、TP目前で失速、または伸び代が乏しい場合は EXIT 寄りで判断してください
- EXITは「利確」「前提崩壊」「構造崩れ」「ニュースで前提破壊」のいずれかがあれば積極的に選んでください
- HOLDは「トレンド継続が明確」「前提維持」「直近で利益拡大余地が明確」の全てを満たす時に限定してください

【回答フォーマット (JSON)】
{{
    "decision": "HOLD" or "EXIT",
    "confidence": 0-100,
    "entry_premise_valid": true or false,
    "reasoning": "判断理由の詳細",
    "news_impact": "関連ニュースの要約"
}}

重要: entry_premise_valid が false の場合は decision を必ず EXIT にしてください。
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
            entry_premise_valid=True,
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
                              primary_signal: EntrySignal,
                              smc_data: dict | None = None) -> EntrySignal:
    # SMC価格レベルの再検証セクション
    if smc_data and config.SMC_FILTER_ENABLED:
        smc_verify_section = f"""【SMC一次判定の再検証】
一次モデルのSMC判定:
- smc_liquidity_sweep: {primary_signal.smc_liquidity_sweep}
- smc_sweep_direction: {primary_signal.smc_sweep_direction}
- smc_ob_confirmed: {primary_signal.smc_ob_confirmed}
- smc_fvg_present: {primary_signal.smc_fvg_present}

参照価格レベル:
- 前日高値(PDH): {smc_data.get('pdh', 'N/A')} / 前日安値(PDL): {smc_data.get('pdl', 'N/A')}
- 前週高値(PWH): {smc_data.get('pwh', 'N/A')} / 前週安値(PWL): {smc_data.get('pwl', 'N/A')}
- H1スウィング高値: {smc_data.get('swing_highs', [])} / 安値: {smc_data.get('swing_lows', [])}

チャートと価格レベルを照合し、「本当にLiquidity Sweepが起きているか」を厳しく再判定してください。
一次モデルが smc_liquidity_sweep=true と報告していても、チャートで確認できない場合は SKIP にしてください。
"""
    else:
        smc_verify_section = ""

    prompt = f"""あなたは最終承認を担当するシニアSMCアナリストです。
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
- tp_distance: {primary_signal.tp_distance}

【現在の市場データ】
- 現在価格: {current_price}
- M15 ATR(14): {atr_m15:.5f}
- H1 ATR(14): {atr_h1:.5f}
- 口座残高: ¥{balance:,.0f}

{smc_verify_section}
【あなたの役割】
SMCの観点から一次判定を独立して再評価してください。特に:
  1. Liquidity Sweepは本物か（単なる高安値タッチではないか）
  2. Sweep後のBOS (Break of Structure) は明確か
  3. Order BlockまたはFVGへの適切な戻り（引きつけ）があるか
  4. SLはSweep起点の外側に配置されていて論理的か

【最終承認ポリシー】
- SMC3条件 (Sweep+反転+BOS) が揃わない場合は SKIP
- 損切り貧乏を避けるため、少しでも期待値が怪しければ SKIP
- 「当たるか」より「SweepとBOSの構造で利大損小が成立するか」を優先

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
    "smc_liquidity_sweep": true or false,
    "smc_sweep_direction": "HIGH" or "LOW" or "NONE",
    "smc_ob_confirmed": true or false,
    "smc_fvg_present": true or false,
    "reasoning": "最終承認の理由（SMC観点を含む）",
    "news_impact": "ニュースとイベントの評価",
    "sl_distance": 数値,
    "tp_distance": 数値
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
            tp_distance=primary_signal.tp_distance,
            raw_response=primary_signal.raw_response + "\n\n--- FINAL APPROVAL ERROR ---\n\n" + str(e),
            smc_liquidity_sweep=False,
            approved_by_final_model=False,
            final_model_name=config.OPENAI_FINAL_APPROVAL_MODEL,
        )


# ── レスポンスパーサ ────────────────────

def _parse_bool(value, default: bool = False) -> bool:
    """LLMが返すboolを厳密に解釈する。文字列"false"をTrue扱いしない。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", "", "none", "null"}:
            return False
    return default

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
            tp_distance=fallback_atr * 1.8,
            raw_response=raw_text,
            smc_liquidity_sweep=False,
        )

    sl_distance = float(data.get("sl_distance", fallback_atr * 1.5))
    if sl_distance <= 0:
        sl_distance = fallback_atr * 1.5

    tp_distance = float(data.get("tp_distance", sl_distance * config.ENTRY_TP_R))
    min_tp_distance = sl_distance * config.ENTRY_MIN_TP_R
    if tp_distance < min_tp_distance:
        tp_distance = min_tp_distance

    return EntrySignal(
        decision=data.get("decision", "SKIP").upper(),
        confidence=int(data.get("confidence", 0)),
        h1_trend=data.get("h1_trend", "UNKNOWN").upper(),
        m15_signal=data.get("m15_signal", ""),
        alignment=_parse_bool(data.get("alignment", False), default=False),
        reasoning=data.get("reasoning", ""),
        news_impact=data.get("news_impact", ""),
        sl_distance=sl_distance,
        tp_distance=tp_distance,
        raw_response=raw_text,
        smc_liquidity_sweep=_parse_bool(data.get("smc_liquidity_sweep", False), default=False),
        smc_sweep_direction=str(data.get("smc_sweep_direction", "NONE")).upper(),
        smc_ob_confirmed=_parse_bool(data.get("smc_ob_confirmed", False), default=False),
        smc_fvg_present=_parse_bool(data.get("smc_fvg_present", False), default=False),
    )


def _parse_exit_response(raw_text: str) -> ExitSignal:
    """AIレスポンスからJSONを抽出してExitSignalに変換"""
    data = _extract_json(raw_text)
    if data is None:
        logger.warning("ExitレスポンスのJSON解析失敗 → HOLD")
        return ExitSignal(
            decision="HOLD", confidence=0,
            entry_premise_valid=True,
            reasoning=raw_text[:300], news_impact="",
            raw_response=raw_text,
        )

    return ExitSignal(
        decision=data.get("decision", "HOLD").upper(),
        confidence=int(data.get("confidence", 0)),
        entry_premise_valid=_parse_bool(data.get("entry_premise_valid", True), default=True),
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
