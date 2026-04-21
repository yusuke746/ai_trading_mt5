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
    invalidation_price: float | None = None  # SMC構造が完全に崩壊する無効化ライン


@dataclass
class ExitSignal:
    decision: str       # "HOLD", "EXIT"
    confidence: int
    entry_premise_valid: bool
    reasoning: str
    news_impact: str
    raw_response: str
    invalidation_breached: bool = False  # 無効化ラインを終値でブレイクしたか


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
【機械判定ゲート (Python計算済み・確定事実)】
- エントリータイプ: {_mech_entry_type} ({_entry_type_label})
- Sweep検出: {mech_gate.get('sweep_pass', 'N/A')} (方向: {mech_gate.get('sweep_type', 'N/A')})
- BOS/MAトレンド確認: {mech_gate.get('bos_pass', 'N/A')}
- RR充足 (最低{config.ENTRY_MIN_TP_R:.1f}R以上): {mech_gate.get('rr_pass', 'N/A')}
"""
    else:
        mech_section = ""

    # ── セットアップ別の必須条件 (SMCフィルタ有効時) ──
    if config.SMC_FILTER_ENABLED:
        if _mech_entry_type == "CONTINUATION_BOS":
            setup_condition = """
【セットアップ条件 — 順張り (BOS後の押し目/戻し)】
チャートのBOS/OB/FVGゾーンを確認し、以下をすべて満たす場合のみエントリー:
  ① H1とM15のオーダーフロー方向が一致している
  ② 明確なBOSが確認でき、トレンド方向が確立されている
  ③ 押し目/戻しがOBまたはFVGゾーンに到達し、反転シグナルがある
  → smc_ob_confirmed=true が必須
"""
        else:
            setup_condition = f"""
【セットアップ条件 — 逆張り (Liquidity Sweep後の反転)】
チャートのLiquidityライン・OB・FVGゾーンを確認し、以下をすべて満たす場合のみエントリー:
  ① H1とM15のオーダーフロー方向が一致している
  ② Liquidityラインを ATR({atr_h1:.5f})×{config.SMC_SWEEP_ATR_MULT}以上 侵食後に反転したSweepが確認できる
  ③ Sweep後にBOS/長ヒゲ/Engulfingの反転アクションがある
  → smc_liquidity_sweep=true が必須
"""
    else:
        setup_condition = ""

    prompt = f"""あなたはSMCアナリストです。
H1・M15チャート画像（BOS/CHoCH/OB/FVG/Liquidity/PDH/PDL/PWH/PWL描画済み）を見て
{symbol}のエントリー判断をしてください。
{mech_section}
【市場データ】
- 銘柄: {symbol} / 現在価格: {current_price}
- M15 ATR: {atr_m15:.5f} / H1 ATR: {atr_h1:.5f}
- 口座残高: ¥{balance:,.0f}
{setup_condition}
【SKIP条件 (1つでも該当したらSKIP)】
- confidence < 70
- h1_trend = SIDEWAYS
- H1とM15のトレンドが不一致 (alignment=false)
- 機械ゲート rr_pass=False かつ伸び代を具体的に説明できない
- 逆張り: smc_liquidity_sweep=false
- 順張り: smc_ob_confirmed=false
- web検索で重大なニュース/指標リスクが確認できた場合

【回答フォーマット (JSONのみ)】
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
    "reasoning": "判断理由（チャートで確認した構造を簡潔に）",
    "news_impact": "web検索で確認した直近ニュースの要約",
    "sl_distance": SL幅の数値(price単位、Sweep起点または直近スウィング外側),
    "tp_distance": TP幅の数値(price単位、最低{config.ENTRY_MIN_TP_R:.1f}R以上),
    "invalidation_price": このエントリーのSMC構造が完全に崩壊する具体的な価格(数値)
}}

必ずJSON形式のみで回答してください。"""

    try:
        client = _get_client()
        response = client.responses.create(
            model=config.OPENAI_ENTRY_MODEL,
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

        primary_signal = _apply_entry_signal_guards(
            _parse_entry_response(raw_text, atr_m15),
            mech_gate=mech_gate,
        )

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
                 tp_price: float | None = None, current_sl: float | None = None,
                 invalidation_price: float | None = None) -> ExitSignal:
    """保有ポジションのエグジット判断"""

    m15_b64 = chart_capture.chart_to_base64(m15_image)
    safe_hold_minutes = max(0, hold_minutes)

    # 無効化ライン情報のセクション
    if invalidation_price is not None:
        invalidation_section = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━
【最優先確認事項: 無効化ライン (赤線)】
━━━━━━━━━━━━━━━━━━━━━━━━━━━
チャート画像に【赤い太線】で無効化ラインが引かれています。
無効化ライン価格: {invalidation_price}

「最新の確定済みローソク足の実体（ヒゲは除外、終値）」がこの赤線を明確にブレイクしているかどうかを
冷徹に事実確認してください。

- ブレイク確定 (終値が赤線の反対側にある): entry_premise_valid = false → decision = "EXIT"
- ブレイク未確認 (ヒゲだけが触れた、または終値はまだ赤線の味方側にある): 下記の追加評価へ進む

重要: 「惜しい」「あと少し」「ほぼブレイク」は「未確認」として扱ってください。
終値ベースの事実のみで判定してください。
"""
    else:
        invalidation_section = ""

    prompt = f"""あなたはSMCトレードの監視AIです。保有ポジションを冷静・客観的に評価し、
「構造的に決済すべき根拠があるか」だけを判定してください。感情的な判断は不要です。

【ポジション情報】
- 銘柄: {symbol}
- 方向: {direction}
- エントリー価格: {entry_price}
- 現在価格: {current_price}
- 含み損益: ¥{unrealized_pnl:,.0f}
- 保有時間: {safe_hold_minutes}分
- TP価格: {tp_price if tp_price is not None else '未設定'}
- 現在SL価格: {current_sl if current_sl is not None else '未設定'}

【エントリー時の判断根拠】
{entry_reasoning[:600] if entry_reasoning else '記録なし'}

【エントリー時のニュース判断】
{entry_news_impact[:400] if entry_news_impact else '記録なし'}

15分足のチャート画像を添付しています。{invalidation_section}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
【評価手順】
━━━━━━━━━━━━━━━━━━━━━━━━━━━

■ ステップ1: 無効化ライン確認 (赤線がある場合は最優先)
  - 最新確定足の「終値」が赤線をブレイクしているか？ (YESなら即EXIT)

■ ステップ2: TP・利益状況の確認
  - TP（{tp_price if tp_price is not None else '未設定'}）に到達または目前か？
  - 目前で明確な反転シグナル（長ヒゲ・Engulfinなど）が出ているか？

■ ステップ3: オーダーフロー継続確認
  - 直近3〜5本の実体の方向がエントリー方向と一致しているか？
  - 直近スウィング高値（BUYなら）または安値（SELLなら）を更新できているか？

■ ステップ4: ニュース確認
  - web検索で{symbol}の直近ニュースを確認し、エントリー前提を覆す材料がないか確認する

━━━━━━━━━━━━━━━━━━━━━━━━━━━
【判断基準】
━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXIT条件 (1つ以上が明確):
  ① 無効化ライン (赤線) を終値でブレイク確定
  ② TP到達 or TP目前で反転シグナル確認
  ③ オーダーフローが明確に逆転 (安値切り下げ継続 / 高値切り下げ確立)
  ④ 重要ファンダメンタル前提が崩れた

HOLD条件 (以下をすべて満たす):
  ① 無効化ラインを終値でブレイクしていない
  ② TP到達余地あり、またはオーダーフロー継続中
  ③ SLを侵食していない (含み損はノイズ範囲)

【回答フォーマット (JSON)】
{{
    "decision": "HOLD" or "EXIT",
    "confidence": 0-100,
    "entry_premise_valid": true or false,
    "invalidation_breached": true or false,
    "reasoning": "判断理由（赤線ブレイクの有無、終値ベースの事実を含む）",
    "news_impact": "web検索で確認した直近ニュースの要約"
}}

重要: invalidation_breached が true の場合は entry_premise_valid を false、decision を必ず EXIT にしてください。
含み損のみ・保有時間短い・SL未到達、これだけでは EXIT にしないでください。
必ずJSON形式のみで回答してください。"""

    try:
        client = _get_client()
        response = client.responses.create(
            model=config.OPENAI_EXIT_MODEL,
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


def _apply_entry_signal_guards(signal: EntrySignal, mech_gate: dict | None = None) -> EntrySignal:
    """LLMの矛盾したエントリー判断をSKIPに矯正する。"""
    if signal.decision not in {"BUY", "SELL"}:
        return signal

    skip_reasons: list[str] = []
    entry_type = mech_gate.get("entry_type") if mech_gate else None

    if signal.confidence < 70:
        skip_reasons.append("confidence_below_70")
    if signal.h1_trend == "SIDEWAYS":
        skip_reasons.append("h1_sideways")
    if not signal.alignment:
        skip_reasons.append("trend_misalignment")
    if mech_gate and not mech_gate.get("rr_pass", False):
        skip_reasons.append("mechanical_rr_failed")
    if mech_gate and not mech_gate.get("bos_pass", False):
        skip_reasons.append("mechanical_bos_failed")

    mech_sweep_type = str(mech_gate.get("sweep_type", "NONE")).upper() if mech_gate else "NONE"

    if entry_type == "REVERSAL_SWEEP" and not signal.smc_liquidity_sweep:
        skip_reasons.append("reversal_without_sweep")
    if entry_type == "REVERSAL_SWEEP" and mech_sweep_type in {"HIGH", "LOW"} and signal.smc_sweep_direction != mech_sweep_type:
        skip_reasons.append("sweep_direction_mismatch")
    if entry_type == "CONTINUATION_BOS" and not signal.smc_ob_confirmed:
        skip_reasons.append("continuation_without_ob")

    if not skip_reasons:
        return signal

    guard_note = "Entry guard forced SKIP: " + ", ".join(skip_reasons)
    reasoning = signal.reasoning.strip()
    if reasoning:
        reasoning = f"{guard_note}. {reasoning}"
    else:
        reasoning = guard_note

    signal.decision = "SKIP"
    signal.reasoning = reasoning
    return signal


def _run_entry_final_approval(symbol: str, current_price: float,
                              atr_h1: float, atr_m15: float,
                              balance: float, h1_b64: str, m15_b64: str,
                              primary_signal: EntrySignal,
                              smc_data: dict | None = None) -> EntrySignal:
    prompt = f"""あなたは最終承認を担当するシニアSMCアナリストです。
一次判定モデルが {symbol} の {primary_signal.decision} を提案しています。
チャート画像（BOS/CHoCH/OB/FVG/Liquidity描画済み）を独立して再評価し、承認可否を判断してください。

【一次判定サマリー】
- decision: {primary_signal.decision} / confidence: {primary_signal.confidence}
- h1_trend: {primary_signal.h1_trend} / alignment: {primary_signal.alignment}
- smc_liquidity_sweep: {primary_signal.smc_liquidity_sweep} ({primary_signal.smc_sweep_direction})
- smc_ob_confirmed: {primary_signal.smc_ob_confirmed} / smc_fvg_present: {primary_signal.smc_fvg_present}
- reasoning: {primary_signal.reasoning[:300]}
- sl_distance: {primary_signal.sl_distance} / tp_distance: {primary_signal.tp_distance}
- 現在価格: {current_price} / M15 ATR: {atr_m15:.5f} / H1 ATR: {atr_h1:.5f}

【承認チェック (すべてYESでのみ承認)】
  ① チャート上でLiquidity Sweepが本物か (ヒゲタッチだけでなく明確な侵食か)
  ② Sweep後のBOS/反転アクションが明確か
  ③ SLがSweep起点の外側に置けてリスクが限定されているか
  → 1つでも疑わしければ SKIP

【JSONのみで回答】
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
    "reasoning": "承認/否決の理由",
    "news_impact": "ニュース評価",
    "sl_distance": 数値,
    "tp_distance": 数値,
    "invalidation_price": SMC構造崩壊の無効化ライン価格(数値)
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
        invalidation_price=float(data["invalidation_price"]) if data.get("invalidation_price") is not None else None,
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

    invalidation_breached = _parse_bool(data.get("invalidation_breached", False), default=False)
    entry_premise_valid = _parse_bool(data.get("entry_premise_valid", True), default=True)
    # invalidation_breached=true の場合は必ず前提崩壊扱いに矯正
    if invalidation_breached:
        entry_premise_valid = False

    return ExitSignal(
        decision=data.get("decision", "HOLD").upper(),
        confidence=int(data.get("confidence", 0)),
        entry_premise_valid=entry_premise_valid,
        reasoning=data.get("reasoning", ""),
        news_impact=data.get("news_impact", ""),
        raw_response=raw_text,
        invalidation_breached=invalidation_breached,
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

    # テキスト中の最初のJSONオブジェクトを、波括弧バランスで抽出して試行
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue

            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

        start = text.find("{", start + 1)

    return None
