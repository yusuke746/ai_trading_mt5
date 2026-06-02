"""OpenAI Responses API 連携モジュール

GPT-5系モデルを使用して:
    1. H1 + M15 チャート画像からエントリー判断
    2. 保有ポジションのエグジット判断

ニュース評価は別モジュール(news_monitor.py)のバックグラウンド処理で実施する。
"""

import json
import logging
from dataclasses import dataclass

from openai import OpenAI

import config
import chart_capture

logger = logging.getLogger(__name__)

_client: OpenAI | None = None
_openrouter_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client


def _get_openrouter_client() -> OpenAI:
    """OpenRouter用クライアント (openaiライブラリ互換、base_urlだけ差し替え)"""
    global _openrouter_client
    if _openrouter_client is None:
        if not config.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY が .env に設定されていません")
        _openrouter_client = OpenAI(
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _openrouter_client


# ── レスポンス構造体 ────────────────────

@dataclass
class EntrySignal:
    decision: str                    # "BUY", "SELL", "SKIP"
    confidence: int                  # 0-100
    m15_signal_visual_check: str     # 視覚的確認の説明
    reasoning: str
    raw_response: str                # AI生テキスト
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

    _mech_entry_type = mech_gate.get("entry_type", "REVERSAL_SWEEP") if mech_gate else "REVERSAL_SWEEP"
    _mech_sweep_type = str(mech_gate.get("sweep_type", "NONE")).upper() if mech_gate else "NONE"
    _expected_dir = "BUY" if _mech_sweep_type == "LOW" else ("SELL" if _mech_sweep_type == "HIGH" else "BUY or SELL")

    if _mech_entry_type == "CONTINUATION_BOS":
        prompt = f"""[{symbol} / CONTINUATION_BOS / {_expected_dir}]
あなたはチャート画像の「図形とローソク足のパターン」を視覚的に確認するアシスタントです。
複雑な数値計算や相場予測は不要です。画像に描画されている図形（ゾーン）とローソク足の位置関係・形状だけを確認し、以下の条件を満たしているか判定してください。

【前提条件 (Python判定済み事実)】
・エントリータイプ: CONTINUATION_BOS (トレンドフォロー/BOS後の押し目・戻し)
・判定方向: {_expected_dir}
※BUYの場合のみBUYを、SELLの場合のみSELLを判定し、逆の場合はSKIPとしてください。

【視覚的チェック項目】
① トレンド確認: H1チャートのMA（白い曲線）が{_expected_dir}方向に傾斜し、ローソク足の流れが一方向のトレンドを示しているか？
② OB/FVGへの押し目: 現在のローソク足が、チャートに描画された色付きゾーン（緑＝Bull OB、青＝FVG）に到達または接触しているか？（BUYの場合）または（赤＝Bear OB）に接触しているか？（SELLの場合）
③ 空間的ゆとり: {_expected_dir}方向のすぐ先に、逆方向の分厚いゾーン（障害物）が立ち塞がっていないか？視覚的に価格が伸びるスペースがあるか？

【SKIP基準】
・MAが横ばいでトレンドが視覚的に不明瞭。
・現在価格がOB/FVGゾーンから明らかに離れている（タッチなし）。
・{_expected_dir}方向にすぐ大きな障害ゾーンがある。

【回答フォーマット (JSONのみ・コメント不要)】
{{
  "decision": "BUY" または "SELL" または "SKIP",
  "confidence": 0-100,
  "m15_signal_visual_check": "チャート上で視覚的に確認できたMAの傾き、OB/FVGゾーンとの位置関係、価格の伸び代を簡潔に説明",
  "reasoning": "最終判断の理由"
}}
必ずJSON形式のみで回答してください。"""
    else:
        # REVERSAL_SWEEP
        _sweep_dir = _mech_sweep_type  # "HIGH" or "LOW"
        prompt = f"""[{symbol} / REVERSAL_SWEEP / {_sweep_dir}]
あなたはチャート画像の「図形とローソク足のパターン」を視覚的に確認するアシスタントです。
複雑な数値計算や相場予測は不要です。画像に描画されている図形（ゾーン）とローソク足の位置関係・形状だけを確認し、以下の条件を満たしているか判定してください。

【前提条件 (Python判定済み事実)】
・エントリータイプ: REVERSAL_SWEEP (Liquidity Sweep後の反転)
・判定方向: {_sweep_dir}
※LOW sweepの場合はBUYのみ、HIGH sweepの場合はSELLのみを検討し、逆の場合はSKIPとしてください。

【視覚的チェック項目】
① ヒゲの反発（Sweep）: ローソク足が、チャートの下部（または上部）にある主要なラインやゾーン（色のついた帯）を一度突き抜けた後、長いヒゲ（ピンバー）を残して内側に戻って確定しているか？
② プライスアクション: ヒゲを付けた後、反転方向への強い動き（包み足、または反発を示す明確な大陽線/大陰線）が確認できるか？
③ 空間的ゆとり: エントリー方向のすぐ目の前に、逆方向の分厚いゾーン（障害物）が立ち塞がっていないか？（視覚的に価格が伸びるスペースがあるか）

【SKIP基準】
・ヒゲが短すぎる、または実体でゾーンを完全に抜けてしまっている（ブレイクアウトになっている）。
・反発の勢いが弱く、セットアップの形が視覚的に美しくない・不明確である。

【回答フォーマット (JSONのみ・コメント不要)】
{{
  "decision": "BUY" または "SELL" または "SKIP",
  "confidence": 0-100,
  "m15_signal_visual_check": "チャート上で視覚的に確認できたヒゲの長さやローソク足の形状、描画ゾーンとの位置関係を簡潔に説明",
  "reasoning": "最終判断の理由"
}}
必ずJSON形式のみで回答してください。"""

    try:
        if config.USE_OPENROUTER_FOR_ENTRY:
            raw_text = _analyze_entry_via_openrouter(
                symbol=symbol, prompt=prompt,
                h1_b64=h1_b64, m15_b64=m15_b64,
            )
        else:
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
            )
            raw_text = response.output_text

        logger.info("[AI Entry] %s raw response length: %d", symbol, len(raw_text))

        primary_signal = _apply_entry_signal_guards(
            _parse_entry_response(raw_text),
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
            m15_signal_visual_check="API Error",
            reasoning=str(e),
            raw_response=str(e),
        )


def _analyze_entry_via_openrouter(symbol: str, prompt: str, h1_b64: str, m15_b64: str) -> str:
    """OpenRouter (Qwen-VL等) を使ったエントリー分析。
    Chat Completions API (vision対応) を使用。
    ※ web_search_preview は非対応のためnews_impactは 'N/A' になる。
    """
    model_name = config.OPENROUTER_ENTRY_MODEL
    logger.info("[AI Entry OpenRouter] %s model=%s", symbol, model_name)

    # web検索がないことをプロンプトに付記
    or_prompt = (
        prompt
        + "\n\n※ web検索ツールは使用不可。news_impact は \"N/A (web検索非対応)\" と記入してください。"
    )

    client = _get_openrouter_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": or_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{h1_b64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{m15_b64}"},
                    },
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


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

    # ── Python側でTP到達率・フェーズを計算 ──
    if tp_price is not None and entry_price is not None and tp_price != entry_price:
        _total_dist = abs(tp_price - entry_price)
        _current_dist = abs(current_price - entry_price)
        tp_progress_pct = min(100.0, (_current_dist / _total_dist) * 100.0)
    else:
        tp_progress_pct = 0.0

    _NEAR_TP_THRESHOLD = 75.0   # TP到達率75%以上 → NEAR_TPフェーズ
    _LONG_HOLD_THRESHOLD_MIN = 250

    if tp_progress_pct >= _NEAR_TP_THRESHOLD:
        phase = "NEAR_TP"
    elif safe_hold_minutes >= _LONG_HOLD_THRESHOLD_MIN:
        phase = "LONG_HOLD"
    else:
        phase = "NORMAL"

    inv_text = f"{invalidation_price:.5f}" if invalidation_price is not None else "N/A"
    tp_text  = f"{tp_price}"               if tp_price is not None             else "N/A"
    sl_text  = f"{current_sl}"             if current_sl is not None           else "N/A"

    # ── フェーズ別タスクセクション ──
    if phase == "NEAR_TP":
        task_section = f"""【task: NEAR_TP】
TP到達率が {tp_progress_pct:.0f}% です。チャートの緑点線（TP={tp_text}）付近での「反転シグナル」を視覚的に確認してください。

① TP付近のゾーン・ラインで長ヒゲ・ピンバーが出て跳ね返されているか？
② 包み足または明確な大ローソクが逆方向に出ているか？

【HOLDデフォルト】反転シグナルが弱い・不明確 → HOLD
【EXIT】TP付近ゾーンでの明確な反転パターンが視覚的に確認できる → EXIT

【output JSON only】
{{
    "decision": "HOLD" or "EXIT",
    "confidence": 0-100,
    "entry_premise_valid": true,
    "invalidation_breached": false,
    "reasoning": "TP付近の視覚パターンを1文で説明",
    "news_impact": "N/A"
}}"""

    elif phase == "LONG_HOLD":
        task_section = f"""【task: LONG_HOLD】
保有時間が {safe_hold_minutes} 分経過、TP到達率は {tp_progress_pct:.0f}% です。
チャートのMA（白い曲線）の向きとローソク足の流れを視覚的に確認してください。

① MA（白い曲線）: 今もエントリー方向（{direction}）に傾いているか？逆方向に曲がっていないか？
② 直近ローソク足の流れ: TP方向への動きが継続しているか、それとも横ばい・押し戻しを繰り返しているか？

【HOLD → entry_premise_valid=true】MAが方向維持、かつローソク足がTP方向へ継続している場合のみ
【EXIT → entry_premise_valid=false】以下のいずれかに該当する場合（両方ともfalse）:
  ・MAが明確に逆転している（エントリー方向と逆向きに傾いている）
  ・TP到達率が{tp_progress_pct:.0f}%以下かつ長時間停滞（TP方向に動く勢いが見えない）
※EXITを返す場合は必ず entry_premise_valid を false にしてください。

【output JSON only】
{{
    "decision": "HOLD" or "EXIT",
    "confidence": 0-100,
    "entry_premise_valid": true（HOLD時）or false（EXIT時は必ずfalse）,
    "invalidation_breached": false,
    "reasoning": "MAの向きとローソク足の状態を1文で説明",
    "news_impact": "N/A"
}}"""

    else:  # NORMAL
        task_section = f"""【task: NORMAL】
TP到達率は {tp_progress_pct:.0f}% です（まだTP付近ではありません）。
現在価格付近のローソク足とMAの状態を視覚的に確認してください。

【HOLD】現在価格付近に明確な反転シグナルがない → HOLDデフォルト
【EXIT】以下の両方が視覚的に確認できる場合のみEXIT:
  ・現在価格付近でエントリー方向（{direction}）と逆の強い大ローソク足（包み足・急反転）が出ている
  ・かつ MA（白い曲線）がエントリー方向と逆向きに明確に転換している

不明確・片方だけ → 必ずHOLD

【output JSON only】
{{
    "decision": "HOLD" or "EXIT",
    "confidence": 0-100,
    "entry_premise_valid": true,
    "invalidation_breached": false,
    "reasoning": "現在価格付近のパターンとMAの状態を1文で説明",
    "news_impact": "N/A"
}}"""

    prompt = f"""[{symbol} / EXIT / {direction}]
あなたはチャート画像の「図形とローソク足のパターン」を視覚的に確認するアシスタントです。
複雑な数値計算・相場予測は不要です。システムの機械判定の補助として、視覚パターンのみを確認してください。

【重要】
- 無効化ライン（赤い太線）の終値ブレイク判定はシステム側で実施済みです。
- invalidation_breached は必ず false を返してください。
- HOLDに倒すことを優先してください。明確なシグナルのみEXITを返してください。

【ポジション情報 (Python計算済み)】
- symbol: {symbol} / direction: {direction}
- entry: {entry_price} / current: {current_price} / pnl_jpy: {unrealized_pnl:,.0f}
- hold_min: {safe_hold_minutes} / phase: {phase}
- tp: {tp_text} (チャート上の緑点線) / sl: {sl_text} / inv: {inv_text}
- TP到達率: {tp_progress_pct:.0f}%

{task_section}"""

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
    # guards通過済みのシグナルはalignmentによるブロック不要 (REVERSAL_SWEEPはalignment=falseが自然)
    # KIWAMI口座は"GOLD#"等の#付きシンボルを使うため正規化して比較する
    _sym_norm = str(symbol).rstrip("#.").upper()
    _in_list = symbol in config.FINAL_APPROVAL_SYMBOLS or _sym_norm in {
        s.rstrip("#.").upper() for s in config.FINAL_APPROVAL_SYMBOLS
    }
    return _in_list or signal.confidence >= config.FINAL_APPROVAL_MIN_CONFIDENCE


def _apply_entry_signal_guards(signal: EntrySignal, mech_gate: dict | None = None) -> EntrySignal:
    """Sweep方向矛盾のみSKIPに強制矯正する（それ以外はAIの判断を尊重）。"""
    if signal.decision not in {"BUY", "SELL"}:
        return signal

    entry_type = mech_gate.get("entry_type") if mech_gate else None
    mech_sweep_type = str(mech_gate.get("sweep_type", "NONE")).upper() if mech_gate else "NONE"

    # Sweep方向とエントリー方向の矛盾 (HIGH sweep→SELL / LOW sweep→BUY のみ有効)
    if entry_type == "REVERSAL_SWEEP" and mech_sweep_type in {"HIGH", "LOW"}:
        expected_dir = "SELL" if mech_sweep_type == "HIGH" else "BUY"
        if signal.decision != expected_dir:
            guard_note = f"Entry guard forced SKIP: reversal_direction_wrong(sweep={mech_sweep_type},expected={expected_dir},got={signal.decision})"
            signal.decision = "SKIP"
            signal.reasoning = (f"{guard_note}. {signal.reasoning}").strip(". ")

    return signal


def _run_entry_final_approval(symbol: str, current_price: float,
                              atr_h1: float, atr_m15: float,
                              balance: float, h1_b64: str, m15_b64: str,
                              primary_signal: EntrySignal,
                              smc_data: dict | None = None) -> EntrySignal:
    prompt = f"""あなたは最終承認を担当するシニアビジュアルアナリストです。
一次判定モデルが {symbol} の {primary_signal.decision} を提案しています。
チャート画像を独立して再評価し、承認可否を判断してください。

【一次判定サマリー】
- decision: {primary_signal.decision} / confidence: {primary_signal.confidence}
- visual_check: {primary_signal.m15_signal_visual_check[:200]}
- reasoning: {primary_signal.reasoning[:300]}
- 現在価格: {current_price} / M15 ATR: {atr_m15:.5f} / H1 ATR: {atr_h1:.5f}

【承認チェック (すべてYESでのみ承認)】
  ① Sweep/反転パターンがチャート上で本物か (ヒゲタッチだけでなく明確な侵食か)
  ② 反転方向への明確なプライスアクション（包み足/大ローソク）が確認できるか
  ③ エントリー方向に価格が伸びるスペース（空間的ゆとり）があるか
  → 1つでも疑わしければ SKIP

【JSONのみで回答】
{{
    "decision": "{primary_signal.decision}" or "SKIP",
    "confidence": 0-100,
    "m15_signal_visual_check": "視覚的確認の説明",
    "reasoning": "承認/否決の理由"
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
            reasoning={"effort": config.OPENAI_FINAL_APPROVAL_REASONING_EFFORT},
        )

        raw_text = response.output_text
        final_signal = _parse_entry_response(raw_text)
        final_signal.raw_response = primary_signal.raw_response + "\n\n--- FINAL APPROVAL ---\n\n" + raw_text
        final_signal.approved_by_final_model = final_signal.decision in {"BUY", "SELL"}
        final_signal.final_model_name = config.OPENAI_FINAL_APPROVAL_MODEL
        return final_signal
    except Exception as e:
        logger.error("[AI Entry Final] API呼び出しエラー: %s", e)
        return EntrySignal(
            decision="SKIP",
            confidence=0,
            m15_signal_visual_check=primary_signal.m15_signal_visual_check,
            reasoning=f"Final approval failed: {e}",
            raw_response=primary_signal.raw_response + "\n\n--- FINAL APPROVAL ERROR ---\n\n" + str(e),
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

def _parse_entry_response(raw_text: str) -> EntrySignal:
    """AIレスポンスからJSONを抽出してEntrySignalに変換"""
    data = _extract_json(raw_text)
    if data is None:
        logger.warning("EntryレスポンスのJSON解析失敗 → SKIP")
        return EntrySignal(
            decision="SKIP", confidence=0,
            m15_signal_visual_check="JSON parse error",
            reasoning=raw_text[:300],
            raw_response=raw_text,
        )

    return EntrySignal(
        decision=data.get("decision", "SKIP").upper(),
        confidence=int(data.get("confidence", 0)),
        m15_signal_visual_check=data.get("m15_signal_visual_check", ""),
        reasoning=data.get("reasoning", ""),
        raw_response=raw_text,
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
