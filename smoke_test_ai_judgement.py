import json
from datetime import datetime

import config
import mt5_connector
import chart_capture
import ai_analyzer


def main():
    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "FAILED",
        "symbol": "GOLD",
    }

    if not mt5_connector.initialize():
        result["error"] = "MT5 initialize/login failed"
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    try:
        symbol = "GOLD"
        info = mt5_connector.get_account_info()
        if info is None:
            result["error"] = "account_info unavailable"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        h1_img, m15_img = chart_capture.generate_chart_pair(symbol)
        if h1_img is None or m15_img is None:
            result["error"] = "chart generation failed"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        rates_h1 = mt5_connector.get_rates(symbol, config.TREND_TF, 160)
        rates_m15 = mt5_connector.get_rates(symbol, config.EXECUTION_TF, 160)
        if rates_h1 is None or rates_m15 is None:
            result["error"] = "rate fetch failed"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        atr_h1 = mt5_connector.calculate_atr(rates_h1, config.ATR_PERIOD)
        atr_m15 = mt5_connector.calculate_atr(rates_m15, config.ATR_PERIOD)

        tick = mt5_connector.get_current_price(symbol)
        if tick is None:
            result["error"] = "tick fetch failed"
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return

        signal = ai_analyzer.analyze_entry(
            symbol=symbol,
            current_price=tick["bid"],
            atr_h1=atr_h1,
            atr_m15=atr_m15,
            h1_image=h1_img,
            m15_image=m15_img,
            balance=info["balance"],
        )

        result.update(
            {
                "status": "OK",
                "decision": signal.decision,
                "confidence": signal.confidence,
                "alignment": signal.alignment,
                "h1_trend": signal.h1_trend,
                "m15_signal": signal.m15_signal,
                "reasoning": signal.reasoning,
                "news_impact": signal.news_impact,
                "sl_distance": signal.sl_distance,
                "raw_response_preview": signal.raw_response[:500],
            }
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        mt5_connector.shutdown()


if __name__ == "__main__":
    main()
