"""
Scanner de Señales en Vivo — Trading Bot
Corre diariamente via GitHub Actions y manda alertas a Telegram.
"""

import json
import os
import warnings
from datetime import datetime

import pandas as pd
import pandas_ta as ta
import requests
import yfinance as yf

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── Telegram ──────────────────────────────────────────────────────────────────

def send_telegram(token: str, chat_id: str, message: str):
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"  ⚠️  Telegram error: {e}")


# ── Data & Indicadores ────────────────────────────────────────────────────────

def fetch_data(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="120d", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"]  = ta.ema(df["Close"], length=p["ema_fast"])
    df["ema_slow"]  = ta.ema(df["Close"], length=p["ema_slow"])
    df["rsi"]       = ta.rsi(df["Close"], length=p["rsi_period"])
    macd            = ta.macd(df["Close"], fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    df["macd"]      = macd[f"MACD_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]
    df["macd_sig"]  = macd[f"MACDs_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]
    df["macd_hist"] = macd[f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]
    df["vol_ma"]    = df["Volume"].rolling(20).mean()
    df["vol_spike"] = df["Volume"] > df["vol_ma"] * p["volume_multiplier"]
    return df.dropna()


# ── Análisis de señales ───────────────────────────────────────────────────────

def analyze_ticker(ticker: str, df: pd.DataFrame, p: dict) -> dict:
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = float(last["Close"])

    signals = []
    score   = 0  # positivo = bullish, negativo = bearish

    # EMA Crossover
    ema_cross_up   = last["ema_fast"] > last["ema_slow"] and prev["ema_fast"] <= prev["ema_slow"]
    ema_cross_down = last["ema_fast"] < last["ema_slow"] and prev["ema_fast"] >= prev["ema_slow"]
    ema_bull       = last["ema_fast"] > last["ema_slow"]

    if ema_cross_up:
        signals.append("📈 EMA20 cruzó arriba EMA50 (Bullish crossover)")
        score += 2
    elif ema_cross_down:
        signals.append("📉 EMA20 cruzó abajo EMA50 (Bearish crossover)")
        score -= 2
    elif ema_bull:
        score += 1

    # RSI
    rsi = float(last["rsi"])
    if rsi < p["rsi_oversold"]:
        signals.append(f"🟢 RSI en zona de sobreventa ({rsi:.1f}) — posible rebote")
        score += 2
    elif rsi > p["rsi_overbought"]:
        signals.append(f"🔴 RSI en zona de sobrecompra ({rsi:.1f}) — posible reversión")
        score -= 2
    elif 40 <= rsi <= 60:
        score += 0  # neutral
    elif rsi > 60:
        score += 1

    # MACD
    macd_cross_up   = last["macd"] > last["macd_sig"] and prev["macd"] <= prev["macd_sig"]
    macd_cross_down = last["macd"] < last["macd_sig"] and prev["macd"] >= prev["macd_sig"]
    macd_bull       = last["macd"] > last["macd_sig"]
    macd_hist_grow  = last["macd_hist"] > prev["macd_hist"]

    if macd_cross_up:
        signals.append("📈 MACD cruzó arriba signal line (Bullish)")
        score += 2
    elif macd_cross_down:
        signals.append("📉 MACD cruzó abajo signal line (Bearish)")
        score -= 2
    elif macd_bull and macd_hist_grow:
        signals.append("📊 MACD histograma creciendo (momentum bullish)")
        score += 1

    # Volumen
    if bool(last["vol_spike"]):
        vol_ratio = float(last["Volume"]) / float(last["vol_ma"])
        direction = "alcista" if float(last["Close"]) > float(last["Open"]) else "bajista"
        signals.append(f"🔊 Spike de volumen {direction} ({vol_ratio:.1f}x promedio)")
        score += 1 if direction == "alcista" else -1

    # Confluencia (señal premium)
    confluence_bull = (
        ema_bull and macd_bull and
        float(last["rsi"]) < p["rsi_overbought"] and
        bool(last["vol_spike"])
    )
    confluence_bear = (
        not ema_bull and not macd_bull and
        float(last["rsi"]) > p["rsi_oversold"]
    )

    if confluence_bull:
        signals.append("⭐ CONFLUENCIA BULLISH: EMA + MACD + Volumen alineados")
        score += 2
    elif confluence_bear:
        signals.append("⭐ CONFLUENCIA BEARISH: EMA + MACD + RSI alineados")
        score -= 2

    # Cambio diario
    daily_chg = (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100

    # Veredicto
    if score >= 4:
        verdict = "🟢 COMPRA FUERTE"
    elif score >= 2:
        verdict = "🟡 SEÑAL DE COMPRA"
    elif score <= -4:
        verdict = "🔴 VENTA FUERTE"
    elif score <= -2:
        verdict = "🟠 SEÑAL DE VENTA"
    else:
        verdict = "⚪ NEUTRAL"

    return {
        "ticker":    ticker,
        "price":     price,
        "daily_chg": daily_chg,
        "rsi":       rsi,
        "ema_fast":  float(last["ema_fast"]),
        "ema_slow":  float(last["ema_slow"]),
        "macd":      float(last["macd"]),
        "macd_sig":  float(last["macd_sig"]),
        "vol_spike": bool(last["vol_spike"]),
        "signals":   signals,
        "score":     score,
        "verdict":   verdict,
    }


# ── Formato Telegram ──────────────────────────────────────────────────────────

def format_message(analysis: dict, date_str: str) -> str:
    a     = analysis
    emoji = "📈" if a["daily_chg"] >= 0 else "📉"
    lines = [
        f"<b>🤖 SEÑAL DIARIA — {date_str}</b>",
        f"<b>{a['ticker']}</b>  {emoji} ${a['price']:.2f} ({a['daily_chg']:+.2f}%)",
        "",
        f"<b>Indicadores:</b>",
        f"  RSI:  {a['rsi']:.1f}",
        f"  EMA20: {a['ema_fast']:.2f}  |  EMA50: {a['ema_slow']:.2f}",
        f"  MACD: {a['macd']:.3f}  |  Signal: {a['macd_sig']:.3f}",
        f"  Volumen alto: {'Sí ⚡' if a['vol_spike'] else 'No'}",
        "",
        f"<b>Señales detectadas:</b>",
    ]
    if a["signals"]:
        lines += [f"  {s}" for s in a["signals"]]
    else:
        lines.append("  — Sin señales relevantes hoy")

    lines += [
        "",
        f"<b>Score: {a['score']:+d}  |  Veredicto: {a['verdict']}</b>",
        "",
        "<i>⚠️ Esto es educativo, no asesoramiento financiero.</i>",
    ]
    return "\n".join(lines)


def format_summary(analyses: list, date_str: str) -> str:
    buys    = [a for a in analyses if a["score"] >= 2]
    sells   = [a for a in analyses if a["score"] <= -2]
    neutral = [a for a in analyses if -2 < a["score"] < 2]

    lines = [f"<b>📊 RESUMEN DIARIO — {date_str}</b>", ""]

    if buys:
        lines.append("<b>🟢 Señales de Compra:</b>")
        for a in sorted(buys, key=lambda x: x["score"], reverse=True):
            lines.append(f"  {a['ticker']} — {a['verdict']} (score: {a['score']:+d})")
        lines.append("")

    if sells:
        lines.append("<b>🔴 Señales de Venta:</b>")
        for a in sorted(sells, key=lambda x: x["score"]):
            lines.append(f"  {a['ticker']} — {a['verdict']} (score: {a['score']:+d})")
        lines.append("")

    if neutral:
        tickers = ", ".join(a["ticker"] for a in neutral)
        lines.append(f"<b>⚪ Neutral:</b> {tickers}")
        lines.append("")

    lines.append("<i>⚠️ Esto es educativo, no asesoramiento financiero.</i>")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg      = load_config()
    p        = cfg["strategy_params"]
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN") or cfg["telegram"]["bot_token"]
    tg_chat  = os.environ.get("TELEGRAM_CHAT_ID")  or cfg["telegram"]["chat_id"]
    tickers  = cfg["tickers"]
    date_str = datetime.now().strftime("%d/%m/%Y")

    print(f"\n🤖 Scanner de Señales | {date_str}")
    print(f"   Tickers: {', '.join(tickers)}\n")

    analyses = []

    for ticker in tickers:
        print(f"  Analizando {ticker}...", end=" ")
        try:
            df       = fetch_data(ticker)
            df       = add_indicators(df, p)
            analysis = analyze_ticker(ticker, df, p)
            analyses.append(analysis)

            msg = format_message(analysis, date_str)
            send_telegram(tg_token, tg_chat, msg)
            print(f"{analysis['verdict']}")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Resumen consolidado
    if len(analyses) > 1:
        summary_msg = format_summary(analyses, date_str)
        send_telegram(tg_token, tg_chat, summary_msg)
        print(f"\n  → Resumen enviado a Telegram")

    print("\n✅ Scanner finalizado.\n")


if __name__ == "__main__":
    main()
