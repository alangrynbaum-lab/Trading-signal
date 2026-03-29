"""
Scanner de Señales en Vivo — Trading Bot
Corre diariamente via GitHub Actions y manda alertas a Telegram.
"""

import json
import os
import time
import warnings
from datetime import datetime

import pandas as pd
import ta
import requests
import yfinance as yf

warnings.filterwarnings("ignore")


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def send_telegram(token: str, chat_id: str, message: str):
    url  = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        r = requests.post(url, data=data, timeout=10)
        r.raise_for_status()
        print(f"    → Telegram OK")
    except Exception as e:
        print(f"  ⚠️  Telegram error: {e}")


def fetch_data(ticker: str, retries: int = 3) -> pd.DataFrame:
    for attempt in range(retries):
        try:
            time.sleep(2)
            tk = yf.Ticker(ticker)
            df = tk.history(period="120d", interval="1d", auto_adjust=True)
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df
        except Exception as e:
            print(f"    Intento {attempt+1}/{retries} fallido: {e}")
            if attempt < retries - 1:
                time.sleep(5 * (attempt + 1))
    raise ValueError(f"No se pudo obtener datos para {ticker}")


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df     = df.copy()
    close  = df["Close"].squeeze()
    volume = df["Volume"].squeeze()
    df["ema_fast"]  = ta.trend.ema_indicator(close, window=p["ema_fast"])
    df["ema_slow"]  = ta.trend.ema_indicator(close, window=p["ema_slow"])
    df["rsi"]       = ta.momentum.rsi(close, window=p["rsi_period"])
    macd_obj        = ta.trend.MACD(close, window_fast=p["macd_fast"], window_slow=p["macd_slow"], window_sign=p["macd_signal"])
    df["macd"]      = macd_obj.macd()
    df["macd_sig"]  = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()
    df["vol_ma"]    = volume.rolling(20).mean()
    df["vol_spike"] = volume > df["vol_ma"] * p["volume_multiplier"]
    return df.dropna()


def analyze_ticker(ticker: str, df: pd.DataFrame, p: dict) -> dict:
    last  = df.iloc[-1]
    prev  = df.iloc[-2]
    price = float(last["Close"])
    signals = []
    score   = 0

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

    rsi = float(last["rsi"])
    if rsi < p["rsi_oversold"]:
        signals.append(f"🟢 RSI en sobreventa ({rsi:.1f}) — posible rebote")
        score += 2
    elif rsi > p["rsi_overbought"]:
        signals.append(f"🔴 RSI en sobrecompra ({rsi:.1f}) — posible reversión")
        score -= 2
    elif rsi > 60:
        score += 1

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

    if bool(last["vol_spike"]):
        vol_ratio = float(last["Volume"]) / float(last["vol_ma"])
        direction = "alcista" if float(last["Close"]) > float(last["Open"]) else "bajista"
        signals.append(f"🔊 Spike de volumen {direction} ({vol_ratio:.1f}x promedio)")
        score += 1 if direction == "alcista" else -1

    confluence_bull = ema_bull and macd_bull and rsi < p["rsi_overbought"] and bool(last["vol_spike"])
    confluence_bear = not ema_bull and not macd_bull and rsi > p["rsi_oversold"]

    if confluence_bull:
        signals.append("⭐ CONFLUENCIA BULLISH: EMA + MACD + Volumen alineados")
        score += 2
    elif confluence_bear:
        signals.append("⭐ CONFLUENCIA BEARISH: EMA + MACD + RSI alineados")
        score -= 2

    daily_chg = (float(last["Close"]) - float(prev["Close"])) / float(prev["Close"]) * 100

    if score >= 4:   verdict = "🟢 COMPRA FUERTE"
    elif score >= 2: verdict = "🟡 SEÑAL DE COMPRA"
    elif score <= -4: verdict = "🔴 VENTA FUERTE"
    elif score <= -2: verdict = "🟠 SEÑAL DE VENTA"
    else:             verdict = "⚪ NEUTRAL"

    return {"ticker": ticker, "price": price, "daily_chg": daily_chg, "rsi": rsi,
            "ema_fast": float(last["ema_fast"]), "ema_slow": float(last["ema_slow"]),
            "macd": float(last["macd"]), "macd_sig": float(last["macd_sig"]),
            "vol_spike": bool(last["vol_spike"]), "signals": signals, "score": score, "verdict": verdict}


def format_message(a: dict, date_str: str) -> str:
    emoji = "📈" if a["daily_chg"] >= 0 else "📉"
    lines = [
        f"<b>🤖 SEÑAL DIARIA — {date_str}</b>",
        f"<b>{a['ticker']}</b>  {emoji} ${a['price']:.2f} ({a['daily_chg']:+.2f}%)",
        "",
        "<b>Indicadores:</b>",
        f"  RSI: {a['rsi']:.1f}",
        f"  EMA20: {a['ema_fast']:.2f}  |  EMA50: {a['ema_slow']:.2f}",
        f"  MACD: {a['macd']:.3f}  |  Signal: {a['macd_sig']:.3f}",
        f"  Volumen alto: {'Sí ⚡' if a['vol_spike'] else 'No'}",
        "",
        "<b>Señales detectadas:</b>",
    ]
    lines += [f"  {s}" for s in a["signals"]] if a["signals"] else ["  — Sin señales relevantes hoy"]
    lines += ["", f"<b>Score: {a['score']:+d}  |  Veredicto: {a['verdict']}</b>", "",
              "<i>⚠️ Esto es educativo, no asesoramiento financiero.</i>"]
    return "\n".join(lines)


def format_summary(analyses: list, date_str: str) -> str:
    buys    = [a for a in analyses if a["score"] >= 2]
    sells   = [a for a in analyses if a["score"] <= -2]
    neutral = [a for a in analyses if -2 < a["score"] < 2]
    lines   = [f"<b>📊 RESUMEN DIARIO — {date_str}</b>", ""]
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
        lines.append(f"<b>⚪ Neutral:</b> {', '.join(a['ticker'] for a in neutral)}")
        lines.append("")
    lines.append("<i>⚠️ Esto es educativo, no asesoramiento financiero.</i>")
    return "\n".join(lines)


def main():
    cfg      = load_config()
    p        = cfg["strategy_params"]
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN") or cfg["telegram"]["bot_token"]
    tg_chat  = os.environ.get("TELEGRAM_CHAT_ID")   or cfg["telegram"]["chat_id"]
    tickers  = cfg["tickers"]
    date_str = datetime.now().strftime("%d/%m/%Y")

    print(f"\n🤖 Scanner de Señales | {date_str}")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Chat ID configurado: {'SI' if tg_chat else 'NO'}\n")

    analyses = []

    for ticker in tickers:
        print(f"\n  Analizando {ticker}...")
        try:
            df       = fetch_data(ticker)
            df       = add_indicators(df, p)
            analysis = analyze_ticker(ticker, df, p)
            analyses.append(analysis)
            send_telegram(tg_token, tg_chat, format_message(analysis, date_str))
            print(f"  {ticker}: {analysis['verdict']}")
        except Exception as e:
            print(f"  ❌ {ticker} Error: {e}")

    if len(analyses) > 1:
        send_telegram(tg_token, tg_chat, format_summary(analyses, date_str))
        print(f"\n  → Resumen enviado a Telegram")

    print("\n✅ Scanner finalizado.\n")


if __name__ == "__main__":
    main()
