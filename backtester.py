"""
Backtesting Engine — Trading Bot
Estrategias: EMA Crossover, RSI Extremos, MACD Crossover, Confluencia, Volumen
Timeframe: Diario (1D)
"""

import json
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ta
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def fetch_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna()


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    close  = df["Close"]
    volume = df["Volume"]
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


def signal_ema_crossover(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))] =  1
    sig[(df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))] = -1
    return sig

def signal_rsi(df, oversold, overbought):
    sig = pd.Series(0, index=df.index)
    sig[(df["rsi"] < oversold)   & (df["rsi"].shift(1) >= oversold)]   =  1
    sig[(df["rsi"] > overbought) & (df["rsi"].shift(1) <= overbought)] = -1
    return sig

def signal_macd(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1))] =  1
    sig[(df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1))] = -1
    return sig

def signal_confluence(df, p):
    sig  = pd.Series(0, index=df.index)
    buy  = (df["ema_fast"] > df["ema_slow"]) & (df["macd"] > df["macd_sig"]) & (df["rsi"] < p["rsi_overbought"]) & df["vol_spike"]
    sell = (df["ema_fast"] < df["ema_slow"]) & (df["macd"] < df["macd_sig"])
    sig[buy  & ~buy.shift(1).fillna(False)]  =  1
    sig[sell & ~sell.shift(1).fillna(False)] = -1
    return sig

def signal_volume_breakout(df):
    sig = pd.Series(0, index=df.index)
    sig[(df["Close"] > df["Close"].shift(1)) & df["vol_spike"]] =  1
    sig[(df["Close"] < df["Close"].shift(1)) & df["vol_spike"]] = -1
    return sig


def run_backtest(df, signals, capital, stop_loss, take_profit):
    equity = [capital]; cash = capital; position = 0; entry_px = 0.0; trades = []
    prices = df["Close"].values; dates = df.index; sigs = signals.values

    for i in range(len(df)):
        price = float(prices[i]); date = dates[i]
        if position > 0:
            change = (price - entry_px) / entry_px
            if change <= -stop_loss or change >= take_profit:
                pnl = position * (price - entry_px); cash += position * price
                if trades: trades[-1].update({"exit_date": date, "exit_price": price, "pnl": pnl, "exit_reason": "SL" if change <= -stop_loss else "TP"})
                position = 0
        if sigs[i] == 1 and position == 0 and cash > price:
            shares = int(cash // price); cash -= shares * price; position = shares; entry_px = price
            trades.append({"entry_date": date, "entry_price": price, "shares": shares})
        elif sigs[i] == -1 and position > 0:
            pnl = position * (price - entry_px); cash += position * price
            if trades: trades[-1].update({"exit_date": date, "exit_price": price, "pnl": pnl, "exit_reason": "signal"})
            position = 0
        equity.append(cash + position * price)

    if position > 0:
        price = float(prices[-1]); pnl = position * (price - entry_px); cash += position * price
        if trades: trades[-1].update({"exit_date": dates[-1], "exit_price": price, "pnl": pnl, "exit_reason": "EOD"})

    equity_s = pd.Series(equity[1:], index=df.index)
    completed = [t for t in trades if "pnl" in t]
    wins = [t for t in completed if t["pnl"] > 0]
    rets = equity_s.pct_change().dropna()
    return {
        "equity_curve": equity_s,
        "trades": completed,
        "total_return_pct": round((equity_s.iloc[-1] - capital) / capital * 100, 2),
        "win_rate_pct": round(len(wins) / len(completed) * 100 if completed else 0, 2),
        "max_drawdown_pct": round(((equity_s - equity_s.cummax()) / equity_s.cummax() * 100).min(), 2),
        "sharpe_ratio": round(float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0, 2),
        "total_trades": len(completed),
        "final_capital": round(float(equity_s.iloc[-1]), 2),
    }


def main():
    cfg = load_config(); p = cfg["strategy_params"]; risk = cfg["risk"]
    tickers = sys.argv[1:] if len(sys.argv) > 1 else cfg["tickers"]
    print(f"\n🤖 Backtesting | Capital: ${risk['capital_per_trade']:,} | Tickers: {', '.join(tickers)}\n")
    summary = []

    for ticker in tickers:
        print(f"📊 {ticker}...")
        try:
            df = add_indicators(fetch_data(ticker, cfg["lookback_days"]), p)
        except Exception as e:
            print(f"  ❌ {e}"); continue

        strategies = {
            "EMA Crossover":  signal_ema_crossover(df),
            "RSI Extremos":   signal_rsi(df, p["rsi_oversold"], p["rsi_overbought"]),
            "MACD Crossover": signal_macd(df),
            "Confluencia":    signal_confluence(df, p),
            "Vol Breakout":   signal_volume_breakout(df),
        }

        for name, sigs in strategies.items():
            m = run_backtest(df, sigs, risk["capital_per_trade"], risk["stop_loss_pct"], risk["take_profit_pct"])
            print(f"  {name}: Ret={m['total_return_pct']}% | WR={m['win_rate_pct']}% | DD={m['max_drawdown_pct']}% | Sharpe={m['sharpe_ratio']}")
            summary.append({"ticker": ticker, "strategy": name, **{k: v for k, v in m.items() if k not in ("equity_curve","trades")}})

    if summary:
        Path("results").mkdir(exist_ok=True)
        pd.DataFrame(summary).sort_values("total_return_pct", ascending=False).to_csv("results/summary.csv", index=False)
        print("\n✅ Resultados guardados en results/summary.csv\n")

if __name__ == "__main__":
    main()
