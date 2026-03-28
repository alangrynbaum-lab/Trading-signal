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
import pandas_ta as ta
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def fetch_data(ticker: str, lookback_days: int) -> pd.DataFrame:
    period = f"{lookback_days}d"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data for {ticker}")
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def add_indicators(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = ta.ema(df["Close"], length=p["ema_fast"])
    df["ema_slow"] = ta.ema(df["Close"], length=p["ema_slow"])
    df["rsi"]      = ta.rsi(df["Close"], length=p["rsi_period"])
    macd           = ta.macd(df["Close"], fast=p["macd_fast"], slow=p["macd_slow"], signal=p["macd_signal"])
    df["macd"]     = macd[f"MACD_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]
    df["macd_sig"] = macd[f"MACDs_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]
    df["macd_hist"]= macd[f"MACDh_{p['macd_fast']}_{p['macd_slow']}_{p['macd_signal']}"]
    df["vol_ma"]   = df["Volume"].rolling(20).mean()
    df["vol_spike"]= df["Volume"] > df["vol_ma"] * p["volume_multiplier"]
    return df.dropna()


# ── Estrategias ───────────────────────────────────────────────────────────────

def signal_ema_crossover(df: pd.DataFrame) -> pd.Series:
    """BUY cuando EMA20 cruza arriba EMA50, SELL cuando cruza abajo."""
    sig = pd.Series(0, index=df.index)
    cross_up   = (df["ema_fast"] > df["ema_slow"]) & (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1))
    cross_down = (df["ema_fast"] < df["ema_slow"]) & (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1))
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def signal_rsi(df: pd.DataFrame, oversold: float, overbought: float) -> pd.Series:
    """BUY en RSI < oversold (rebote), SELL en RSI > overbought (reversión)."""
    sig = pd.Series(0, index=df.index)
    buy  = (df["rsi"] < oversold)  & (df["rsi"].shift(1) >= oversold)
    sell = (df["rsi"] > overbought) & (df["rsi"].shift(1) <= overbought)
    sig[buy]  =  1
    sig[sell] = -1
    return sig


def signal_macd(df: pd.DataFrame) -> pd.Series:
    """BUY cuando MACD cruza arriba signal line, SELL cuando cruza abajo."""
    sig = pd.Series(0, index=df.index)
    cross_up   = (df["macd"] > df["macd_sig"]) & (df["macd"].shift(1) <= df["macd_sig"].shift(1))
    cross_down = (df["macd"] < df["macd_sig"]) & (df["macd"].shift(1) >= df["macd_sig"].shift(1))
    sig[cross_up]   =  1
    sig[cross_down] = -1
    return sig


def signal_confluence(df: pd.DataFrame, p: dict) -> pd.Series:
    """BUY cuando EMA crossover + MACD bullish + RSI no overbought + volumen alto."""
    sig = pd.Series(0, index=df.index)
    ema_bull  = df["ema_fast"] > df["ema_slow"]
    macd_bull = df["macd"] > df["macd_sig"]
    rsi_ok    = df["rsi"] < p["rsi_overbought"]
    vol_ok    = df["vol_spike"]
    buy = ema_bull & macd_bull & rsi_ok & vol_ok
    # Señal de entrada: primer día que se cumplen todas
    sig[buy & ~buy.shift(1).fillna(False)] = 1
    ema_bear  = df["ema_fast"] < df["ema_slow"]
    macd_bear = df["macd"] < df["macd_sig"]
    sell = ema_bear & macd_bear
    sig[sell & ~sell.shift(1).fillna(False)] = -1
    return sig


def signal_volume_breakout(df: pd.DataFrame) -> pd.Series:
    """BUY cuando precio sube con volumen alto (breakout), SELL cuando baja con volumen alto."""
    sig = pd.Series(0, index=df.index)
    price_up   = df["Close"] > df["Close"].shift(1)
    price_down = df["Close"] < df["Close"].shift(1)
    buy  = price_up   & df["vol_spike"]
    sell = price_down & df["vol_spike"]
    sig[buy]  =  1
    sig[sell] = -1
    return sig


# ── Backtest engine ───────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, signals: pd.Series, capital: float,
                 stop_loss: float, take_profit: float) -> dict:
    """
    Simulación long-only con stop loss y take profit.
    Retorna dict con métricas y equity curve.
    """
    equity    = [capital]
    cash      = capital
    position  = 0      # shares held
    entry_px  = 0.0
    trades    = []

    prices = df["Close"].values
    dates  = df.index
    sigs   = signals.values

    for i in range(len(df)):
        price = float(prices[i])
        date  = dates[i]

        # Check stop loss / take profit si estamos en posición
        if position > 0:
            change = (price - entry_px) / entry_px
            if change <= -stop_loss or change >= take_profit:
                pnl  = position * (price - entry_px)
                cash += position * price
                trades.append({
                    "exit_date": date,
                    "exit_price": price,
                    "pnl": pnl,
                    "exit_reason": "SL" if change <= -stop_loss else "TP"
                })
                position = 0

        sig = sigs[i]

        if sig == 1 and position == 0 and cash > price:
            # BUY
            shares   = int(cash // price)
            cost     = shares * price
            cash    -= cost
            position = shares
            entry_px = price
            if trades and "exit_date" in trades[-1]:
                trades[-1]["entry_date"]  = date
                trades[-1]["entry_price"] = price
            trades.append({"entry_date": date, "entry_price": price, "shares": shares})

        elif sig == -1 and position > 0:
            # SELL
            pnl  = position * (price - entry_px)
            cash += position * price
            if trades:
                trades[-1].update({"exit_date": date, "exit_price": price, "pnl": pnl, "exit_reason": "signal"})
            position = 0

        equity.append(cash + position * price)

    # Cerrar posición abierta al final
    if position > 0:
        price = float(prices[-1])
        pnl   = position * (price - entry_px)
        cash += position * price
        if trades:
            trades[-1].update({"exit_date": dates[-1], "exit_price": price, "pnl": pnl, "exit_reason": "EOD"})
        equity.append(cash)

    equity_s = pd.Series(equity[1:], index=df.index)

    # Métricas
    total_return = (equity_s.iloc[-1] - capital) / capital * 100
    completed    = [t for t in trades if "pnl" in t]
    wins         = [t for t in completed if t["pnl"] > 0]
    win_rate     = len(wins) / len(completed) * 100 if completed else 0

    # Drawdown máximo
    roll_max   = equity_s.cummax()
    drawdown   = (equity_s - roll_max) / roll_max * 100
    max_dd     = drawdown.min()

    # Sharpe (diario, anualizado)
    rets   = equity_s.pct_change().dropna()
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    return {
        "equity_curve": equity_s,
        "trades": completed,
        "total_return_pct": round(total_return, 2),
        "win_rate_pct": round(win_rate, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "total_trades": len(completed),
        "final_capital": round(float(equity_s.iloc[-1]), 2),
    }


# ── Reporte ───────────────────────────────────────────────────────────────────

def print_report(ticker: str, strategy: str, metrics: dict):
    print(f"\n{'='*55}")
    print(f"  {ticker} | {strategy}")
    print(f"{'='*55}")
    print(f"  Retorno total   : {metrics['total_return_pct']:>8.2f}%")
    print(f"  Win rate        : {metrics['win_rate_pct']:>8.2f}%")
    print(f"  Max drawdown    : {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Sharpe ratio    : {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Trades totales  : {metrics['total_trades']:>8}")
    print(f"  Capital final   : ${metrics['final_capital']:>10,.2f}")
    print(f"{'='*55}")


def plot_results(ticker: str, df: pd.DataFrame, results: dict, output_dir: str = "results"):
    Path(output_dir).mkdir(exist_ok=True)
    strategies = list(results.keys())
    n = len(strategies)

    fig, axes = plt.subplots(n + 1, 1, figsize=(14, 4 * (n + 1)))
    fig.patch.set_facecolor("#0d1117")

    # Precio + EMAs
    ax0 = axes[0]
    ax0.set_facecolor("#0d1117")
    ax0.plot(df.index, df["Close"],    color="#c9d1d9", lw=1.2, label="Close")
    ax0.plot(df.index, df["ema_fast"], color="#58a6ff", lw=1,   label="EMA20", linestyle="--")
    ax0.plot(df.index, df["ema_slow"], color="#f78166", lw=1,   label="EMA50", linestyle="--")
    ax0.set_title(f"{ticker} — Precio & EMAs", color="#c9d1d9", fontsize=11)
    ax0.legend(fontsize=8, facecolor="#161b22", labelcolor="#c9d1d9")
    ax0.tick_params(colors="#8b949e")
    for spine in ax0.spines.values():
        spine.set_edgecolor("#30363d")

    # Equity curves
    colors = ["#3fb950", "#58a6ff", "#d2a8ff", "#ffa657", "#f78166"]
    for i, (strat, metrics) in enumerate(results.items()):
        ax = axes[i + 1]
        ax.set_facecolor("#0d1117")
        equity = metrics["equity_curve"]
        color  = colors[i % len(colors)]
        ax.plot(equity.index, equity, color=color, lw=1.5)
        ax.axhline(equity.iloc[0], color="#8b949e", lw=0.8, linestyle=":")
        ax.fill_between(equity.index, equity.iloc[0], equity, alpha=0.15, color=color)
        label = (f"{strat} | Ret: {metrics['total_return_pct']}% | "
                 f"WR: {metrics['win_rate_pct']}% | DD: {metrics['max_drawdown_pct']}% | "
                 f"Sharpe: {metrics['sharpe_ratio']}")
        ax.set_title(label, color="#c9d1d9", fontsize=9)
        ax.tick_params(colors="#8b949e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    plt.tight_layout(pad=2)
    out = f"{output_dir}/{ticker}_backtest.png"
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  → Gráfico guardado: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg     = load_config()
    p       = cfg["strategy_params"]
    risk    = cfg["risk"]
    capital = risk["capital_per_trade"]
    tickers = cfg["tickers"]

    if len(sys.argv) > 1:
        tickers = sys.argv[1:]

    print(f"\n🤖 Backtesting Engine | Capital por trade: ${capital:,} | Timeframe: 1D")
    print(f"   Tickers: {', '.join(tickers)}")
    print(f"   Período: {cfg['lookback_days']} días\n")

    summary = []

    for ticker in tickers:
        print(f"\n📊 Procesando {ticker}...")
        try:
            df  = fetch_data(ticker, cfg["lookback_days"])
            df  = add_indicators(df, p)
        except Exception as e:
            print(f"  ❌ Error obteniendo datos: {e}")
            continue

        strategies = {
            "EMA Crossover":    signal_ema_crossover(df),
            "RSI Extremos":     signal_rsi(df, p["rsi_oversold"], p["rsi_overbought"]),
            "MACD Crossover":   signal_macd(df),
            "Confluencia":      signal_confluence(df, p),
            "Volume Breakout":  signal_volume_breakout(df),
        }

        results = {}
        for name, signals in strategies.items():
            metrics = run_backtest(
                df, signals, capital,
                risk["stop_loss_pct"], risk["take_profit_pct"]
            )
            results[name] = metrics
            print_report(ticker, name, metrics)
            summary.append({"ticker": ticker, "strategy": name, **{k: v for k, v in metrics.items() if k != "equity_curve" and k != "trades"}})

        plot_results(ticker, df, results)

    # Tabla resumen final
    if summary:
        print(f"\n{'='*75}")
        print(f"  RESUMEN COMPARATIVO")
        print(f"{'='*75}")
        df_sum = pd.DataFrame(summary)
        df_sum = df_sum.sort_values("total_return_pct", ascending=False)
        print(df_sum[["ticker", "strategy", "total_return_pct", "win_rate_pct",
                      "max_drawdown_pct", "sharpe_ratio", "total_trades"]].to_string(index=False))
        print(f"{'='*75}\n")
        df_sum.to_csv("results/summary.csv", index=False)
        print("  → Resumen guardado: results/summary.csv")


if __name__ == "__main__":
    main()
