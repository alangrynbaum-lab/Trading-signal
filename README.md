# 🤖 Trading Bot — Señales & Backtesting

Bot de análisis técnico que corre gratis en **GitHub Actions**.  
Manda alertas diarias a **Telegram** y hace backtesting semanal.

---

## Estrategias incluidas

| Estrategia | Descripción |
|---|---|
| EMA Crossover | BUY cuando EMA20 cruza arriba EMA50 |
| RSI Extremos | BUY en sobreventa (<30), SELL en sobrecompra (>70) |
| MACD Crossover | BUY cuando MACD cruza arriba signal line |
| Confluencia | Las 3 anteriores + volumen alineados |
| Volume Breakout | Movimiento de precio con volumen anormalmente alto |

---

## Setup (15 minutos)

### 1. Subir a GitHub

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/TU_USUARIO/trading-bot.git
git push -u origin main
```

### 2. Configurar tickers

Editá `config.json` y reemplazá los tickers:

```json
{
  "tickers": ["SPY", "VGT", "SCHD", "MSTR", "BTC-USD"],
  ...
}
```

**Formatos de ticker válidos:**
- Acciones US: `SPY`, `AAPL`, `TSLA`
- ETFs: `VGT`, `SCHD`, `GLD`
- Crypto (via Yahoo Finance): `BTC-USD`, `ETH-USD`
- CEDEARs: NO disponibles via yfinance — ver nota abajo

### 3. Configurar Telegram

Tu bot de Telegram ya existe. Solo necesitás el token y el chat_id.

En GitHub repo → **Settings → Secrets and variables → Actions → New repository secret**:

| Secret | Valor |
|---|---|
| `TELEGRAM_BOT_TOKEN` | El token de tu bot (lo tenés en Railway) |
| `TELEGRAM_CHAT_ID` | Tu chat ID personal |

### 4. Actualizar signals.py para leer secrets

En `signals.py`, el `load_config()` ya lee el config.json.  
Para que GitHub Actions pase los secrets, el script los lee así:

```python
import os

tg_token = os.environ.get("TELEGRAM_BOT_TOKEN") or cfg["telegram"]["bot_token"]
tg_chat  = os.environ.get("TELEGRAM_CHAT_ID")  or cfg["telegram"]["chat_id"]
```

**Este cambio ya está aplicado en la versión final del script.**

---

## Uso local

```bash
pip install -r requirements.txt

# Backtesting (crea gráficos en /results)
python backtester.py
python backtester.py SPY VGT MSTR   # tickers específicos

# Scanner de señales
python signals.py
```

---

## Schedule

| Job | Horario | Descripción |
|---|---|---|
| Señales | Lun-Vie 11:35 AM ET | Análisis post-apertura NY |
| Señales | Lun-Vie 6:05 PM ET | Análisis post-cierre NY |
| Backtest | Domingos 7 AM ET | Reporte semanal completo |

Podés correrlo manualmente en cualquier momento desde:  
**GitHub → Actions → Trading Bot → Run workflow**

---

## Nota sobre CEDEARs

Yahoo Finance no tiene datos de CEDEARs argentinos.  
Opciones para incorporarlos en el futuro:
- **Bull Market API** (cuando abras la cuenta y tengas acceso)
- **IOL API** (Invertir Online tiene API pública)
- Scraping de Rava Bursátil (no recomendado)

---

## Roadmap

- [x] Backtesting engine (5 estrategias)
- [x] Scanner de señales en vivo
- [x] Alertas Telegram
- [x] GitHub Actions (gratis, sin servidor)
- [ ] Paper trading con registro de P&L
- [ ] Reporte PDF semanal
- [ ] Soporte CEDEARs (Bull Market API)
- [ ] Ejecución real (IBKR / Binance APIs)
