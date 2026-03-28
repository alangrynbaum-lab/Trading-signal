name: Trading Bot — Señales Diarias

on:
  schedule:
    - cron: "35 14 * * 1-5"
    - cron: "05 21 * * 1-5"
  workflow_dispatch:

jobs:
  signals:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Instalar dependencias
        run: pip install -r requirements.txt

      - name: Correr scanner de señales
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python signals.py
