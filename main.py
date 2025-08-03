import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from funciones_utiles import estrategia




TICKERS = [
    "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "VOO", "VOO", "GOOGL", "META", "KO",
    "RR.L", "O", "AMD", "PLTR", "MCD", "IBM", "JPM", "JNJ", "VEU", "PEP", "SBUX",
    "BP", "WMT", "CSCO", "PG", "MA", "CVX", "MAIN", "AGNC", "BLK", "GD", "STAG",
    "AVGO", "NIO", "LTC", "ADP", "LOW", "CB", "KMB", "TROW", "BMO", "RY", "BNS",
    "ITW", "ECL", "NUE", "CNQ", "EMR", "CAH", "SYY", "AFL", "PPG", "ROP", "SHW",
    "GWW", "ADC", "BMY", "IAU", "BAESY", "GOOD", "SLB", "TD", "LYG", "NFLX", "ASML",
    "INTC", "LGEN.L", "RHM.DE", "NGG", "NKE", "QCOM", "BCS", "C", "BABA", "SHEL", 
    "ORCL", "SPY", "RGLD", "DUK", "COIN", "DIS", "PYPL", "GOOG", "MSTR", "BTI",
    "GSK", "CSL.AX", "SPYG", "RIO", "TSCO.L", "V", "AV.L", "PSEC", "BRK-B", "TSM",
    "AZN", "HSBC"
]

TICKERS_RED = [
    "NVDA", "AAPL", "TSLA", "MSFT", "AMZN", "VOO", "VOO", "GOOGL", "META", "KO",
    "RR.L", "O", "AMD", "PLTR", "MCD", "IBM", "JPM", "JNJ", "VEU", "PEP", "SBUX"
]
RED = [
    "NVDA", "RR.L"
]

estrategia(TICKERS_RED, ['analisis_lstm_multiclase'], '2020-01-01', False)
