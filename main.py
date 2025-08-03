import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from estrategia_backend import estrategia
import optuna


def objective(trial):
    # Buscar n_days entre 1 y 10 (enteros)
    n_days = trial.suggest_int("n_days", 1, 10)

    # Buscar threshold entre 0.01 y 0.2 (floats)
    threshold = trial.suggest_float("threshold", 0.03, 0.05)
    print(f"Probando n_days={n_days}, threshold={threshold}")
    # Ejecutar la función que quieres maximizar
    resultado = estrategia(
        SP500,
        ['analisis_lstm_multiclase'],
        '2020-01-01',
        param_grafica=False,        # ⚠️ Desactivamos gráficos para más velocidad
        comparar_features=False,
        n_days=n_days,
        threshold=threshold
    )

    # Devolver el valor que queremos maximizar
    return resultado  # ⬅️ Asegúrate de que esto sea escalar y representativo (por ejemplo accuracy o sharpe)



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
GANADORAS = ['AAPL', 'AMZN', 'O', 'AMD', 'PEP', 'SBUX']
SP500 = ['VUSA.DE']


# # Crear y ejecutar el estudio
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=30)  # Cambia a más si necesitas más precisión

# # Mostrar resultados
# print("🎯 Mejor valor:", study.best_value)
# print("📈 Mejores parámetros:", study.best_params)


estrategia(
        SP500,
        ['analisis_lstm_multiclase'],
        '2020-01-01',
        param_grafica=False,        # ⚠️ Desactivamos gráficos para más velocidad
        comparar_features=False,
        n_days=6,
        threshold=0.04049634189642107
)