import yfinance as yf
import time

# Lista de tickers (adaptada de la que diste; algunos nombres cambiados a tickers válidos)
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

# Parámetros
UMBRAL_CAMBIO = 2.0  # porcentaje para avisar
DURACION_MINUTOS = 5
SLEEP_INTERVAL = 60  # segundos entre chequeos

# Diccionario para almacenar precios históricos {ticker: [(timestamp, precio)]}
historial_precios = {}

def obtener_precios(tickers):
    data = yf.download(tickers, period="1d", interval="1m", group_by='ticker', threads=True)
    precios = {}
    for ticker in tickers:
        try:
            # Tomamos el último precio de cierre
            precio = data[ticker]['Close'][-1]
            precios[ticker] = precio
        except Exception:
            # Si no se puede obtener precio, se ignora
            continue
    return precios

print("🔄 Iniciando monitoreo con yfinance...\n")

while True:
    timestamp = time.time()
    precios_actuales = obtener_precios(TICKERS)

    for ticker, precio in precios_actuales.items():
        if ticker not in historial_precios:
            historial_precios[ticker] = []

        historial_precios[ticker].append((timestamp, precio))
        # Limpiamos precios fuera de ventana de DURACION_MINUTOS
        historial_precios[ticker] = [
            (t, p) for t, p in historial_precios[ticker]
            if timestamp - t <= DURACION_MINUTOS * 60
        ]

        if len(historial_precios[ticker]) >= 2:
            precio_inicial = historial_precios[ticker][0][1]
            cambio = ((precio - precio_inicial) / precio_inicial) * 100

            if cambio >= UMBRAL_CAMBIO:
                print(f"🚀 {ticker} subió {cambio:.2f}% en {DURACION_MINUTOS} min")
            elif cambio <= -UMBRAL_CAMBIO:
                print(f"🔻 {ticker} bajó {cambio:.2f}% en {DURACION_MINUTOS} min")

    time.sleep(SLEEP_INTERVAL)
