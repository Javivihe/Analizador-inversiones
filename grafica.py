import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def plot_stock_and_return(flag, ticker, start_date, end_date=None, grafica=False):
    if end_date is None:
        end_date = pd.Timestamp.now()
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date, interval='1d')
    # Filtrar solo días con datos válidos
    stock_data = stock_data.dropna(subset=['Close', 'Open'])
    fechas = stock_data.index.strftime('%Y-%m-%d').tolist()
    valores = stock_data['Close'].tolist()
    retornos = ((stock_data['Close'] - stock_data['Open']) / stock_data['Open']).tolist()
    
    df = pd.DataFrame({
        'stock': ticker,
        'open': stock_data['Open'].values,
        'close': stock_data['Close'].values,
        'retorno': (stock_data['Close'] - stock_data['Open']) / stock_data['Open'],
        'high': stock_data['High'].values,
        'low': stock_data['Low'].values,

        # === Medias Móviles (SMA) ===
        'SMA_5':  stock_data['Close'].rolling(window=5).mean(),
        'SMA_10':  stock_data['Close'].rolling(window=10).mean(),
        'SMA_20':  stock_data['Close'].rolling(window=20).mean(),

        # === Volatilidad (Desviación estándar de los retornos) ===

        'volatility_5': stock_data['Close'].rolling(window=5).std(),
        'volatility_10': stock_data['Close'].rolling(window=10).std()

    }, index=stock_data.index.strftime('%Y-%m-%d').str[:10])

    # === 5. RSI ===
    df['RSI'] = ta.momentum.RSIIndicator(close=stock_data['Close'], window=14).rsi()

    # === 6. MACD ===
    macd = ta.trend.MACD(close=stock_data['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    df.dropna(inplace=True)

    if flag == 'analisis_basico':

        df['signal_sma'] = 0
        df.loc[df['SMA_5'] > df['SMA_20'], 'signal_sma'] = 1
        df.loc[df['SMA_5'] < df['SMA_20'], 'signal_sma'] = -1

        df['signal_rsi'] = 0
        df.loc[df['RSI'] < 30, 'signal_rsi'] = 1
        df.loc[df['RSI'] > 70, 'signal_rsi'] = -1

        df['signal_macd'] = 0
        df.loc[df['MACD'] > df['MACD_signal'], 'signal_macd'] = 1
        df.loc[df['MACD'] < df['MACD_signal'], 'signal_macd'] = -1

        df['signal_total'] = df['signal_sma'] + df['signal_rsi'] + df['signal_macd']

        # Señal final:
        df['resultado_final'] = 0
        df.loc[df['signal_total'] >= 2, 'resultado_final'] = 1
        df.loc[df['signal_total'] <= -2, 'resultado_final'] = -1

    elif flag == 'random_forest':
        # MACHINE LEARNING

        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        features = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_signal', 'volatility_5', 'volatility_10', 'retorno']
        X = df[features]
        y = df['target']

        # Dividir en entrenamiento y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Entrenar
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Evaluar
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))
        df['prediction'] = model.predict(X)
        df['resultado_final'] = df['prediction'].replace({1: 1, 0: -1})  # 1 = compra, -1 = venta

    if grafica == True:
        fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
        # Subplot 1: Precio de cierre + SMA + señales
        axs[0].plot(df['close'], label='Precio de Cierre', color='blue')
        axs[0].plot(df['SMA_5'], label='SMA 5', alpha=0.6)
        axs[0].plot(df['SMA_20'], label='SMA 20', alpha=0.6)
        buy_signals = df[df['resultado_final'] == 1]
        sell_signals = df[df['resultado_final'] == -1]
        axs[0].scatter(buy_signals.index, buy_signals['close'], label='Compra', marker='^', color='green', s=100)
        axs[0].scatter(sell_signals.index, sell_signals['close'], label='Venta', marker='v', color='red', s=100)
        axs[0].set_title("Precio de cierre, SMA y señales de compra/venta")
        axs[0].legend()
        axs[0].grid()
        # Subplot 2: Evolución del valor de cierre con colores
        fechas = df.index.tolist()
        valores = df['close'].tolist()
        for i in range(1, len(valores)):
            color = 'green' if valores[i] >= valores[i-1] else 'red'
            axs[1].plot(fechas[i-1:i+1], valores[i-1:i+1], color=color, marker='o')
        axs[1].set_title(f'Evolución del valor de cierre: {ticker}')
        axs[1].set_ylabel('Valor de cierre')
        axs[1].legend(['Sube/Baja'], loc='upper left')
        axs[1].grid(True)
        # Subplot 3: Retorno diario
        axs[2].plot(fechas, df['retorno'], color='purple', linestyle='-', marker='o', label='Retorno diario')
        axs[2].set_title('Evolución del retorno diario')
        axs[2].set_xlabel('Fecha')
        axs[2].set_ylabel('Retorno diario')
        axs[2].legend()
        axs[2].grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return df

def backtest_strategy(df):

    """
    Realiza un backtest de la estrategia de trading basada en señales generadas.
    """
    
    initial_cash = 1_000  # Capital inicial
    cash = initial_cash
    position = 0  # Número de acciones compradas
    in_market = False  # Si estamos dentro o no
    df['portfolio_value'] = 0  # Valor total cartera
    df['trade'] = ''  # Registro de operación

    for i in range(len(df)):
        signal = df.iloc[i]['resultado_final']
        close_price = df.iloc[i]['close']

        if signal == 1 and not in_market:
            # Comprar
            position = cash / close_price
            cash = 0
            in_market = True
            df.iloc[i, df.columns.get_loc('trade')] = 'BUY'

        elif signal == -1 and in_market:
            # Vender
            cash = position * close_price
            position = 0
            in_market = False
            df.iloc[i, df.columns.get_loc('trade')] = 'SELL'

        # Actualizar valor total de cartera
        portfolio_value = cash + (position * close_price)
        df.iloc[i, df.columns.get_loc('portfolio_value')] = portfolio_value

    # Si aún estamos dentro al final, cerramos posición
    if in_market:
        final_price = df.iloc[-1]['close']
        cash = position * final_price
        position = 0
        in_market = False

    # Resultado final
    final_value = cash
    final_value_roi = (final_value - initial_cash) / initial_cash * 100

    # Para comparar: estrategia pasiva de mantener
    buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100

   

    return final_value_roi, buy_hold_return


def estrategia(TICKERS, flags):


    for flag in flags:
        
        cont_estrategia = 0
        cont_total = 0

        print(f"Evaluando estrategia con flag: {flag}")

        for i in TICKERS:
            
            print(f"Procesando {i}...")

            df = plot_stock_and_return(flag, i, '2025-01-01', grafica=True)
            final_value_roi, buy_hold_return = backtest_strategy(df)

            print(f"Resultado de estrategia: ROI = {final_value_roi:.2f}%)")
            print(f"Estrategia de mantener: ROI = {buy_hold_return:.2f}%")

            if (final_value_roi > buy_hold_return):

                cont_estrategia = cont_estrategia + 1
            cont_total = cont_total + 1

        print(f"Total de acciones analizadas: {cont_total}")
        print(f"Acciones con mejor ROI que estrategia de mantener: {cont_estrategia}")
        print("--------------------------------------------------\n")


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
    "NVDA", "AAPL", "TSLA"
]

estrategia(TICKERS_RED, ['random_forest'])
