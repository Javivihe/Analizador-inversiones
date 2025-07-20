import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from funciones_analisis import analisis_basico, analisis_random_forest, analisis_lstm, analisis_lstm_multiclase, analisis_xgb_multiclase

def plot_stock_and_return(flag, ticker, start_date, end_date=None, grafica=False):
    if end_date is None:
        end_date = pd.Timestamp.now()
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date, interval='1d')
    # Filtrar solo días con datos válidos
    stock_data = stock_data.dropna(subset=['Close', 'Open'])
    # Las siguientes variables no se usan en la función principal, solo en la visualización
    
    df = pd.DataFrame({
        'stock': ticker,
        'open': stock_data['Open'].values,
        'close': stock_data['Close'].values,
        'retorno': (stock_data['Close'] - stock_data['Open']) / stock_data['Open'],
        'high': stock_data['High'].values,
        'low': stock_data['Low'].values,
        'volume': stock_data['Volume'].values,
        # Medias Móviles (SMA)
        'SMA_5':  stock_data['Close'].rolling(window=5).mean(),
        'SMA_10':  stock_data['Close'].rolling(window=10).mean(),
        'SMA_20':  stock_data['Close'].rolling(window=20).mean(),
        # Volatilidad
        'volatility_5': stock_data['Close'].rolling(window=5).std(),
        'volatility_10': stock_data['Close'].rolling(window=10).std()
    }, index=stock_data.index.strftime('%Y-%m-%d').str[:10])

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=stock_data['Close'], window=14).rsi()
    # MACD
    macd = ta.trend.MACD(close=stock_data['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    df.dropna(inplace=True)

    if flag == 'analisis_basico':
        df = analisis_basico(df)
    elif flag == 'random_forest':
        df = analisis_random_forest(df)
    elif flag == 'lstm':
        df = analisis_lstm(df)
    elif flag == 'analisis_lstm_multiclase':
        df = analisis_lstm_multiclase(df)
    elif flag == 'analisis_xgb_multiclase':
        df = analisis_xgb_multiclase(df)

    if grafica:
        plot_signals(df, ticker)
    return df

def plot_signals(df, ticker):
    """
    Visualiza el precio de cierre, señales de compra/venta y retorno diario.
    Args:
        df (pd.DataFrame): DataFrame con los datos y señales.
        ticker (str): Símbolo del activo.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
    # Subplot 1: Precio de cierre + SMA + señales
    axs[0].plot(df['close'], label='Precio de Cierre', color='blue')
    axs[0].plot(df['SMA_5'], label='SMA 5', alpha=0.6)
    axs[0].plot(df['SMA_20'], label='SMA 20', alpha=0.6)
    buy_signals = df[df['resultado_final'] == 1]
    neutral_signals = df[df['resultado_final'] == 0]
    sell_signals = df[df['resultado_final'] == -1]
    axs[0].scatter(buy_signals.index, buy_signals['close'], label='Compra', marker='^', color='green', s=100)
    axs[0].scatter(neutral_signals.index, neutral_signals['close'], label='Nada', marker='o', color='blue', s=100)
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


def backtest_strategy(df):
    """
    Realiza un backtest de la estrategia de trading basada en señales generadas.
    Args:
        df (pd.DataFrame): DataFrame con señales y precios.
    Returns:
        tuple: (ROI estrategia, ROI buy&hold)
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


def estrategia(TICKERS, flags, init_day, param_grafica):
    """
    Ejecuta la estrategia para una lista de tickers y flags.
    Args:
        TICKERS (list): Lista de símbolos.
        flags (list): Lista de flags de estrategia.
    """
    arr_estrategia = []
    arr_mantener = []
    for flag in flags:
        cont_estrategia = 0
        cont_total = 0
        print(f"Evaluando estrategia con flag: {flag}")
        for i in TICKERS:
            print(f"Procesando {i}...")
            df = plot_stock_and_return(flag, i, init_day, grafica=param_grafica)
            final_value_roi, buy_hold_return = backtest_strategy(df)
            arr_estrategia.append(final_value_roi)
            arr_mantener.append(buy_hold_return)
            print(f"Resultado de estrategia: ROI = {final_value_roi:.2f}%)")
            print(f"Estrategia de mantener: ROI = {buy_hold_return:.2f}%")
            if (final_value_roi > buy_hold_return):
                cont_estrategia += 1
            cont_total += 1
        print(f"Total de acciones analizadas: {cont_total}")
        print(f"Acciones con mejor ROI que estrategia de mantener: {cont_estrategia}")
        print("--------------------------------------------------\n")
        print("Resultados finales:")
        print(f"ROI estrategia: {np.mean(arr_estrategia):.2f}%")
        print(f"ROI estrategia de mantener: {np.mean(arr_mantener):.2f}%")      
