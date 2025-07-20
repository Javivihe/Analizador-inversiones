import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ta

from funciones_analisis import analisis_basico, analisis_random_forest, analisis_lstm, analisis_lstm_multiclase, analisis_xgb_multiclase

import yfinance as yf
import pandas as pd
import ta  # Asegúrate de tener la librería instalada: pip install ta

def plot_stock_and_return(flag, ticker, start_date, end_date=None, grafica=False):
    if end_date is None:
        end_date = pd.Timestamp.now()
    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date, end=end_date, interval='1d')
    stock_data = stock_data.dropna(subset=['Close', 'Open'])

    df = pd.DataFrame({
        'stock': ticker,
        'open': stock_data['Open'],
        'close': stock_data['Close'],
        'retorno': (stock_data['Close'] - stock_data['Open']) / stock_data['Open'],
        'high': stock_data['High'],
        'low': stock_data['Low'],
        'volume': stock_data['Volume'],
        # Medias Móviles Simples (SMA)
        'SMA_5':  stock_data['Close'].rolling(window=5).mean(),
        'SMA_10':  stock_data['Close'].rolling(window=10).mean(),
        'SMA_20':  stock_data['Close'].rolling(window=20).mean(),
        # Volatilidad
        'volatility_5': stock_data['Close'].rolling(window=5).std(),
        'volatility_10': stock_data['Close'].rolling(window=10).std(),
        # Nuevas features básicas:
        'daily_return': stock_data['Close'].pct_change(),
        'volume_change': stock_data['Volume'].pct_change(),
        'price_diff': stock_data['Close'] - stock_data['Open'],
        'rolling_max_10': stock_data['Close'].rolling(window=10).max(),
        'rolling_min_10': stock_data['Close'].rolling(window=10).min(),
        # Gap: apertura menos cierre anterior
        'gap': stock_data['Open'] - stock_data['Close'].shift(1),
        # Consecutivos días al alza o baja (1/-1, 0 neutro)
        'up_down': (stock_data['Close'] - stock_data['Close'].shift(1)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    }, index=stock_data.index.strftime('%Y-%m-%d').str[:10])

    # Indicadores técnicos con 'ta' (más avanzados)

    # EMA
    df['EMA_5'] = ta.trend.EMAIndicator(close=stock_data['Close'], window=5).ema_indicator()
    df['EMA_10'] = ta.trend.EMAIndicator(close=stock_data['Close'], window=10).ema_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(close=stock_data['Close'], window=20).ema_indicator()

    # Bollinger Bands (20 periodos, 2 desviaciones)
    bb_indicator = ta.volatility.BollingerBands(close=stock_data['Close'], window=20, window_dev=2)
    df['bb_bbm'] = bb_indicator.bollinger_mavg()
    df['bb_bbh'] = bb_indicator.bollinger_hband()
    df['bb_bbl'] = bb_indicator.bollinger_lband()
    df['bb_bandwidth'] = df['bb_bbh'] - df['bb_bbl']

    # Momentum (Rate of Change 5 y 10 días)
    df['momentum_5'] = ta.momentum.ROCIndicator(close=stock_data['Close'], window=5).roc()
    df['momentum_10'] = ta.momentum.ROCIndicator(close=stock_data['Close'], window=10).roc()

    # Stochastic Oscillator (%K y %D)
    stoch = ta.momentum.StochasticOscillator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # ATR - Average True Range (volatilidad intradía)
    df['ATR_14'] = ta.volatility.AverageTrueRange(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], window=14).average_true_range()

    # ADX - Fuerza de tendencia
    df['ADX_14'] = ta.trend.ADXIndicator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], window=14).adx()

    # CCI - Commodity Channel Index
    df['CCI_20'] = ta.trend.CCIIndicator(high=stock_data['High'], low=stock_data['Low'], close=stock_data['Close'], window=20).cci()

    # OBV - On Balance Volume
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=stock_data['Close'], volume=stock_data['Volume']).on_balance_volume()

    # VIX
    vix = yf.Ticker("^VIX").history(start=start_date, end=end_date, interval='1d')
    vix = vix[['Close']].rename(columns={'Close': 'vix_close'})
    vix['vix_change_1d'] = vix['vix_close'].pct_change()
    vix.index = vix.index.strftime('%Y-%m-%d').str[:10]
    df = df.join(vix, how='left')
    df[['vix_close', 'vix_change_1d']] = df[['vix_close', 'vix_change_1d']].fillna(method='ffill')

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=stock_data['Close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close=stock_data['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()

    # Time features
    df['day_of_week'] = pd.to_datetime(df.index).dayofweek  # 0=lunes ... 6=domingo
    df['month'] = pd.to_datetime(df.index).month
    df['quarter'] = pd.to_datetime(df.index).quarter

    # Limpiar nans (por rolling windows)
    df.dropna(inplace=True)
    # Lógica que tenías para procesar segun flag (igual la mantengo)
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
