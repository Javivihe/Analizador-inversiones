import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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

from funciones_utiles import plot_signals

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


def transformar_datos(df, n_days=3, threshold=0.01):

    # === 1. Crear target multiclase: -1 (vender), 0 (nada), 1 (comprar) ===
    future_return = df['close'].shift(-n_days) / df['close'] - 1
    print(future_return)
    df['target'] = np.where(
        future_return > threshold, 1,
        np.where(future_return < -threshold, -1, 0)
    )
    df.dropna(inplace=True)

    # === 2. Agregar nuevas variables ===
    df['daily_return'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['price_diff'] = df['high'] - df['low']
    df['rolling_max'] = df['close'].rolling(5).max()
    df['rolling_min'] = df['close'].rolling(5).min()

    # === 3. Lista de features ===

    features = [
        # Medias Móviles Simples (SMA)
        'SMA_5', 'SMA_10', 'SMA_20',
        
        # Medias Móviles Exponenciales (EMA)
        'EMA_5', 'EMA_10', 'EMA_20',
        
        # Indicadores de momentum y tendencia
        'RSI',
        'MACD', 'MACD_signal',
        'momentum_5', 'momentum_10',
        'stoch_k', 'stoch_d',
        'ADX_14',
        'CCI_20',
        
        # Indicadores de volatilidad
        'volatility_5', 'volatility_10',
        'ATR_14',
        'bb_bbm', 'bb_bbh', 'bb_bbl', 'bb_bandwidth',
        
        # Retornos y variaciones
        'retorno',
        'daily_return',
        'price_diff',
        'gap',
        
        # Volumen
        'volume',
        'volume_change',
        'OBV',
        
        # Precios extremos y rangos
        'rolling_max_10', 'rolling_min_10',
        
        # VIX como indicador externo
        'vix_close', 'vix_change_1d',
        
        # Características temporales
        'day_of_week', 'month', 'quarter',
        
        # Otros
        'up_down'
    ]

    df = df.dropna(subset=features + ['target'])  # fuera nulos
