import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

def analisis_basico(df):
    """
    Aplica señales técnicas básicas (SMA, RSI, MACD) y genera la columna resultado_final.
    """
    df['signal_sma'] = np.where(df['SMA_5'] > df['SMA_20'], 1, np.where(df['SMA_5'] < df['SMA_20'], -1, 0))
    df['signal_rsi'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    df['signal_macd'] = np.where(df['MACD'] > df['MACD_signal'], 1, np.where(df['MACD'] < df['MACD_signal'], -1, 0))
    df['signal_total'] = df['signal_sma'] + df['signal_rsi'] + df['signal_macd']
    df['resultado_final'] = np.where(df['signal_total'] >= 2, 1, np.where(df['signal_total'] <= -2, -1, 0))
    return df


def analisis_random_forest(df):
    """
    Aplica el análisis de Random Forest y genera la columna resultado_final.
    """
    threshold = 0.01
    df['target'] = np.where((df['close'].shift(-1) / df['close'] - 1) > threshold, 1,
                            np.where((df['close'].shift(-1) / df['close'] - 1) < -threshold, -1, 0))
    df.dropna(inplace=True)
    df = df[df['target'] != 0]
    features = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_signal', 'volatility_5', 'volatility_10', 'retorno']
    X = df[features]
    y = df['target'].replace({-1: 0})
    tscv = TimeSeriesSplit(n_splits=5)
    all_predictions = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_predictions[test_idx] = y_pred
    print(classification_report(y, all_predictions))
    df['prediction'] = all_predictions
    df['resultado_final'] = df['prediction'].replace({1: 1, 0: -1})
    return df


def analisis_lstm(df):
    """
    Aplica el análisis LSTM y genera la columna resultado_final.
    """
    threshold = 0.01
    df['target'] = np.where((df['close'].shift(-1) / df['close'] - 1) > threshold, 1,
                            np.where((df['close'].shift(-1) / df['close'] - 1) < -threshold, 0, 0))
    df.dropna(inplace=True)
    features = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_signal', 'volatility_5', 'volatility_10', 'retorno']
    df = df.dropna(subset=features + ['target'])
    X = df[features].values
    y = df['target'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    def create_sequences(X, y, window_size=10):
        X_seq, y_seq = [], []
        for i in range(window_size, len(X)):
            X_seq.append(X[i-window_size:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    def build_model(hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int("units", 32, 128, step=32),
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))
        model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log")),
            metrics=["accuracy"]
        )
        return model


    # ========= 4. Ejecutar el proceso completo =========

    features = ['SMA_5', 'SMA_10', 'SMA_20', 'RSI', 'MACD', 'MACD_signal', 'volatility_5', 'volatility_10', 'retorno']
    X = df[features].values
    y = df['target'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    window_size = 10
    global X_train  # Necesario para Keras Tuner
    X_seq, y_seq = create_sequences(X_scaled, y, window_size=window_size)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Búsqueda de hiperparámetros
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="tuner_dir",
        project_name=f"lstm"
    )

    tuner.search(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"✅ Mejores hiperparámetros: {best_hp.values}")

    # Entrenamiento final
    final_model = build_model(best_hp)
    final_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predicciones
    y_pred_proba = final_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\n📊 Clasificación en test:")
    print(classification_report(y_test, y_pred))

    # Agregar predicción al DataFrame original
    df_result = df.iloc[window_size + split:].copy()
    df_result['prediction'] = y_pred
    df_result['resultado_final'] = df_result['prediction'].replace({1: 1, 0: -1})

    return df_result

