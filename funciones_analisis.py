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
from tensorflow.keras.callbacks import EarlyStopping

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
    threshold = 0.001
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
    Aplica el análisis LSTM mejorado con horizonte de predicción ajustado y nuevas features.
    """

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import classification_report
    from sklearn.utils.class_weight import compute_class_weight
    import keras_tuner as kt
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping

    # === 1. Crear target: predicción de subida en 3 días con 0.5% de umbral ===
    n_days = 3
    threshold = 0.003
    future_return = df['close'].shift(-n_days) / df['close'] - 1

    # === Crear máscara: solo subidas o bajadas relevantes (filtramos neutros) ===
    mask = (future_return > threshold) | (future_return < -threshold)
    df = df[mask].copy()

    # === Crear target binario: 1 = comprar, 0 = vender ===
    df['target'] = np.where(future_return > threshold, 1, 0)

    # === 2. Agregar nuevas variables ===
    df['daily_return'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['price_diff'] = df['high'] - df['low']
    df['rolling_max'] = df['close'].rolling(5).max()
    df['rolling_min'] = df['close'].rolling(5).min()

    # === 3. Lista de features ===
    features = [
        'SMA_5', 'SMA_10', 'SMA_20',
        'RSI', 'MACD', 'MACD_signal',
        'volatility_5', 'volatility_10',
        'retorno',
        'daily_return', 'volume_change',
        'price_diff', 'rolling_max', 'rolling_min'
    ]
    df = df.dropna(subset=features + ['target'])

    # === 4. Preprocesamiento ===
    X = df[features].values
    y = df['target'].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(window_size, len(X)):
            X_seq.append(X[i - window_size:i])
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

    # === 5. Crear secuencias ===
    window_size = 5
    global X_train  # necesario para tuner
    X_seq, y_seq = create_sequences(X_scaled, y, window_size=window_size)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # === 6. Ver distribución ===
    unique, counts = np.unique(y_train, return_counts=True)
    print("📊 Distribución de clases en entrenamiento:")
    print(dict(zip(unique, counts)))

    # === 7. Tuning de hiperparámetros ===
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="tuner_dir",
        project_name="lstm_opt"
    )

    tuner.search(X_train, y_train, epochs=30, batch_size=32,
                 validation_data=(X_test, y_test),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
                 verbose=0)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"✅ Mejores hiperparámetros: {best_hp.values}")

    # === 8. Pesos de clase ===
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), weights))
    print(f"📊 Pesos aplicados: {class_weights}")

    # === 9. Entrenamiento final ===
    final_model = build_model(best_hp)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    final_model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights,
                    callbacks=[early_stop],
                    verbose=0)

    # === 10. Predicciones ===
    y_pred_proba = final_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()

    print("\n📊 Clasificación en test:")
    print(classification_report(y_test, y_pred))

    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print(f"🔍 Distribución de predicciones: {dict(zip(unique_pred, counts_pred))}")

    # === 11. Resultado final ===
    df_result = df.iloc[window_size + split:].copy()
    df_result['prediction'] = y_pred
    df_result['resultado_final'] = df_result['prediction'].replace({1: 1, 0: -1})

    return df_result

def analisis_lstm_multiclase(df):
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

    # === 1. Crear target multiclase: -1 (vender), 0 (nada), 1 (comprar) ===
    n_days = 3
    threshold = 0.003
    future_return = df['close'].shift(-n_days) / df['close'] - 1

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
        'SMA_5', 'SMA_10', 'SMA_20',
        'RSI', 'MACD', 'MACD_signal',
        'volatility_5', 'volatility_10',
        'retorno', 'daily_return', 'volume_change',
        'price_diff', 'rolling_max', 'rolling_min'
    ]
    df = df.dropna(subset=features + ['target'])

    # === 4. Escalado y creación de secuencias ===
    X = df[features].values
    y_raw = df['target'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_encoded = encoder.fit_transform(y_raw)

    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(window_size, len(X)):
            X_seq.append(X[i - window_size:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    window_size = 5
    global X_train
    X_seq, y_seq = create_sequences(X_scaled, y_encoded, window_size=window_size)

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    def build_model(hp):
        model = Sequential()
        model.add(LSTM(
            units=hp.Int("units", 32, 128, step=32),
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))
        model.add(Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
        model.add(Dense(3, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=hp.Float("lr", 1e-4, 1e-2, sampling="log")),
            metrics=["accuracy"]
        )
        return model

    # === 5. Tuning ===
    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory="tuner_dir",
        project_name="lstm_multiclass"
    )

    tuner.search(X_train, y_train, epochs=30, batch_size=32,
                 validation_data=(X_test, y_test),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=5)],
                 verbose=0)

    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"✅ Mejores hiperparámetros: {best_hp.values}")

    # === 6. Reentrenamiento con pesos ===
    y_train_labels = np.argmax(y_train, axis=1)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weights = dict(zip(np.unique(y_train_labels), weights))
    print(f"📊 Pesos aplicados: {class_weights}")

    final_model = build_model(best_hp)
    final_model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test),
                    class_weight=class_weights,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=0)

    # === 7. Evaluación y predicción ===
    y_pred_probs = final_model.predict(X_test)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_true_labels = np.argmax(y_test, axis=1)

    print("\n📊 Clasificación en test:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=['Vender (-1)', 'Nada (0)', 'Comprar (1)']))

    # === 8. Reconstrucción del DataFrame ===
    clases = encoder.categories_[0]
    df_result = df.iloc[window_size + split:].copy()
    df_result['prediction'] = clases[y_pred_labels]
    df_result['resultado_final'] = df_result['prediction']  # ya es -1, 0 o 1

    return df_result
