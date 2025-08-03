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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt
from comparar_features import importancia_features_lstm
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


def analisis_random_forest(df, features):
    """
    Aplica el análisis de Random Forest y genera la columna resultado_final.
    """

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

def analisis_lstm_multiclase(df, features, comparar_features):


    # === 4. Escalado y creación de secuencias ===
    X = df[features].values
    y_raw = df['target'].values.reshape(-1, 1)
    print(df[features])
    scaler = MinMaxScaler()

    print(X)
    X_scaled = scaler.fit_transform(X)
    print(X_scaled)
    encoder = OneHotEncoder(sparse=False, categories='auto')
    y_encoded = encoder.fit_transform(y_raw)

    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(window_size, len(X)):
            X_seq.append(X[i - window_size:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    window_size = 20
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

    if (comparar_features):
        print("\n📊 Comparando Features...:")
        resultados = importancia_features_lstm(
            df=df,
            features=features,
            window_size=20,
            metric='accuracy',
            build_model_fn=build_model,
            hp=best_hp,
            verbose=1
        )

        print(resultados)

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


# def analisis_xgb_multiclase_old(df):
#     """
#     Entrena un XGBoostClassifier para predecir señales -1/0/1 a partir de indicadores técnicos
#     y devuelve un DataFrame con las predicciones sobre la partición de test.

#     Parámetros
#     ----------
#     df : pd.DataFrame
#         Debe contener al menos ['close', 'high', 'low', 'volume'] y los indicadores técnicos
#         mencionados en `features`. Si no los tienes, añádelos antes de llamar a la función.

#     Devuelve
#     --------
#     df_result : pd.DataFrame
#         Filas del conjunto de test con la predicción (`prediction`) y la columna `resultado_final`
#         (‑1, 0 o 1).
#     """

#     df, features = transformar_datos(df)


#     # corr_matrix = df[features].corr().abs()

#     # plt.figure(figsize=(12,10))
#     # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
#     # plt.title('Matriz de correlación entre features')
#     # plt.show()

#     # 2. Eliminar features muy correlacionadas (> 0.9)
#     # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#     # to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
#     # print(f"Features a eliminar por alta correlación: {to_drop}")

#     # features = [f for f in features if f not in to_drop]
#     # ------------------------------------------------------------------
#     # 4. Preparar X, y y división temporal 80 / 20
#     # ------------------------------------------------------------------
#     X = df[features].values

#     # Mapear (-1,0,1) → (0,1,2) porque XGBoost requiere enteros consecutivos desde 0
#     label_map = {-1: 0, 0: 1, 1: 2}
#     y = df['target'].map(label_map).values

#     split_idx = int(len(df) * 0.8)
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]

#     # ------------------------------------------------------------------
#     # 5. Compensar el desbalance de clases con pesos por muestra
#     # ------------------------------------------------------------------
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     cw_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
#     sample_weight_train = np.vectorize(cw_dict.get)(y_train)

#     # ------------------------------------------------------------------
#     # 6. Definir y entrenar el modelo
#     #    (hiperparámetros razonables; ajusta si quieres con Grid/Random Search)
#     # ------------------------------------------------------------------
#     model = XGBClassifier(
#         n_estimators=200,
#         learning_rate=0.1,
#         max_depth=4,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='multi:softprob',   # salida de probabilidades
#         num_class=3,
#         eval_metric='mlogloss',
#         random_state=42,
#         verbosity=0
#     )

#     model.fit(X_train, y_train, sample_weight=sample_weight_train)

#     # ------------------------------------------------------------------
#     # 7. Evaluación clásica + reporte
#     # ------------------------------------------------------------------
#     y_pred = model.predict(X_test)
#     inv_label_map = {v: k for k, v in label_map.items()}

#     print("\n📊 Clasificación en test:")
#     print(classification_report(y_test,
#                                 y_pred,
#                                 target_names=['Vender (-1)', 'Nada (0)', 'Comprar (1)']))

#     # ------------------------------------------------------------------
#     # 8. Reconstruir DataFrame con resultados
#     # ------------------------------------------------------------------
#     df_result = df.iloc[split_idx:].copy()
#     df_result['prediction']     = [inv_label_map[i] for i in y_pred]
#     df_result['resultado_final'] = df_result['prediction']   # por si acaso

#     return df_result

# def analisis_xgb_multiclase(df):
#     """
#     Entrena un XGBoostClassifier para predecir señales -1/0/1 a partir de indicadores técnicos
#     y devuelve un DataFrame con las predicciones sobre la partición de test.
#     Además entrena un LogisticRegression con regularización L1 para analizar importancia de features.

#     Parámetros
#     ----------
#     df : pd.DataFrame
#         Debe contener al menos ['close', 'high', 'low', 'volume'] y los indicadores técnicos
#         mencionados en `features`.

#     Devuelve
#     --------
#     df_result : pd.DataFrame
#         Filas del conjunto de test con la predicción (`prediction`) y la columna `resultado_final` (-1, 0 o 1).
#     """
    
#     df, features = transformar_datos(df)


#     # Preparamos X e y para XGBoost
#     X = df[features].values
#     label_map = {-1: 0, 0: 1, 1: 2}  # Mapear etiquetas
#     y = df['target'].map(label_map).values

#     split_idx = int(len(df)*0.8)
#     X_train, X_test = X[:split_idx], X[split_idx:]
#     y_train, y_test = y[:split_idx], y[split_idx:]

#     # Pesos para balancear clases en XGBoost
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     cw_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
#     sample_weight_train = np.vectorize(cw_dict.get)(y_train)

#     # --- Modelo XGBoost ---
#     model = XGBClassifier(
#         n_estimators=200,
#         learning_rate=0.1,
#         max_depth=4,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         objective='multi:softprob',
#         num_class=3,
#         eval_metric='mlogloss',
#         random_state=42,
#         verbosity=0
#     )
#     model.fit(X_train, y_train, sample_weight=sample_weight_train)
#     y_pred = model.predict(X_test)

#     inv_label_map = {v: k for k, v in label_map.items()}

#     print("\n📊 Clasificación XGBoost en test:")
#     print(classification_report(y_test,
#                                 y_pred,
#                                 target_names=['Vender (-1)', 'Nada (0)', 'Comprar (1)']))

#     # --- Modelo Logistic Regression con L1 (regularización) para análisis de features ---
#     # Escalado necesario para modelos lineales
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X[:split_idx])
#     X_test_scaled = scaler.transform(X[split_idx:])

#     lr = LogisticRegression(
#         penalty='l1',
#         solver='saga',
#         max_iter=10000,
#         random_state=42,
#         multi_class='multinomial',
#         C=1.0
#     )
#     lr.fit(X_train_scaled, y_train)
#     y_pred_lr = lr.predict(X_test_scaled)

#     print("\n📊 Clasificación Logistic Regression L1 en test:")
#     print(classification_report(y_test,
#                                 y_pred_lr,
#                                 target_names=['Vender (-1)', 'Nada (0)', 'Comprar (1)']))

#     # Importancia según coeficientes (suma absoluta en multiclase)
#     import numpy as np
#     coef_importance = np.sum(np.abs(lr.coef_), axis=0)
#     features_importance = sorted(zip(features, coef_importance), key=lambda x: x[1], reverse=True)
#     print("\n🎯 Importancia de features según Logistic Regression L1:")
#     for f, imp in features_importance:
#         print(f"{f}: {imp:.4f}")

#     # Devolver resultados XGBoost para test
#     df_result = df.iloc[split_idx:].copy()
#     df_result['prediction'] = [inv_label_map[i] for i in y_pred]
#     df_result['resultado_final'] = df_result['prediction']

#     return df_result
