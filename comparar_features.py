import numpy as np
import pandas as pd

def importancia_features_lstm(df, features, window_size=20, metric='accuracy', 
                              build_model_fn=None, hp=None, verbose=1):
    """
    Evalúa la importancia de cada feature en un modelo LSTM multiclase mediante permutación.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las features y la columna 'target'.
    features : list
        Lista de nombres de las columnas a evaluar.
    window_size : int
        Tamaño de la ventana temporal para las secuencias.
    metric : str
        Métrica de evaluación: 'accuracy' o 'loss'.
    build_model_fn : function
        Función que construye el modelo. Debe aceptar un objeto HyperParameters.
    hp : keras_tuner.HyperParameters
        Hiperparámetros para construir el modelo. Se puede pasar del tuner.
    verbose : int
        Nivel de verbosidad.

    Retorna:
    --------
    DataFrame con la importancia de cada feature.
    """
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping
    import copy

    # Preprocesamiento base
    X = df[features].values
    y_raw = df['target'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y_raw)

    def create_sequences(X, y, window_size):
        X_seq, y_seq = [], []
        for i in range(window_size, len(X)):
            X_seq.append(X[i - window_size:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    X_seq, y_seq = create_sequences(X_scaled, y_encoded, window_size=window_size)
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # Entrenamiento modelo base
    model = build_model_fn(hp)
    y_train_labels = np.argmax(y_train, axis=1)
    weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weights = dict(zip(np.unique(y_train_labels), weights))

    model.fit(X_train, y_train, epochs=30, batch_size=32,
              validation_data=(X_test, y_test),
              class_weight=class_weights,
              callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
              verbose=0)

    y_pred = model.predict(X_test, verbose=0)
    if metric == 'accuracy':
        baseline_score = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    elif metric == 'loss':
        baseline_score = log_loss(y_test, y_pred)
    else:
        raise ValueError("Métrica no soportada")

    if verbose:
        print(f"🔹 Score base ({metric}): {baseline_score:.4f}")

    # Evaluar permutación por feature
    importancias = {}
    for i, col in enumerate(features):
        if verbose:
            print(f"🔄 Permutando feature: {col}")
        X_perturbed = copy.deepcopy(X_scaled)
        np.random.shuffle(X_perturbed[:, i])  # Permutar columna

        X_seq_perm, y_seq_perm = create_sequences(X_perturbed, y_encoded, window_size=window_size)
        X_train_p, X_test_p = X_seq_perm[:split], X_seq_perm[split:]
        y_train_p, y_test_p = y_seq_perm[:split], y_seq_perm[split:]

        model_p = build_model_fn(hp)
        y_train_labels_p = np.argmax(y_train_p, axis=1)
        weights_p = compute_class_weight('balanced', classes=np.unique(y_train_labels_p), y=y_train_labels_p)
        class_weights_p = dict(zip(np.unique(y_train_labels_p), weights_p))

        model_p.fit(X_train_p, y_train_p, epochs=30, batch_size=32,
                    validation_data=(X_test_p, y_test_p),
                    class_weight=class_weights_p,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
                    verbose=0)

        y_pred_p = model_p.predict(X_test_p, verbose=0)
        if metric == 'accuracy':
            score_p = accuracy_score(np.argmax(y_test_p, axis=1), np.argmax(y_pred_p, axis=1))
            delta = baseline_score - score_p
        else:  # loss
            score_p = log_loss(y_test_p, y_pred_p)
            delta = score_p - baseline_score  # al revés porque menor loss es mejor

        importancias[col] = delta

    importancia_df = pd.DataFrame({
        'feature': list(importancias.keys()),
        'importance_drop': list(importancias.values())
    }).sort_values(by='importance_drop', ascending=False).reset_index(drop=True)

    return importancia_df
