from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import copy

def comparar_importancia_features_oof(df, features, target_col='target', n_splits=5, random_state=42):
    """
    Evalúa la importancia de cada feature usando permutación + predicción OOF.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con las features y columna target.
    features : list
        Lista de nombres de columnas a evaluar.
    target_col : str
        Nombre de la columna objetivo.
    n_splits : int
        Número de particiones en KFold.
    
    Retorna:
    --------
    pd.DataFrame con la importancia estimada de cada feature.
    """

    X = df[features].values
    y = df[target_col].values
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # 1. Predicciones OOF originales (baseline)
    oof_preds = np.zeros(len(y))
    for train_idx, val_idx in kf.split(X):
        model = RandomForestClassifier(random_state=random_state)
        model.fit(X[train_idx], y[train_idx])
        oof_preds[val_idx] = model.predict(X[val_idx])
    baseline_score = accuracy_score(y, oof_preds)

    importancias = {}
    for i, col in enumerate(features):
        X_permuted = copy.deepcopy(X)
        np.random.shuffle(X_permuted[:, i])  # Permutar solo esta feature

        oof_permuted = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X_permuted):
            model = RandomForestClassifier(random_state=random_state)
            model.fit(X_permuted[train_idx], y[train_idx])
            oof_permuted[val_idx] = model.predict(X_permuted[val_idx])

        permuted_score = accuracy_score(y, oof_permuted)
        delta = baseline_score - permuted_score
        importancias[col] = delta

    # Ordenar resultado
    importancia_df = pd.DataFrame({
        'feature': list(importancias.keys()),
        'importance_drop': list(importancias.values())
    }).sort_values(by='importance_drop', ascending=False).reset_index(drop=True)

    return importancia_df
