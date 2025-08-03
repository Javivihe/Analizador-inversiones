import numpy as np
from transformar_datos import plot_stock_and_return
import pandas as pd


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


def estrategia(TICKERS, flags, init_day, param_grafica, comparar_features, n_days, threshold):
    """
    Ejecuta la estrategia para una lista de tickers y flags.
    Args:
        TICKERS (list): Lista de símbolos.
        flags (list): Lista de flags de estrategia.
    """
    arr_estrategia = []
    arr_mantener = []
    arr_ganadoras = []
    
    for flag in flags:
        cont_estrategia = 0
        cont_total = 0
        print(f"Evaluando estrategia con flag: {flag}")
        for i in TICKERS:
            print(f"Procesando {i}...")
            df = plot_stock_and_return(flag, i, init_day, n_days=n_days, threshold=threshold, grafica=param_grafica, comparar_features=comparar_features)
            final_value_roi, buy_hold_return = backtest_strategy(df)
            arr_estrategia.append(final_value_roi)
            arr_mantener.append(buy_hold_return)
            print(f"Resultado de estrategia: ROI = {final_value_roi:.2f}%)")
            print(f"Estrategia de mantener: ROI = {buy_hold_return:.2f}%")
            if (final_value_roi > buy_hold_return):
                cont_estrategia += 1
                arr_ganadoras.append(i)
            cont_total += 1
        print(f"Total de acciones analizadas: {cont_total}")
        print(f"Acciones con mejor ROI que estrategia de mantener: {cont_estrategia}")
        print("--------------------------------------------------\n")
        print("Resultados finales:")
        print(f"ROI estrategia: {np.mean(arr_estrategia):.2f}%")
        print(f"ROI estrategia de mantener: {np.mean(arr_mantener):.2f}%")      
        print(f"Acciones ganadoras: {arr_ganadoras}")
    return np.mean(arr_estrategia)
