import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_stock_value_on_date(ticker_key, date=None):
    """
    Fetch the value of a given stock ticker based on a key from the dictionary on a specific date using yfinance.
    If no date is provided, use yesterday if it's a weekday or the previous Friday if it's a weekend.
    """

    ticker_dict = {
        'Etf.vang S&p500 Usdd': 'VUSA.AS',
        'Vanguard S&P 500 (Dist)': 'VUSA.AS',
        'Etf Ishares Stoxx Europe 600': 'EXSA.DE',
        'iShares STOXX Europe 600 DE (Dist)': 'EXSA.DE',
        'Vanguard FTSE All-World (Acc)': 'VWCE.DE',
        'Vanguard FTSE Emerging Markets (Acc)': 'VFEA.DE',
        'Vanguard FTSE Emerging Markets (Acc)': 'VFEA.DE',
        'iShares China Large Cap (Acc)': 'FXAC.AS'
    }

    ticker = ticker_dict.get(ticker_key)
    if not ticker:
        return {"error": f"No ticker found for key: {ticker_key}"}

    # Determine the start_date
    if date:
        start_date = pd.to_datetime(date)
    else:
        today = pd.Timestamp.now()
        if today.weekday() == 0:  # Monday yesterday is Sunday 
            start_date = today - pd.Timedelta(days=3)  # Previous Friday
        if today.weekday() == 6:  # Sunday, yesterday is Saturday 
            start_date = today - pd.Timedelta(days=2)  # Previous Friday
        else:
            start_date = today - pd.Timedelta(days=1)  # Yesterday

    # Adjust end_date based on whether the start_date is a weekend
    if start_date.weekday() == 5:  # Saturday
        end_date = (start_date + pd.Timedelta(days=3)).strftime('%Y-%m-%d')  # Move to Tuesday
    elif start_date.weekday() == 6:  # Sunday
        end_date = (start_date + pd.Timedelta(days=2)).strftime('%Y-%m-%d')  # Move to Tuesday
    else:
        end_date = (start_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')  # Next day

    stock = yf.Ticker(ticker)
    stock_data = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date, interval='1d')
    
    if not stock_data.empty:
        return {"id": ticker, "value": stock_data['Close'].iloc[-1], "date": stock_data.index[-1].strftime('%Y-%m-%d')}
    else:
        return {"id": ticker, "value": f"No data available for {start_date.strftime('%Y-%m-%d')}", "date": start_date.strftime('%Y-%m-%d')}



# Ejemplo:   bucle de días desde el 1 de enero de 2024 hasta hoy
start_loop = pd.Timestamp('2024-01-01')
end_loop = pd.Timestamp.now()
results = []
for single_date in pd.date_range(start=start_loop, end=end_loop, freq='W'):
    result = get_stock_value_on_date('Etf.vang S&p500 Usdd', date=single_date.strftime('%Y-%m-%d'))
    results.append(result)
    # print(result)  # Puedes quitar este print si solo quieres guardar los resultados

# Graficar los resultados con colores según sube o baja
fechas = [r['date'] for r in results if isinstance(r['value'], (int, float, np.number))]
valores = [r['value'] for r in results if isinstance(r['value'], (int, float, np.number))]

plt.figure(figsize=(12, 6))
for i in range(1, len(valores)):
    color = 'green' if valores[i] >= valores[i-1] else 'red'
    plt.plot(fechas[i-1:i+1], valores[i-1:i+1], color=color, marker='o')
plt.xlabel('Fecha')
plt.ylabel('Valor de cierre')
plt.title('Evolución del valor de cierre')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Si quieres trabajar con los resultados, ahora están en la lista 'results'
# print(results)  # Descomenta para ver todos los resultados juntos