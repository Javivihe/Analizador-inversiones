import yfinance as yf
import pandas as pd
import numpy as np
import requests


def get_earnings_calendar(symbol, start, end):
    """
    Obtiene fechas de earnings y sorpresas para un símbolo dado entre start y end (YYYY-MM-DD)
    """
    API_KEY = 'd1ug0q9r01qpci1cer40d1ug0q9r01qpci1cer4g'
    url = f'https://finnhub.io/api/v1/calendar/earnings?symbol={symbol}&from={start}&to={end}&token={API_KEY}'
    print(url)
    response = requests.get(url)
    data = response.json()
    
    # data['earningsCalendar'] es lista de dicts con fecha y surprise
    earnings = []
    for item in data.get('earningsCalendar', []):
        date = item.get('date')
        surprise = item.get('epsActual', 0) - item.get('epsEstimate', 0)
        earnings.append({'date': pd.to_datetime(date), 'eps_surprise': surprise})
    
    df_earnings = pd.DataFrame(earnings)
    return df_earnings

import finnhub
finnhub_client = finnhub.Client(api_key="d1ug0q9r01qpci1cer40d1ug0q9r01qpci1cer4g")

print(finnhub_client.company_earnings('TSLA', limit=5))

def añadir_mock_earnings(df, earnings_dict):
    """
    Añade columnas de earnings mockeadas:
    - earnings_soon (1 si dentro de los próximos 5 días)
    - eps_surprise (mock: positivo, negativo o neutro)
    
    earnings_dict: dict con fechas de earnings y sorpresas, e.g.
        {
            '2024-11-01': 0.05,
            '2025-02-05': -0.02,
            ...
        }
    """
    df = df.copy()
    df['earning_soon'] = 0
    df['eps_surprise'] = 0.0
    
    for date_str, surprise in earnings_dict.items():
        date = pd.to_datetime(date_str)
        mask = (df.index >= date - pd.Timedelta(days=5)) & (df.index <= date)
        df.loc[mask, 'earning_soon'] = 1
        df.loc[mask, 'eps_surprise'] = surprise

    return df
