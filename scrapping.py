import requests
from bs4 import BeautifulSoup

def get_ibex35_value():
    """Obtiene el valor actual del IBEX 35 desde su página específica en Yahoo Finanzas España."""
    url = 'https://es.finance.yahoo.com/quote/%5EIBEX'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Buscar el valor principal del IBEX 35
    value_span = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
    if value_span:
        return value_span.text
    return 'No se pudo obtener el valor del IBEX 35'

# Ejemplo de uso:
if __name__ == "__main__":
    print('Valor IBEX 35:', get_ibex35_value())
