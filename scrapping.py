import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import yfinance as yf

def get_ibex35_value():
    """Obtiene el valor actual del IBEX 35 desde su página específica en Yahoo Finanzas España."""
    url = 'https://es.finance.yahoo.com/quote/%5EIBEX'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # Buscar el valor principal del IBEX 35
    value_span = soup.find('fin-streamer', {'data-field': 'regularMarketPrice'})
    print(soup.find('fin-streamer'))
    if value_span:
        return value_span.text
    return 'No se pudo obtener el valor del IBEX 35'

def get_ibex35_value_selenium():
    """Obtiene el valor actual del IBEX 35 usando Selenium desde Yahoo Finanzas España."""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Ejecuta el navegador en modo headless (sin ventana)
    options.add_argument('--ignore-certificate-errors')  # Ignora errores de certificado SSL
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get('https://es.finance.yahoo.com/quote/%5EIBEX')
    time.sleep(3)  # Espera a que cargue el contenido dinámico
    try:
        value_elem = driver.find_element(By.XPATH, '//fin-streamer[@data-field="regularMarketPrice"]')
        value = value_elem.text
    except Exception as e:
        value = f'No se pudo obtener el valor del IBEX 35: {e}'
    driver.quit()
    return value

def debug_ibex35_scraping():
    """Imprime todos los valores de fin-streamer para depuración."""
    url = 'https://es.finance.yahoo.com/quote/%5EIBEX'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    fin_streamers = soup.find_all('fin-streamer')
    print(f"Encontrados {len(fin_streamers)} elementos <fin-streamer>:")
    for i, elem in enumerate(fin_streamers):
        print(f"{i}: {elem.attrs} -> {elem.text}")

def get_indices_values():
    indices = {
        'IBEX 35': '^IBEX',
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'Nasdaq': '^IXIC',
        'DAX': '^GDAXI',
        'CAC 40': '^FCHI',
        'FTSE 100': '^FTSE',
        'Nikkei 225': '^N225',
        'Hang Seng': '^HSI',
        'Euro Stoxx 50': '^STOXX50E',
    }
    results = {}
    for name, ticker in indices.items():
        try:
            data = yf.Ticker(ticker).history(period='1d')
            if not data.empty:
                results[name] = data['Close'].iloc[-1]
            else:
                results[name] = 'No data'
        except Exception as e:
            results[name] = f'Error: {e}'
    return results

def get_wikipedia_indices():
    """Obtiene una lista de índices bursátiles desde Wikipedia."""
    url = 'https://es.wikipedia.org/wiki/Anexo:Índices_bursátiles'
    response = requests.get(url)
    if response.status_code != 200:
        return f'Error al acceder a Wikipedia: {response.status_code}'
    soup = BeautifulSoup(response.text, 'html.parser')
    indices = []
    # Buscar todas las tablas de índices bursátiles
    for table in soup.find_all('table', {'class': 'wikitable'}):
        for row in table.find_all('tr')[1:]:  # Saltar cabecera
            cols = row.find_all('td')
            if len(cols) >= 2:
                nombre = cols[0].get_text(strip=True)
                pais = cols[1].get_text(strip=True)
                print(nombre)
                indices.append({'nombre': nombre, 'pais': pais})
    return indices

def get_all_etfs_yahoo():
    """Scrapea la tabla principal de ETF de Yahoo Finance usando Selenium (sin headless)."""
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Quitado para ver el navegador
    options.add_argument('--lang=en-US')
    options.add_argument('--ignore-certificate-errors')  # Ignora errores de certificado SSL
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get('https://finance.yahoo.com/etfs')

    # Hacer scroll para forzar la carga de la tabla y el botón de cookies
    driver.execute_script("window.scrollTo(0, 500)")
    time.sleep(1)

    # Si hay un botón que dice "Ir al final" (o "Go to end"), hacer clic en él
    try:
        go_to_end_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(translate(., "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ", "abcdefghijklmnopqrstuvwxyzáéíóúüñ"), "ir al final") or contains(translate(., "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "go to end")]'))
        )
        go_to_end_btn.click()
        time.sleep(1)
    except Exception:
        pass  # Si no aparece, continuar

    # Intentar aceptar cookies si aparece el popup (haciendo scroll dentro del popup)
    try:
        popup = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, '//div[contains(@class, "consent") or contains(@class, "dialog") or contains(@role, "dialog")]'))
        )
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", popup)
        time.sleep(0.5)
    except Exception:
        pass  # Si no aparece, continuar

    # Intentar aceptar cookies si aparece el botón (más robusto)
    try:
        consent_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, '//button[contains(translate(., "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "accept")]'))
        )
        consent_btn.click()
        time.sleep(1)
    except Exception:
        try:
            consent_btn = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, '//button[contains(translate(., "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ", "abcdefghijklmnopqrstuvwxyzáéíóúüñ"), "aceptar")]'))
            )
            consent_btn.click()
            time.sleep(1)
        except Exception:
            pass  # Si no aparece, continuar

    # Esperar a que la tabla esté presente
    try:
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, '//table//tbody/tr'))
        )
    except Exception as e:
        print(f'No se encontró la tabla de ETF: {e}')
        driver.quit()
        return []

    etfs = []
    try:
        rows = driver.find_elements(By.XPATH, '//table//tbody/tr')
        for row in rows:
            try:
                cols = row.find_elements(By.TAG_NAME, 'td')
                if len(cols) >= 2:
                    ticker = cols[0].text
                    name = cols[1].text
                    etfs.append({'ticker': ticker, 'name': name})
            except Exception:
                continue
    except Exception as e:
        print(f'Error al scrapear ETF: {e}')
    driver.quit()
    return etfs

# Ejemplo de uso:
if __name__ == "__main__":

    etfs = get_all_etfs_yahoo()
    for etf in etfs:  # Muestra los primeros 20 ETF
        print(f"{etf['ticker']}: {etf['name']}")
