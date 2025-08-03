import matplotlib.pyplot as plt



def plot_signals(df, ticker):
    """
    Visualiza el precio de cierre, señales de compra/venta y retorno diario.
    Args:
        df (pd.DataFrame): DataFrame con los datos y señales.
        ticker (str): Símbolo del activo.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True)
    # Subplot 1: Precio de cierre + SMA + señales
    axs[0].plot(df['close'], label='Precio de Cierre', color='blue')
    axs[0].plot(df['SMA_5'], label='SMA 5', alpha=0.6)
    axs[0].plot(df['SMA_20'], label='SMA 20', alpha=0.6)
    buy_signals = df[df['resultado_final'] == 1]
    neutral_signals = df[df['resultado_final'] == 0]
    sell_signals = df[df['resultado_final'] == -1]
    axs[0].scatter(buy_signals.index, buy_signals['close'], label='Compra', marker='^', color='green', s=100)
    axs[0].scatter(neutral_signals.index, neutral_signals['close'], label='Nada', marker='o', color='blue', s=100)
    axs[0].scatter(sell_signals.index, sell_signals['close'], label='Venta', marker='v', color='red', s=100)
    axs[0].set_title("Precio de cierre, SMA y señales de compra/venta")
    axs[0].legend()
    axs[0].grid()
    # Subplot 2: Evolución del valor de cierre con colores
    fechas = df.index.tolist()
    valores = df['close'].tolist()
    for i in range(1, len(valores)):
        color = 'green' if valores[i] >= valores[i-1] else 'red'
        axs[1].plot(fechas[i-1:i+1], valores[i-1:i+1], color=color, marker='o')
    axs[1].set_title(f'Evolución del valor de cierre: {ticker}')
    axs[1].set_ylabel('Valor de cierre')
    axs[1].legend(['Sube/Baja'], loc='upper left')
    axs[1].grid(True)
    # Subplot 3: Retorno diario
    axs[2].plot(fechas, df['retorno'], color='purple', linestyle='-', marker='o', label='Retorno diario')
    axs[2].set_title('Evolución del retorno diario')
    axs[2].set_xlabel('Fecha')
    axs[2].set_ylabel('Retorno diario')
    axs[2].legend()
    axs[2].grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()