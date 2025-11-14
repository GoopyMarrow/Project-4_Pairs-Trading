import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')


def plot_pair_prices(data: pd.DataFrame, ticker1: str, ticker2: str):
    plt.figure(figsize=(14, 7))

    data[ticker1].plot(label=ticker1, color='blue', alpha=0.9)
    data[ticker2].plot(label=ticker2, color='orange', alpha=0.9)

    plt.title(f'Historical Prices ({ticker1} & {ticker2}) - Test Set')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_normalized_prices(data: pd.DataFrame, ticker1: str, ticker2: str):
    plt.figure(figsize=(14, 7))

    # Normaliza los datos (Z-score) para ponerlos en la misma escala
    norm_data = (data - data.mean()) / data.std()

    norm_data[ticker1].plot(label=ticker1, color='blue', alpha=0.9)
    norm_data[ticker2].plot(label=ticker2, color='orange', alpha=0.9)

    plt.title(f'Normalized Historical Prices ({ticker1} & {ticker2}) - Test Set')
    plt.ylabel('Normalized Price (Z-Score)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_portfolio_value(portfolio_value: list, dates: pd.DatetimeIndex, ticker1: str, ticker2: str, threshold: float):
    plt.figure(figsize=(14, 7))
    pd.Series(portfolio_value, index=dates, name="Portfolio Value").plot(color='blue')
    plt.title(f'Portfolio Value Over Time ({ticker1} & {ticker2})\nStd: {threshold:.2f}')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.show()


def plot_dynamic_spread(vecm_norm_history: list, dates: pd.DatetimeIndex, threshold: float):
    plt.figure(figsize=(14, 7))
    pd.Series(vecm_norm_history, index=dates, name="VECM Norm").plot(color='purple', label='Normalized VECM Signal')
    plt.axhline(threshold, color='red', linestyle='--', label=f'+{threshold:.2f} Std (Entry)')
    plt.axhline(-threshold, color='green', linestyle='--', label=f'-{threshold:.2f} Std (Entry)')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Dynamic VECM Spread Signal and Trading Thresholds')
    plt.ylabel('Z-Score')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_hedge_ratio(hedge_ratio_history: list, dates: pd.DatetimeIndex):
    plt.figure(figsize=(14, 7))
    pd.Series(hedge_ratio_history, index=dates, name="Hedge Ratio").plot(color='orange')
    plt.title('Dynamic Hedge Ratio (KF1) Over Time')
    plt.ylabel('Hedge Ratio (B1)')
    plt.xlabel('Date')
    plt.show()


def plot_trades_on_price(data: pd.DataFrame, ticker1: str, ticker2: str, all_positions: list):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    data[ticker1].plot(ax=ax1, label='Price', color='gray')
    data[ticker2].plot(ax=ax2, label='Price', color='gray')

    ax1.set_title(f'Trades for {ticker1}')
    ax2.set_title(f'Trades for {ticker2}')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Price')

    for pos in all_positions:
        if pos.ticker == ticker1:
            ax = ax1
        elif pos.ticker == ticker2:
            ax = ax2
        else:
            continue

        if pos.position_type == 'long':
            ax.plot(pos.entry_date, pos.entry_price, '^', color='green', markersize=10, label='Long Entry')
            if pos.exit_date:
                ax.plot(pos.exit_date, pos.exit_price, 'x', color='green', markersize=10, label='Long Exit')
        elif pos.position_type == 'short':
            ax.plot(pos.entry_date, pos.entry_price, 'v', color='red', markersize=10, label='Short Entry')
            if pos.exit_date:
                ax.plot(pos.exit_date, pos.exit_price, 'x', color='red', markersize=10, label='Short Exit')

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


def plot_returns_distribution(all_positions: list):
    if not all_positions:
        print("No positions to plot returns distribution.")
        return

    returns = [pos.pnl / (pos.n_shares * pos.entry_price) for pos in all_positions if pos.n_shares > 0]

    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=50, kde=True, color='blue')
    plt.title('Distribution of PnL per Trade')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.axvline(0, color='black', linestyle='--')
    plt.show()


def plot_vecm_signals(observed_signal: list, filtered_signal: list, dates: pd.DatetimeIndex):
    plt.figure(figsize=(14, 7))

    observed_series = pd.Series(observed_signal, index=dates, name='Vecm (Observed)')
    filtered_series = pd.Series(filtered_signal, index=dates, name='Vecm Hat (Filtered)')

    observed_series.plot(color='gray', alpha=0.7, label='Vecm (Observed)')
    filtered_series.plot(color='blue', linewidth=2, label='Vecm Hat (Filtered)')

    plt.title('VECM Signal vs. Kalman-Filtered Signal (KF2)')
    plt.ylabel('Signal Value (Spread)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()

def plot_dynamic_eigenvectors(e1_history: list, e2_history: list, dates: pd.DatetimeIndex):
    plt.figure(figsize=(14, 7))
    pd.Series(e1_history, index=dates, name="Componente 1 (e1)").plot(color='c', label='Componente 1 (e1)')
    pd.Series(e2_history, index=dates, name="Componente 2 (e2)").plot(color='m', label='Componente 2 (e2)')
    plt.title('Evolución de Componentes del Eigenvector (KF2)')  # <-- Título cambiado
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Componente')
    plt.legend()
    plt.show()

def plot_kf1_spread(spread_history: list, dates: pd.DatetimeIndex, ticker1: str, ticker2: str):
    plt.figure(figsize=(14, 7))
    spread_series = pd.Series(spread_history, index=dates, name="Spread KF1")
    spread_series.plot(color='darkorange', label='Spread (KF1)', alpha=0.9)  # <-- Color cambiado

    mean_spread = spread_series.mean()
    std_spread = spread_series.std()

    plt.axhline(mean_spread, color='blue', linestyle='--', linewidth=1.5,
                    label='Media del Spread')  # <-- Color cambiado
    plt.fill_between(
        spread_series.index,
        mean_spread + std_spread,
        mean_spread - std_spread,
        color='blue',  # <-- Color cambiado
        alpha=0.1,
        label='±1 Desv. Estándar'
    )

    plt.title(f'Análisis del Spread (KF1) - {ticker2} vs {ticker1}')  # <-- Título cambiado
    plt.xlabel('Fecha')
    plt.ylabel('Valor del Spread (P2 - β_t * P1)')
    plt.legend()
    plt.show()

def plot_spread_comparison(kf1_spread_history: list, vecm_norm_history: list, dates: pd.DatetimeIndex):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Eje 1 para Spread KF1 (Bruto)
    color_ax1 = 'tab:orange'  # <-- Color cambiado
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Spread Bruto (KF1)', color=color_ax1)
    ax1.plot(dates, kf1_spread_history, color=color_ax1, label='Spread (KF1)')
    ax1.tick_params(axis='y', labelcolor=color_ax1)
    ax1.grid(False)  # Desactivar grid de este eje para que no se vea doble

    # Eje 2 para VECM Norm (Z-Score)
    ax2 = ax1.twinx()
    color_ax2 = 'tab:purple'  # <-- Color cambiado
    ax2.set_ylabel('Señal Normalizada (VECM KF2)', color=color_ax2)
    ax2.plot(dates, vecm_norm_history, color=color_ax2, label='VECM Normalizado (KF2)', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color_ax2)
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)  # Línea cero para Z-score

    fig.suptitle('Comparativa de Señales: KF1 (Bruto) vs. KF2 (Normalizado)')  # <-- Título cambiado
    fig.tight_layout()

    # Leyenda combinada
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()