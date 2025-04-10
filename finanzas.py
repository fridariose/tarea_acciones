import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import datetime
#a
# Configurar la página
st.set_page_config(page_title="VaR y CVaR de Portafolio", layout="wide")
st.title("📉 Análisis de Riesgo: VaR y CVaR de un Portafolio de Acciones")

# Opciones de portafolios
portfolios = {
    "META & GOOGL": (["META", "GOOGL"], [0.5, 0.5]),
    "AAPL & MSFT": (["AAPL", "MSFT"], [0.5, 0.5]),
    "TSLA & AMZN": (["TSLA", "AMZN"], [0.5, 0.5]),
}

# Selección del portafolio
selected_portfolio = st.selectbox("Selecciona un portafolio:", list(portfolios.keys()))
tickers, weights = portfolios[selected_portfolio]
weights = np.array(weights)

# Rango de fechas
year = datetime.datetime.today().year
start_date = f"{year}-01-02"
end_date = datetime.datetime.today().strftime("%Y-%m-%d")

# Nivel de confianza
confidence_level = st.radio("Nivel de confianza:", [0.95, 0.99], index=0)
tail_prob = 1 - confidence_level

# Descargar datos
data = yf.download(tickers, start=start_date, end=end_date)["Close"].dropna()
returns = data.pct_change().dropna()
portfolio_returns = returns.dot(weights)

# VaR y CVaR
mean_ret = portfolio_returns.mean()
std_ret = portfolio_returns.std()
z_score = norm.ppf(tail_prob)

historical_VaR = np.percentile(portfolio_returns, tail_prob * 100)
parametric_VaR = mean_ret + z_score * std_ret
simulated_returns = np.random.normal(mean_ret, std_ret, 10000)
mc_VaR = np.percentile(simulated_returns, tail_prob * 100)
historical_CVaR = portfolio_returns[portfolio_returns <= historical_VaR].mean()

# Mostrar resultados
st.subheader("📊 Resultados de Riesgo")
st.write(f"Portafolio: **{selected_portfolio}** ({confidence_level*100:.0f}% confianza)")
st.write(f"- **VaR Histórico**: {historical_VaR:.4%}")
st.write(f"- **VaR Paramétrico**: {parametric_VaR:.4%}")
st.write(f"- **VaR Monte Carlo**: {mc_VaR:.4%}")
st.write(f"- **CVaR Histórico**: {historical_CVaR:.4%}")

# Graficar
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(portfolio_returns, bins=50, density=True, alpha=0.5, label="Retornos del Portafolio")
ax1.axvline(historical_VaR, color="red", linestyle="--", linewidth=2, label="VaR Histórico")
ax1.axvline(parametric_VaR, color="blue", linestyle="--", linewidth=2, label="VaR Paramétrico")
ax1.axvline(mc_VaR, color="green", linestyle="--", linewidth=2, label="VaR Monte Carlo")
ax1.set_title("Histograma de Retornos con VaR")
ax1.set_xlabel("Retorno Diario")
ax1.set_ylabel("Densidad")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(portfolio_returns.index, portfolio_returns, label="Retornos diarios", color="purple")
ax2.axhline(historical_VaR, color="red", linestyle="--", linewidth=1.5, label="VaR Histórico")
ax2.axhline(parametric_VaR, color="blue", linestyle="--", linewidth=1.5, label="VaR Paramétrico")
ax2.axhline(mc_VaR, color="green", linestyle="--", linewidth=1.5, label="VaR Monte Carlo")
ax2.set_title("Serie de Tiempo de Retornos del Portafolio")
ax2.set_xlabel("Fecha")
ax2.set_ylabel("Retorno Diario")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)