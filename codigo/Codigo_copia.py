# %% Importacion de paquetes ============================

# Trabajar con rutas relativas en python
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


# %%
# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "datos"
ruta_ipi = (
    DATA_DIR
    / "Índice_de_Producción_Industrial_(IPI)._Serie_desestacionalizada (1).xlsx"
)


# %% volverlo data frame/ Cargar el archivo EXEL
ipi_raw = pd.read_excel(ruta_ipi, sheet_name=1, header=0)
print(ipi_raw.head())


# %% unir mes y año en una sola columna

ipi_serie = ipi_raw.set_index("Fecha")["ipi"]  # extraer fecha y producción real
print(ipi_serie.head())

# %% Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "índice de producción industrial"
plt.figure(figsize=(10, 5))  # tamaño
plt.plot(
    ipi_serie, marker="o", linestyle="-"
)  # "0" pone el punto, "-" pone la linea continua
plt.title("Índice de Producción Industrial, 2014-2026")
plt.xlabel("Año ")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% FAC y FACP de la serie original

fig, axes = plt.subplots(
    1, 2, figsize=(14, 5)
)  # crea dos cuadros para graficar FAC y FACP

plot_acf(
    ipi_serie,
    lags=34,  # numero de rezagos
    alpha=0.05,  # 95% de confianza
    bartlett_confint=False,  # ancho fijo
    ax=axes[0],
)

axes[0].set_title("FAC IPI")

plot_pacf(ipi_serie, lags=34, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP IPI")

plt.tight_layout()  # ajusta el espacio entre los subplots
plt.show()
# %% tiene raiz unitaria? Prueba de Dickey-Fuller aumentada (ADF) y KPSS

adf = adfuller(ipi_serie, regression="c", autolag="AIC")  # prueba ADF

print("Resultados de la prueba ADF:")
print("Estadístico ADF:", adf[0])
print("p-valor:", adf[1])
print("Rezagos usados:", adf[2])
print("N° observaciones:", adf[3])
print("Valores críticos:", adf[4])

print()
# Test KPSS


print()
if adf[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")


# %%
ipi_serie_diff = ipi_serie.diff().dropna()

# Gráfica de la serie diferenciada
plt.figure(figsize=(10, 5))
plt.plot(ipi_serie_diff)
plt.title("IPI Serie diferenciada")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()


# %% fac y facp de la diferencia
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(ipi_serie_diff, lags=34, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de la diferencia")

plot_pacf(ipi_serie_diff, lags=34, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de la diferencia")

plt.tight_layout()
plt.show()

# %% Estimación de modelos SARIMA candidatos basados en FAC y FACP =========
dummy = pd.DataFrame(
    0, index=ipi_serie.index, columns=["covid"]
)  # creo un dataframe de ceros con las fehcas de ipi_log
dummy.loc["2020-3":"2020-7", "covid"] = 1

# Candidato a ARIMA (1,1,0)
modelo_a = SARIMAX(
    ipi_serie,
    order=(1, 1, 1),  # p=1, d=1, q=0
    exog=dummy,
    trend="c",
    cov_type="robust",
)
resultado_a = modelo_a.fit(disp=False)

# fit ajusta el modelo por Máxima Verosimilitud y disp evita que se muestren iteraciones en la consola
print("--- MODELO CANDIDATO A ---")
print(resultado_a.summary())
print(
    f"AIC: {resultado_a.aic:.4f} | BIC: {resultado_a.bic:.4f}\n"
)  # extrae aic y bic del modelo ajustado y los imprime con 4 decimales, n es un salto de linea

# Candidato B (1, 0, 0)
modelo_b = SARIMAX(ipi_serie, order=(3, 1, 0), exog=dummy, trend="c")
resultado_b = modelo_b.fit(disp=False)
print("--- MODELO CANDIDATO B ---")
print(resultado_b.summary())
print(f"AIC: {resultado_b.aic:.4f} | BIC: {resultado_b.bic:.4f}\n")

if resultado_a.aic < resultado_b.aic:
    print("El modelo candidato A tiene un mejor ajuste según AIC.")
else:
    print("El modelo candidato B tiene un mejor ajuste según AIC.")
if resultado_a.bic < resultado_b.bic:
    print("El modelo candidato A tiene un mejor ajuste según BIC.")
else:
    print("El modelo candidato B tiene un mejor ajuste según BIC.")
    # El modelo candidato A tiene un mejor ajuste segun aic y bic
# Todo ES significante

# %%
# paso dos 2 estimacion del modelo final

print("Estimacion del modelo final SARIMA(1,1,0)    ... con dummy")
print(resultado_a.summary())


# %%
# Paso 3: Validación de supuestos
# Prob(Q) = 0.78 (Ljung-Box): no se rechaza la hipótesis de no autocorrelación, así que los residuos se comportan como ruido blanco.
# Prob(JB) = 0.00 (Jarque-Bera): se rechaza normalidad.
# Prob(H) = 0.07>0.05 (heterocedasticidad): no se rechaza homocedasticidad.

# Residuales del modelo
#  el modelo pierda las primeras d + D*s observaciones para inicializarse osea 1
residuales = resultado_a.resid.iloc[1:].dropna()


# Gráfica de los residuales (deberían comportarse como ruido blanco)

plt.figure(figsize=(10, 5))
plt.plot(residuales)
plt.title("Residuales del modelo SARIMA(1,1,0)")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# Descripción de los residuales
print(residuales.describe())
# la dummy no pudo ajustar los residuales

# %% FAC y FACP de los residuales

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(residuales, lags=34, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de los residuales")

plot_pacf(residuales, lags=34, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de los residuales")

plt.tight_layout()
plt.show()

# en la fac: el modelo extrajo la estacionalidad anual, aun algunos rezagos se observa autocorrelacion, pero no es significativo
# %% Prueba Ljung-Box

ljung_box = acorr_ljungbox(
    residuales, lags=[6, 12, 18, 24, 30, 40, 60, 100], return_df=True
)

print("Prueba Ljung-Box")
print(ljung_box)

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.
# se cumple H0 para el resago 6 y 12

# %% FAC y FACP de los residuales al cuadrado

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(residuales**2, lags=50, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de los residuales al cuadrado")

plot_pacf(residuales**2, lags=50, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de los residuales al cuadrado")

plt.tight_layout()
plt.show()
# fac residuales^2: sale solo en el periodo y 13


# %% Prueba ARCH de heterocedasticidad

arch_test = het_arch(residuales, nlags=12)

print("Prueba ARCH")
print("LM statistic:", arch_test[0])
print("LM p-value:", arch_test[1])

# H0: no hay efectos ARCH.
# Si p-value > 0.05, no se rechaza H0.
# homocedasticidad, no hay efectos ARCH, si hay heterocedasticidad

# %% Q-Q plot de los residuos (normalidad)

plt.figure(figsize=(6, 6))
probplot(residuales.iloc[1:-1], dist="norm", plot=plt)
plt.title("Q-Q plot de los residuos")
plt.grid(True)
plt.show()
# segun Q-Q plot,son aproximadamente normales salvo por dos observaciones atípicas en la cola izquierda
# %%Prueba de normalidad Jarque-Bera
jb_test = jarque_bera(residuales.iloc[1:-1])

print("Prueba Jarque-Bera")
print("Jarque-Bera statistic:", jb_test.statistic)
print("Jarque-Bera p-value:", jb_test.pvalue)

# H0: los residuos no siguen una distribución normal.
# Si p-value < 0.05, se rechaza H0.
# el modelo rechaza normalidad


# %% PASO 4: Pronóstico

# La dummy covid vale 0 en el futuro

fechas_futuras = pd.date_range(
    ipi_serie.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS"
)  # -1 ultima fecha, pd.offsets.MonthBegin(1) para obtener el primer día del mes siguiente,
exog_futuro = pd.DataFrame(0, index=fechas_futuras, columns=["covid"])


# %%


pronostico = resultado_b.get_forecast(steps=12, exog=exog_futuro)

# Pronóstico puntual e intervalos de predicción (ya en niveles)
pronostico_puntual = pronostico.predicted_mean
intervalos = pronostico.conf_int(alpha=0.05)  # 95%

tabla_pronostico = pd.DataFrame(
    {
        "pronostico": pronostico_puntual,
        "limite_inferior": intervalos.iloc[:, 0],
        "limite_superior": intervalos.iloc[:, 1],
    }
)
print(tabla_pronostico.round(2))

#  Gráfica del pronóstico
plt.figure(figsize=(10, 5))
plt.plot(ipi_serie, label="Datos históricos")
plt.plot(pronostico_puntual, label="Pronóstico", color="red")
plt.fill_between(
    pronostico_puntual.index,
    intervalos.iloc[:, 0],
    intervalos.iloc[:, 1],
    color="orange",
    alpha=0.3,
    label="Intervalo de confianza 95%",
)
plt.title("Índice de Producción Industrial, 2014-2026")
plt.xlabel("Año")
plt.ylabel("Índice de producción real")
plt.legend()
plt.tight_layout()
plt.show()


# %%
