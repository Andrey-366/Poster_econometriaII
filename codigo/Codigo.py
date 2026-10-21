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
ruta_ipi = DATA_DIR / "anex_IPI_ago2026.xlsx"


# %% volverlo data frame/ Cargar el archivo EXEL
ipi_raw = pd.read_excel(ruta_ipi, sheet_name=3, header=8)
print(ipi_raw.head())

# %% Filtrar "Total Índice Produccion Industrial" =========================


df_ipi = ipi_raw[
    ipi_raw["Descripción"].str.contains(
        "Total Índice ProducciOn Industrial", case=False, na=False
    )
].copy()
print(df_ipi.tail())

# %% unir mes y año en una sola columna

df_ipi["Fecha"] = pd.to_datetime(
    {"year": df_ipi["Año"], "month": df_ipi["Mes"], "day": 1}, errors="coerce"
)
ipi_serie = df_ipi.set_index("Fecha")[
    "Producción real"
]  # extraer fecha y producción real
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

adf = adfuller(ipi_serie, regression="ct", autolag="AIC")  # prueba ADF

print("Resultados de la prueba ADF:")
print("Estadístico ADF:", adf[0])
print("p-valor:", adf[1])
print("Rezagos usados:", adf[2])
print("N° observaciones:", adf[3])
print("Valores críticos:", adf[4])

print()
# Test KPSS
kpss_result = kpss(ipi_serie, regression="ct", nlags="auto")


print("Resultados de la prueba KPSS:")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:", kpss_result[3])

print()
if adf[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")
if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# %% test con c en vez de ct constante- constante y tendencia
# prueba ADF
adf = adfuller(ipi_serie, regression="c", autolag="AIC")
print("p-valor:", adf[1])

# Test KPSS
kpss_result = kpss(ipi_serie, regression="ct", nlags="auto")
print("p-valor:", kpss_result[1])


print()
if adf[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")
if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# en ambos casos (c y ct)segun adf la serie es no estacionaria y segun kpss la serie es estacionaria, segun criterio de la ia le damos mas peso a la prueba de adf, por lo que aplicamos diferenciacion
# %% Serie diferenciada
ipi_serie_diff = ipi_serie.diff().dropna()

# Gráfica de la serie diferenciada
plt.figure(figsize=(10, 5))
plt.plot(ipi_serie_diff)
plt.title("IPI Serie diferenciada")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% Diferencia del logaritmo

ipi_serie_log_diff = np.log(ipi_serie).diff().dropna()  # tasa de crecimiento

# Gráfica de la serie de tiempo de la "diferencia del logaritmo de exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(ipi_serie_log_diff)
plt.title("Diferencia del logaritmo de la serie original")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% fac y facp de la diferencia del logaritmo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(ipi_serie_log_diff, lags=34, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de la diferencia del logaritmo")

plot_pacf(ipi_serie_log_diff, lags=34, alpha=0.05, ax=axes[1])

axes[1].set_title("FACP de la diferencia del logaritmo")

plt.tight_layout()
plt.show()
# %%
# segun el la fac y fac p identificamos un pico en el rezago 12 y 24 (parecido a como estaba en la serie original)
# en los primeros rezagos se observa un decaimiento rapido tanto de la fac como de la facp sugiriendo que el modelo sea p=1 o q=1
# ya se hizo una diferencia por lo que d=1
# modelos sugeridos sarima(1,1,0)(1,1,0)12 y sarima(0,1,1)(0,1,1)12, usando el logaritmo de la serie original


# %% Estimación de modelos SARIMA candidatos basados en FAC y FACP =========

ipi_log = np.log(ipi_serie)  # logaritmo de la serie original
# Candidato A (1,1,0)(1,1,0)12
modelo_a = SARIMAX(
    ipi_log,
    order=(1, 1, 0),  # p=1, d=1, q=0
    seasonal_order=(0, 1, 1, 12),  # P=0, D=1, Q=1, s=12 (estacionalidad anual)
)
resultado_a = modelo_a.fit(
    disp=False
)  # fit ajusta el modelo por Máxima Verosimilitud y disp evita que se muestren iteraciones en la consola
print("--- MODELO CANDIDATO A ---")
print(resultado_a.summary())
print(
    f"AIC: {resultado_a.aic:.4f} | BIC: {resultado_a.bic:.4f}\n"
)  # extrae aic y bic del modelo ajustado y los imprime con 4 decimales, n es un salto de linea


# Candidato B (0,1,1)(0,1,1)12
modelo_b = SARIMAX(
    ipi_log,
    order=(0, 1, 1),  # p=0, d=1, q=1
    seasonal_order=(0, 1, 1, 12),
)
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
    # El modelo candidato B tiene un mejor ajuste segun aic y bic


# %%
# paso dos 2 estimacion del modelo final (0,1,1)(0,1,1)12

print("Estimacion del modelo final SARIMA(0,1,1)(0,0,1)12... con dummy")


# para los residuales hay un dato atipico en 2020 por lo que añadiremos una dummy
# la serie estaba sobre diferenciada, resultando en un ma.s.l12 y sigma2 no significativo por lo que se tomo un D=0
dummy = pd.DataFrame(
    0, index=ipi_log.index, columns=["covid"]
)  # creo un dataframe de ceros con las fehcas de ipi_log
dummy.loc["2020-03-01":"2020-07-01", "covid"] = 1

modelo_b = SARIMAX(ipi_log, exog=dummy, order=(0, 1, 1), seasonal_order=(0, 0, 1, 12))
resultado_b = modelo_b.fit(disp=False)
print(resultado_b.summary())


# %%
# Paso 3: Validación de supuestos
# Prob(Q) = 0.57 (Ljung-Box): no se rechaza la hipótesis de no autocorrelación, así que los residuos se comportan como ruido blanco.
# Prob(JB) = 0.00 (Jarque-Bera): se rechaza normalidad.
# Prob(H) = 0.62heterocedasticidad): no se rechaza homocedasticidad.

# Residuales del modelo
#  el modelo pierda las primeras d + D*s observaciones para inicializarse osea 1
residuales = resultado_b.resid.iloc[1:].dropna()


# Gráfica de los residuales (deberían comportarse como ruido blanco)

plt.figure(figsize=(10, 5))
plt.plot(residuales)
plt.title("Residuales del modelo SARIMA(0,1,1)(0,0,1)12")
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

ljung_box = acorr_ljungbox(residuales, lags=[6, 12, 18, 24, 30], return_df=True)

print("Prueba Ljung-Box")
print(ljung_box)

# H0: no hay autocorrelación en los residuos.
# Si p-value > 0.05, no se rechaza H0.
# se cumple H0 para el resago 6 y 12

# %% FAC y FACP de los residuales al cuadrado

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(residuales**2, lags=34, alpha=0.05, bartlett_confint=False, ax=axes[0])

axes[0].set_title("FAC de los residuales al cuadrado")

plot_pacf(residuales**2, lags=34, alpha=0.05, ax=axes[1])

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
# homocedasticidad, no hay efectos ARCH

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
# A partir de la estimación del SARIMA (0, 1, 1) (0, 0, 1, 12) realizó el pronóstico

# La dummy covid vale 0 en el futuro
fechas_futuras = pd.date_range(
    ipi_log.index[-1] + pd.offsets.MonthBegin(1), periods=12, freq="MS"
)  # -1 ultima fecha, pd.offsets.MonthBegin(1) para obtener el primer día del mes siguiente,
exog_futuro = pd.DataFrame(0, index=fechas_futuras, columns=["covid"])


# Protonóstico 12 pasos adelante
pronostico_log = resultado_b.get_forecast(steps=12, exog=exog_futuro)

# Pronóstico puntual e intervalos de predicción del logaritmo de ipi
pronostico_puntual_log = pronostico_log.predicted_mean
varianza_pronostico_log = pronostico_log.var_pred_mean
intervalos_log = pronostico_log.conf_int()

# Al retransformar desde logaritmos, exp(E[log(y)]) corresponde a la mediana.
# Para obtener la media en niveles se usa la corrección por sesgo:
# E[y] = exp(mu + 0.5 * sigma^2), bajo normalidad en la escala logarítmica.
pronostico_nivel = np.exp(pronostico_puntual_log + 0.5 * varianza_pronostico_log)
intervalos_nivel = np.exp(intervalos_log)

tabla_pronostico = pd.DataFrame(
    {
        "pronostico": pronostico_nivel,
        "limite_inferior": intervalos_nivel.iloc[:, 0],
        "limite_superior": intervalos_nivel.iloc[:, 1],
    }
)

print(tabla_pronostico)

# Gráfica del pronóstico
plt.figure(figsize=(10, 5))
plt.plot(ipi_serie, label="Datos históricos")
plt.plot(pronostico_nivel, label="Pronóstico", color="red")
plt.fill_between(
    pronostico_nivel.index,
    intervalos_nivel.iloc[:, 0],
    intervalos_nivel.iloc[:, 1],
    color="orange",
    alpha=0.3,
    label="Intervalo de confianza",
)
plt.title("Índice de Producción Industrial, 2014-2026")
plt.xlabel("Año ")
plt.ylabel("Valor")
plt.show()

# %%
