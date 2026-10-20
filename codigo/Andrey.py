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
    trend="c",  # incluir constante en el modelo
)
resultado_a = modelo_a.fit(
    disp=False
)  # fit ajusta el modelo y disp evita que se muestren iteraciones en la consola
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
    trend="c",
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
