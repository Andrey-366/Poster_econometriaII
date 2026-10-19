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
meses = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}
mes = pd.to_numeric(df_ipi["Mes"], errors="coerce")
mes = mes.fillna(
    df_ipi["Mes"].astype(str).str.strip().str.lower().map(meses)
)
df_ipi["Fecha"] = pd.to_datetime(
    {"year": df_ipi["Año"], "month": mes, "day": 1}, errors="coerce"
)
# %%
print(df_ipi.head())

# %% Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "índice de producción industrial"
plt.figure(figsize=(10, 5))
plt.plot(
    df_ipi["Fecha"], df_ipi["Producción real"], marker="o", linestyle="-"
)  # llamo la base y escogo con [] la columna. "0" pone el punto, "-" pone la linea continua
plt.title("Índice de Producción Industrial, 2014-2026")
plt.xlabel("Año ")
plt.ylabel("Valor")
plt.grid(True)
plt.show()
# %%
