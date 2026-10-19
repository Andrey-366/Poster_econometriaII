# %% Importación de paquetes ============================

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


# %% volverlo data frame/ Cargar el archivo CSV
ipi_raw = pd.read_excel(ruta_ipi, sheet_name=3, header=8)
print(ipi_raw.head())

# %% Filtrar "Total Índice Produccion Industrial" =========================


df_ipi = ipi_raw[
    ipi_raw["Descripción"].str.contains(
        "Total Índice ProducciOn Industrial", case=False, na=False
    )
].copy()
print(df_ipi.tail())

# %% Creación del índice temporal
fechas_expo_base = pd.date_range(start="1990-01-01", periods=len(expo_base), freq="MS")

# Agregar el índice temporal a la base de datos de exportaciones tradicionales
expo_base.index = fechas_expo_base

# El tipo de objeto de la base de datos sigue siendo Pandas.DataFrame
print(type(expo_base))

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print(expo_base.head())  # Primeras observaciones
print(expo_base.tail())  # Últimas observaciones
