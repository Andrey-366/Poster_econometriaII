# %% Importación de paquetes ============================

# Trabajar con rutas relativas en python
from curses import raw
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

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "datos"

# Rutas de las bases de datos
ruta_exp = (
    DATA_DIR / "conjunto_exportaciones_bogota_21052026.xlsx"
)  # Base de datos de exportaciones


# %% volverlo data frame

expo_base = pd.read_excel(ruta_exp)

# %%
