# utils.py

import warnings
warnings.filterwarnings('ignore')

#Procesamiento de datos

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from skforecast.Sarimax import Sarimax
import pmdarima as pm
from statsmodels.tsa.stattools import acf, q_stat
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Obtener Datos
import yfinance as yf

#Prophet

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics

# Visualizacion

import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Multi-processing
from multiprocessing import Pool, cpu_count

# Spark
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

# progreso
#from tqdm import tqdm

# Tiempo
#from time import time

# ==========================================
#           FUNCIONES DE PROCESAMIENTO
# ==========================================

def cargar_dataset(filepath, separador=',', encoding='utf-8', sheet_name=0):
    """
    Carga un dataset desde un archivo CSV o Excel.

    Args:
        filepath (str): Ruta o URL del archivo (CSV o Excel).
        separador (str, optional): Separador utilizado en el archivo CSV. Por defecto es ','.
        encoding (str, optional): Codificación del archivo. Por defecto es 'utf-8'.
        sheet_name (str/int, optional): Nombre o índice de la hoja en archivos Excel. Por defecto es la primera hoja (0).
    
    Returns:
        pd.DataFrame: Dataset cargado en un DataFrame.
    """
    try:
        # Verificar la extensión del archivo para determinar si es CSV o Excel
        if filepath.endswith('.csv'):
            # Cargar archivo CSV
            dataset = pd.read_csv(filepath, sep=separador, encoding=encoding, low_memory=False)
            print(f"Archivo CSV cargado exitosamente: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
        
        elif filepath.endswith('.xlsx'):
            # Cargar archivo Excel
            dataset = pd.read_excel(filepath, sheet_name=sheet_name)
            print(f"Archivo Excel cargado exitosamente: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
        
        else:
            print("Formato de archivo no soportado. Por favor, use .csv o .xlsx")
            return None
        
        return dataset
    
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None
    

# ==========================================
#           FUNCIONES GRAFICAS
# ==========================================

def serie_de_tiempo(Dataset,Cantidades):
    """
    Genera una grafica de una linea de tiempo de un dataset.

    Args:
        Dataset (str): dataset utilizado.
        Cantidades (str): Columna del dataset donde se encuentran las cantidades
    
    Returns:
        pd.DataFrame: Dataset cargado en un DataFrame.
    """

    # Definir las fechas mínima y máxima para el título
    
    fecha_min = Dataset.index.min().strftime('%Y-%m-%d')
    fecha_max = Dataset.index.max().strftime('%Y-%m-%d')

    sns.set_theme()
    plt.figure(figsize=(12,8))
    sns.lineplot(
        data = Dataset,
        x = Dataset.index,
        y = Cantidades
    )
    plt.title(f'Serie de tiempor de {Cantidades} del {fecha_min} al {fecha_max}')
    plt.xlabel('Fecha')  
    plt.ylabel(Cantidades)
    plt.show()

def plot_time_series_analysis(df, column, diff_lag=12, acf_lags=40):
    """
    Función para graficar la serie original, diferenciada y diferenciada estacionalmente, 
    junto con sus ACF y PACF.

    Parámetros:
    - df: DataFrame que contiene la serie de tiempo.
    - column: Columna del DataFrame que se va a analizar.
    - diff_lag: Lag para la diferenciación estacional (por defecto es 12).
    - acf_lags: Número de rezagos para calcular ACF y PACF (por defecto es 40).
    """

    # Diferenciación de la serie
    differenced_series = df[column].diff().dropna()
    seasonal_differenced_series = df[column].diff(diff_lag).dropna()

    plt.figure(figsize=(20, 12))

    # Serie original
    plt.subplot(3, 3, 1)
    df[column].plot(ax=plt.gca())
    plt.title('Serie Original')

    # ACF de la serie original
    plt.subplot(3, 3, 2)
    plot_acf(df[column], lags=acf_lags, ax=plt.gca())
    plt.title('ACF - Serie Original')

    # PACF de la serie original
    plt.subplot(3, 3, 3)
    plot_pacf(df[column], lags=acf_lags, ax=plt.gca())
    plt.title('PACF - Serie Original')

    # Serie diferenciada
    plt.subplot(3, 3, 4)
    differenced_series.plot(ax=plt.gca())
    plt.title('Serie Diferenciada')

    # ACF de la serie diferenciada
    plt.subplot(3, 3, 5)
    plot_acf(differenced_series, lags=acf_lags, ax=plt.gca())
    plt.title('ACF - Serie Diferenciada')

    # PACF de la serie diferenciada
    plt.subplot(3, 3, 6)
    plot_pacf(differenced_series, lags=acf_lags, ax=plt.gca())
    plt.title('PACF - Serie Diferenciada')

    # Serie diferenciada estacionalmente
    plt.subplot(3, 3, 7)
    seasonal_differenced_series.plot(ax=plt.gca())
    plt.title('Serie Diferenciada Estacionalmente')

    # ACF de la serie diferenciada estacionalmente
    plt.subplot(3, 3, 8)
    plot_acf(seasonal_differenced_series, lags=acf_lags, ax=plt.gca())
    plt.title('ACF - Serie Diferenciada Estacionalmente')

    # PACF de la serie diferenciada estacionalmente
    plt.subplot(3, 3, 9)
    plot_pacf(seasonal_differenced_series, lags=acf_lags, ax=plt.gca())
    plt.title('PACF - Serie Diferenciada Estacionalmente')

    plt.tight_layout()
    plt.show()