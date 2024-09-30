# utils.py

#Procesamiento de datos

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
