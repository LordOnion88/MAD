# ventas_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
#           FUNCIONES DE PROCESAMIENTO
# ==========================================

def cargar_dataset(url):
    """
    Cargar el dataset desde una URL o ruta local.
    
    Args:
        url (str): Ruta o URL del archivo CSV.

    Returns:
        pd.DataFrame: Dataset cargado en un DataFrame.
    """
    try:
        dataset = pd.read_csv(url, sep=',')
        print(f"Dataset cargado exitosamente: {dataset.shape[0]} filas, {dataset.shape[1]} columnas")
        return dataset
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        return None
    

