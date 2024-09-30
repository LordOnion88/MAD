# ventas_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

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
    

