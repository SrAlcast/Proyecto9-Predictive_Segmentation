# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve
)
import xgboost as xgb
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold,LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

def metricas(y_train, y_train_pred, y_test, y_test_pred):
    # Convertir DataFrames a Series si es necesario
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.squeeze()
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.squeeze()

    # Convertir a NumPy arrays
    y_train = y_train.values if hasattr(y_train, 'values') else y_train
    y_test = y_test.values if hasattr(y_test, 'values') else y_test

    # Verificar que contienen valores numéricos
    if not np.issubdtype(y_train.dtype, np.number) or not np.issubdtype(y_test.dtype, np.number):
        raise ValueError("y_train o y_test contienen valores no numéricos.")

    # Calcular métricas
    train_metricas = {
        'r2_score': round(r2_score(y_train, y_train_pred), 4),
        'MAE': round(mean_absolute_error(y_train, y_train_pred), 4),
        'MSE': round(mean_squared_error(y_train, y_train_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
    }
    
    test_metricas = {
        'r2_score': round(r2_score(y_test, y_test_pred), 4),
        'MAE': round(mean_absolute_error(y_test, y_test_pred), 4),
        'MSE': round(mean_squared_error(y_test, y_test_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
    }
    
    # Calcular diferencias
    diferencias = {
        metric: round(train_metricas[metric] - test_metricas[metric], 4) for metric in train_metricas
    }
    
    # Calcular porcentaje de diferencia relativa al valor mayor promedio
    porcentaje = {
        metric: round((diferencias[metric] / (train_metricas[metric] + test_metricas[metric]) / 2) * 100, 4)
        for metric in train_metricas
    }
    
    # Calcular el rango (entre y_train y y_test)
    global_min = min(y_train.min(), y_test.min())
    global_max = max(y_train.max(), y_test.max())
    rango = round((global_max - global_min), 4)
    
    # Ratio sobre el rango
    ratio_rango = {metric: (((train_metricas[metric] + test_metricas[metric]) / 2) * 100) / rango for metric in train_metricas}

    # Calcular porcentaje de influencia basado en la referencia
    porcentaje_rango = {
        metric: round((abs(diferencias[metric]) / rango) * 100, 4)
        for metric in diferencias
    }

       # Calcular el valor minimo de la media y mediana de la variable respuesta (entre y_train y y_test)
    media_respuesta = round((np.mean(y_train) + np.mean(y_test)) / 2, 4)
    
    # Ratio sobre la media
    ratio_media= {metric:(((train_metricas[metric]+test_metricas[metric])/2)*100)/media_respuesta for metric in train_metricas}

    # Calcular porcentaje de influencia basado en la referencia
    porcentaje_media = {
        metric: round((abs(diferencias[metric]) / media_respuesta) * 100, 4)
        for metric in diferencias
    }

    # Combinar resultados
    metricas = {
        'Train': train_metricas,
        'Test': test_metricas,
        'Diferencia Train-Test': diferencias,
        'Porcentaje diferencia (%)': porcentaje,
        'Rango valores': rango,
        'Ratio Rango (%)': ratio_rango,
        'Influencia dif rango (%)': porcentaje_rango,
        'Media':media_respuesta,
        'Ratio Media(%)':ratio_media,
        'Influencia dif media (%)': porcentaje_media,    
    }
    return pd.DataFrame(metricas).T


def combinar_metricas(model_names, *dfs):
    """
    Transforma métricas de varios modelos reorganizando las columnas 'train' y 'test' como filas,
    mientras que las métricas se convierten en columnas. Agrega una columna con los nombres de los modelos.

    Parámetros:
        model_names (list): Lista de nombres de modelos en el orden correspondiente a los DataFrames.
        *dfs (pd.DataFrame): Varios DataFrames, uno por cada modelo. Cada DataFrame debe tener:
                             - Filas: Métricas (e.g., `accuracy`, `precision`, `recall`, etc.).
                             - Columnas: `train` y `test`.

    Retorna:
        pd.DataFrame: DataFrame transformado con métricas como columnas, `Train` y `Test` como filas,
                      y una columna adicional para el nombre del modelo.
    """
    # Validar que el número de nombres coincida con el número de DataFrames
    if len(model_names) != len(dfs):
        raise ValueError("El número de nombres de modelos debe coincidir con el número de DataFrames.")

    # Lista para almacenar los DataFrames transformados
    dfs_transformados = []
    
    for model_name, df in zip(model_names, dfs):
        # Transponer el DataFrame para que las métricas sean columnas
        df_transposed = df.T
        df_transposed['Modelo'] = model_name  # Agregar la columna con el nombre del modelo
        dfs_transformados.append(df_transposed)

    # Combinar todos los DataFrames transformados
    df_combinado = pd.concat(dfs_transformados, axis=0, ignore_index=False)

    # Reorganizar columnas para que 'Modelo' sea la última columna
    columnas_reorganizadas = list(df_combinado.columns[:-1]) + ['Modelo']
    df_combinado = df_combinado[columnas_reorganizadas]

    return df_combinado

