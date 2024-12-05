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

def metricas(y_train, y_train_pred, y_test, y_test_pred, prob_train=None, prob_test=None):
    """
    Genera una tabla comparativa de métricas entre los conjuntos de entrenamiento y prueba.

    Parámetros:
        y_train (array-like): Valores reales del conjunto de entrenamiento.
        y_train_pred (array-like): Predicciones del modelo en el conjunto de entrenamiento.
        y_test (array-like): Valores reales del conjunto de prueba.
        y_test_pred (array-like): Predicciones del modelo en el conjunto de prueba.
        prob_train (array-like, opcional): Probabilidades de predicción en el conjunto de entrenamiento.
        prob_test (array-like, opcional): Probabilidades de predicción en el conjunto de prueba.

    Retorna:
        pd.DataFrame: DataFrame con las métricas comparadas para entrenamiento y prueba.
    """
    # Métricas para conjunto de entrenamiento
    metricas_train = {
        "accuracy": accuracy_score(y_train, y_train_pred),
        "precision": precision_score(y_train, y_train_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_train, y_train_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_train, y_train_pred, average='weighted', zero_division=0),
        "kappa": cohen_kappa_score(y_train, y_train_pred)
    }

    # Métricas para conjunto de prueba
    metricas_test = {
        "accuracy": accuracy_score(y_test, y_test_pred),
        "precision": precision_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_test_pred, average='weighted', zero_division=0),
        "kappa": cohen_kappa_score(y_test, y_test_pred)
    }

    # Combinar métricas en un DataFrame
    return pd.DataFrame({"train": metricas_train, "test": metricas_test})

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

def comparador_curvas_auc(modelos, X_test, y_test, nombres_modelos):
    """
    Genera una visualización de las curvas AUC (ROC) para comparar cinco modelos.
    
    Parámetros:
        modelos (list): Lista de los cinco modelos ajustados.
        X_test (array-like): Conjunto de características de prueba.
        y_test (array-like): Etiquetas reales de prueba.
        nombres_modelos (list): Lista de nombres para cada modelo (en el mismo orden que `modelos`).
    
    Retorna:
        None: Muestra un gráfico con las curvas ROC para los modelos.
    """
    if len(modelos) != 5 or len(nombres_modelos) != 5:
        raise ValueError("Debe proporcionar exactamente cinco modelos y cinco nombres.")

    plt.figure(figsize=(10, 8))

    for i, modelo in enumerate(modelos):
        if not hasattr(modelo, "predict_proba"):
            raise ValueError(f"El modelo '{nombres_modelos[i]}' no tiene el método 'predict_proba'.")
        
        # Obtener las probabilidades predichas
        probas_test = modelo.predict_proba(X_test)[:, 1]
        
        # Calcular la curva ROC
        fpr, tpr, _ = roc_curve(y_test, probas_test)
        roc_auc = auc(fpr, tpr)
        
        # Graficar la curva ROC
        plt.plot(fpr, tpr, lw=2, label=f"{nombres_modelos[i]} (AUC = {roc_auc:.2f})")
    
    # Agregar líneas de referencia y etiquetas
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=2, label="Referencia")
    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR)")
    plt.title("Comparación de Curvas ROC para 5 Modelos")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()

def comparar_matrices_confusion(X_train, y_train, X_test, y_test, modelos, nombres_modelos, figsize=(12, 8)):
    """
    Compara las matrices de confusión de varios modelos y muestra valores absolutos.

    Parámetros:
        X_train (array-like): Conjunto de características de entrenamiento.
        y_train (array-like): Etiquetas reales de entrenamiento.
        X_test (array-like): Conjunto de características de prueba.
        y_test (array-like): Etiquetas reales de prueba.
        modelos (list): Lista de modelos ajustados (deben implementar `fit` y `predict`).
        nombres_modelos (list): Lista de nombres para identificar los modelos.
        figsize (tuple): Tamaño del gráfico.

    Retorna:
        None: Muestra los gráficos de las matrices de confusión.
    """
    if len(modelos) != len(nombres_modelos):
        raise ValueError("El número de modelos y nombres de modelos debe coincidir.")

    # Configuración del gráfico
    n_modelos = len(modelos)
    cols = 3  # Número de columnas en el gráfico
    rows = (n_modelos + cols - 1) // cols  # Número de filas necesarias

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, (modelo, nombre) in enumerate(zip(modelos, nombres_modelos)):
        # Ajustar el modelo y generar predicciones
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        # Calcular la matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Graficar la matriz de confusión con valores absolutos
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[i])
        axes[i].set_title(nombre)
        axes[i].set_xlabel('Predicción')
        axes[i].set_ylabel('Real')

    # Eliminar ejes adicionales si hay más subplots que modelos
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()