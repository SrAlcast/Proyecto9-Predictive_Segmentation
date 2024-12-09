
# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: f"{float(x):.4f}"})

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la regresión lineal y la evaluación del modelo
# -----------------------------------------------------------------------
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

def graficar_arbol_decision(modelo, nombres_caracteristicas, tamano_figura=(30, 30), tamano_fuente=12):
    """
    Grafica un árbol de decisión con opciones personalizables.

    Parámetros:
        modelo: Árbol de decisión entrenado (DecisionTreeClassifier o DecisionTreeRegressor).
        nombres_caracteristicas: Lista o índice con los nombres de las características (columnas).
        tamano_figura: Tuple, tamaño de la figura (ancho, alto).
        tamano_fuente: Tamaño de la fuente en la gráfica.
    """
    plt.figure(figsize=tamano_figura)
    plot_tree(
        decision_tree=modelo,
        feature_names=nombres_caracteristicas,
        filled=True,  # Colorear los nodos
        rounded=True,  # Esquinas redondeadas
        fontsize=tamano_fuente,
        proportion=True,  # Mostrar proporciones en lugar de valores absolutos
        impurity=False  # Ocultar impureza de los nodos
    )
    plt.show()

def loo_cross_validation_rmse(model, X, y):
    """
    Realiza validación cruzada Leave-One-Out (LOO) y calcula el RMSE promedio.

    Parámetros:
        model: modelo de regresión (ej. LinearRegression de sklearn)
        X: DataFrame o matriz con las características (features)
        y: Serie o vector con el objetivo (target)
    
    Retorno:
        float: RMSE promedio obtenido en la validación cruzada
    """
    loo = LeaveOneOut()
    scores = []

    for train_index, test_index in tqdm(loo.split(X), total=len(X)):
        # Dividir los datos en entrenamiento y prueba
        X_train_cv, X_test_cv = X.iloc[train_index], X.iloc[test_index]
        y_train_cv, y_test_cv = y.iloc[train_index], y.iloc[test_index]

        # Entrenar el modelo y predecir
        model.fit(X_train_cv, y_train_cv)
        y_pred = model.predict(X_test_cv)

        # Calcular RMSE
        rmse = np.sqrt(mean_squared_error([y_test_cv.values[0]], y_pred))
        scores.append(rmse)

    # Calcular y retornar el RMSE promedio
    return np.mean(scores)


def rmse_plot(df, mse_column, columns, figure_size=(16, 10)):
    """
    Función sencilla para calcular y graficar el RMSE agrupado por columnas específicas.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - mse_column: Columna que contiene los valores de MSE.
    - columns: Lista de columnas para calcular y graficar el RMSE.
    - figure_size: Tamaño de la figura (ancho, alto) en tuplas.
    """
    num_columns = len(columns)
    fig, axes = plt.subplots(1, num_columns, figsize=figure_size, sharey=True)

    if num_columns == 1:  # Ajustar para un solo gráfico
        axes = [axes]

    for ax, column in zip(axes, columns):
        # Calcular RMSE agrupado
        rmse = np.sqrt(df.groupby(column)[mse_column].mean().abs())

        # Graficar
        sns.lineplot(x=rmse.index, y=rmse.values, ax=ax)
        ax.set_title(f"RMSE por {column}")
        ax.grid()

    plt.tight_layout()
    plt.show()

def analizar_correlaciones(df, target_column, threshold=0.05):
    """
    Analiza la correlación de las columnas con respecto a la variable objetivo.
    
    Parámetros:
    - df (DataFrame): Dataset que contiene las variables.
    - target_column (str): Nombre de la columna objetivo para calcular las correlaciones.
    - threshold (float): Umbral para identificar columnas con baja correlación (por defecto, 0.05).
    
    Retorna:
    - low_correlation_columns (list): Lista de columnas con correlación baja con la columna objetivo.
    """
    # Calcular la matriz de correlación
    correlation_matrix = df.corr()
    
    # Correlación con la variable objetivo
    correlation_with_target = correlation_matrix[target_column].sort_values(ascending=False)
    
    # Mostrar las correlaciones
    print(f"Correlaciones con '{target_column}':")
    print(correlation_with_target)
    
    # Visualizar correlaciones
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette="viridis")
    plt.xticks(rotation=90, ha='right')
    plt.title(f'Correlaciones de cada columna con "{target_column}"')
    plt.ylabel('Correlación')
    plt.xlabel('Columnas')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # Identificar columnas con baja correlación
    low_correlation_columns = correlation_with_target[correlation_with_target.abs() < threshold].index.tolist()
    print(f"\nColumnas con baja correlación (abs < {threshold}):")
    print(low_correlation_columns)
    
    return low_correlation_columns

def comparativa_graficos(y_test, y_pred_test):
    """
    Genera 4 gráficos comparativos entre valores reales y predicciones:
    1. Dispersión (Scatter Plot)
    2. Errores residuales
    3. Línea de valores reales vs predicciones
    4. KDE para comparar distribuciones de valores reales y predicciones
    """
    # Asegurarse de que y_test y y_pred_test sean unidimensionales
    if isinstance(y_test, (pd.DataFrame, pd.Series)):
        y_test = y_test.to_numpy().ravel()
    if isinstance(y_pred_test, (pd.DataFrame, pd.Series)):
        y_pred_test = y_pred_test.to_numpy().ravel()

    # Figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # 1. Gráfico de dispersión
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, ax=axes[0])
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Línea de identidad')
    axes[0].set_title('Dispersión: Predicciones vs Valores Reales')
    axes[0].set_xlabel('Valores Reales')
    axes[0].set_ylabel('Predicciones')
    axes[0].legend()
    axes[0].grid()
    
    # 2. Gráfico de errores residuales
    residuos = y_test - y_pred_test
    sns.scatterplot(x=y_pred_test, y=residuos, alpha=0.6, ax=axes[1])
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Errores Residuales')
    axes[1].set_xlabel('Predicciones')
    axes[1].set_ylabel('Residuos')
    axes[1].grid()
    
    # 3. Gráfico de barras de diferencia absoluta
    diferencias = abs(y_test - y_pred_test)
    sns.lineplot(x=range(len(diferencias)), y=diferencias, ax=axes[2], color='purple', alpha=0.7)
    axes[2].set_title('Diferencia Absoluta Suavizada')
    axes[2].set_xlabel('Índice')
    axes[2].set_ylabel('Diferencia Absoluta')
    axes[2].grid()

    # 4. KDE de distribuciones
    sns.kdeplot(y_test, color='blue', label='Valores Reales', fill=True, alpha=0.3, ax=axes[3])
    sns.kdeplot(y_pred_test, color='orange', label='Predicciones', fill=True, alpha=0.3, ax=axes[3])
    axes[3].set_title('Distribución (KDE) de Valores Reales y Predicciones')
    axes[3].set_xlabel('Valor')
    axes[3].set_ylabel('Densidad')
    axes[3].legend()
    axes[3].grid()
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()

def plot_real_vs_predicted(y_test, y_test_pred, y_train, y_train_pred):
    """
    Plots Real vs Predicted Prices for test and train datasets.

    Parameters:
        y_test (array-like): Actual target values for the test set.
        y_test_pred (array-like): Predicted target values for the test set.
        y_train (array-like): Actual target values for the train set.
        y_train_pred (array-like): Predicted target values for the train set.
    """
    plt.figure(figsize=(10, 6), dpi=150)
    plt.suptitle('Real vs. Predicted Prices')

    # Test data plot
    plt.subplot(2, 1, 1)
    sns.scatterplot(x=y_test, y=y_test_pred, alpha=0.6, s=10, label="Test data")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect prediction line", lw=0.7)
    plt.xlabel('Real Prices (y_test)')
    plt.ylabel('Predicted Prices (y_test_pred)')
    plt.legend()

    # Train data plot
    plt.subplot(2, 1, 2)
    sns.scatterplot(x=y_train, y=y_train_pred, alpha=0.6, s=10, color="forestgreen", label="Train data")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect prediction line", lw=0.7)
    plt.xlabel('Real Prices (y_train)')
    plt.ylabel('Predicted Prices (y_train_pred)')
    plt.legend()

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_residuals(y_test, y_test_pred, y_train, y_train_pred):
    """
    Plots residual plots (absolute and relative) for test and train datasets.

    Parameters:
        y_test (array-like): Actual target values for the test set.
        y_test_pred (array-like): Predicted target values for the test set.
        y_train (array-like): Actual target values for the train set.
        y_train_pred (array-like): Predicted target values for the train set.
    """
    plt.figure(figsize=(10, 6), dpi=150)
    plt.suptitle('Residual Plot (Absolute and Relative)')

    # Absolute residuals for test data
    residuals_test = y_test_pred - y_test
    plt.subplot(2, 2, 1)
    sns.scatterplot(x=y_test, y=residuals_test, alpha=0.6, s=10, label="Test Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_test)')
    plt.ylabel('Residuals')
    plt.legend()

    # Absolute residuals for train data
    residuals_train = y_train_pred - y_train
    plt.subplot(2, 2, 2)
    sns.scatterplot(x=y_train, y=residuals_train, alpha=0.6, s=10, color="forestgreen", label="Train Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_train)')
    plt.ylabel('Residuals')
    plt.legend()

    # Relative residuals (%) for test data
    relative_residuals_test = (y_test_pred - y_test) / y_test * 100
    plt.subplot(2, 2, 3)
    sns.scatterplot(x=y_test, y=relative_residuals_test, alpha=0.6, s=10, label="Test Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_test)')
    plt.ylabel('Residuals (%)')
    plt.legend()

    # Relative residuals (%) for train data
    relative_residuals_train = (y_train_pred - y_train) / y_train * 100
    plt.subplot(2, 2, 4)
    sns.scatterplot(x=y_train, y=relative_residuals_train, alpha=0.6, s=10, color="forestgreen", label="Train Data")
    plt.axhline(0, color='red', linestyle='--', label="Perfect prediction", lw=0.7)
    plt.xlabel('Real Prices (y_train)')
    plt.ylabel('Residuals (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def analizar_correlaciones(df, target_column, threshold=0.05):
    """
    Analiza la correlación de las columnas con respecto a la variable objetivo.
    
    Parámetros:
    - df (DataFrame): Dataset que contiene las variables.
    - target_column (str): Nombre de la columna objetivo para calcular las correlaciones.
    - threshold (float): Umbral para identificar columnas con baja correlación (por defecto, 0.05).
    
    Retorna:
    - low_correlation_columns (list): Lista de columnas con correlación baja con la columna objetivo.
    """
    # Calcular la matriz de correlación
    correlation_matrix = df.corr()
    
    # Correlación con la variable objetivo
    correlation_with_target = correlation_matrix[target_column].sort_values(ascending=False)
    
    # Mostrar las correlaciones
    print(f"Correlaciones con '{target_column}':")
    print(correlation_with_target)
    
    # Visualizar correlaciones
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlation_with_target.index, y=correlation_with_target.values, palette="viridis")
    plt.xticks(rotation=90, ha='right')
    plt.title(f'Correlaciones de cada columna con "{target_column}"')
    plt.ylabel('Correlación')
    plt.xlabel('Columnas')
    plt.grid()
    plt.tight_layout()
    plt.show()
    
    # Identificar columnas con baja correlación
    low_correlation_columns = correlation_with_target[correlation_with_target.abs() < threshold].index.tolist()
    print(f"\nColumnas con baja correlación (abs < {threshold}):")
    print(low_correlation_columns)
    
    return low_correlation_columns

def comparativa_graficos(y_test, y_pred_test):
    """
    Genera 4 gráficos comparativos entre valores reales y predicciones:
    1. Dispersión (Scatter Plot)
    2. Errores residuales
    3. Línea de valores reales vs predicciones
    4. KDE para comparar distribuciones de valores reales y predicciones
    """
    # Asegurarse de que y_test y y_pred_test sean unidimensionales
    if isinstance(y_test, (pd.DataFrame, pd.Series)):
        y_test = y_test.to_numpy().ravel()
    if isinstance(y_pred_test, (pd.DataFrame, pd.Series)):
        y_pred_test = y_pred_test.to_numpy().ravel()

    # Figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # 1. Gráfico de dispersión
    sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.6, ax=axes[0])
    axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Línea de identidad')
    axes[0].set_title('Dispersión: Predicciones vs Valores Reales')
    axes[0].set_xlabel('Valores Reales')
    axes[0].set_ylabel('Predicciones')
    axes[0].legend()
    axes[0].grid()
    
    # 2. Gráfico de errores residuales
    residuos = y_test - y_pred_test
    sns.scatterplot(x=y_pred_test, y=residuos, alpha=0.6, ax=axes[1])
    axes[1].axhline(0, color='red', linestyle='--')
    axes[1].set_title('Errores Residuales')
    axes[1].set_xlabel('Predicciones')
    axes[1].set_ylabel('Residuos')
    axes[1].grid()
    
    # 3. Gráfico de barras de diferencia absoluta
    diferencias = abs(y_test - y_pred_test)
    sns.lineplot(x=range(len(diferencias)), y=diferencias, ax=axes[2], color='purple', alpha=0.7)
    axes[2].set_title('Diferencia Absoluta Suavizada')
    axes[2].set_xlabel('Índice')
    axes[2].set_ylabel('Diferencia Absoluta')
    axes[2].grid()

    # 4. KDE de distribuciones
    sns.kdeplot(y_test, color='blue', label='Valores Reales', fill=True, alpha=0.3, ax=axes[3])
    sns.kdeplot(y_pred_test, color='orange', label='Predicciones', fill=True, alpha=0.3, ax=axes[3])
    axes[3].set_title('Distribución (KDE) de Valores Reales y Predicciones')
    axes[3].set_xlabel('Valor')
    axes[3].set_ylabel('Densidad')
    axes[3].legend()
    axes[3].grid()
    
    # Ajustar diseño
    plt.tight_layout()
    plt.show()