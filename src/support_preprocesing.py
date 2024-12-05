# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math
import numpy as np
from itertools import product  # Para generar combinaciones
from tqdm import tqdm  # Para mostrar progreso
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt

# Imputación de nulos usando métodos avanzados estadísticos
# -----------------------------------------------------------------------
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE
import pandas as pd
from collections import Counter
import pickle

# Encoding de variables
# -----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from category_encoders import TargetEncoder
from scipy.stats import chi2_contingency


def exploracion_basica_dataframe(dataframe):
    """
    Realiza un análisis exploratorio básico de un DataFrame, mostrando información sobre duplicados,
    valores nulos, tipos de datos, valores únicos para columnas categóricas y estadísticas descriptivas
    para columnas categóricas y numéricas, agrupadas por la columna de control.

    Params:
    - dataframe (DataFrame): El DataFrame que se va a explorar.
    - columna_control (str): El nombre de la columna que se utilizará como control para dividir el DataFrame.

    Returns: 
    No devuelve nada directamente, pero imprime en la consola la información exploratoria.
    """
    print(f"El número de datos es {dataframe.shape[0]} y el de columnas es {dataframe.shape[1]}")
    print("\n ------------------------------- \n")

    print(f"Los duplicados que tenemos en el conjunto de datos son: {dataframe.duplicated().sum()}")
    print("\n ------------------------------- \n")
    # generamos un DataFrame con cantidad de valores unicos
    print("Los unicos que tenemos en el conjunto de datos son:")
    df_unique = pd.DataFrame({
        "count": dataframe.nunique(),
        "% unique": (dataframe.nunique() / dataframe.shape[0] * 100).round(2)})
    # Ordenar por porcentaje de valores únicos en orden descendente
    df_unique_sorted = df_unique.sort_values(by="% unique", ascending=False)
    # Mostrar el resultado
    display(df_unique_sorted)
    columnas_mayor_50_unicos = df_unique_sorted[df_unique_sorted["% unique"] > 50].index.tolist()
    # Imprimimos los nombres de las columnas
    print("Las columnas con más del 50% de valores unicos son:")
    for col in columnas_mayor_50_unicos:
        print(col)
    print("\n ------------------------------- \n")
    columnas_solo_1_unico = df_unique_sorted[df_unique_sorted["count"] ==1].index.tolist()
    # Imprimimos los nombres de las columnas
    print("Las columnas con solo 1 valor único son:")
    for col in columnas_solo_1_unico:
        print(col)
    print("\n ------------------------------- \n")
    # generamos un DataFrame para los valores nulos
    df_nulos = pd.DataFrame({"count": dataframe.isnull().sum(),"% nulos": (dataframe.isnull().sum() / dataframe.shape[0]).round(3) * 100}).sort_values(by="% nulos", ascending=False)
    df_nulos = df_nulos[df_nulos["count"] > 0]
    df_nulos_sorted = df_nulos.sort_values(by="% nulos", ascending=False)
    # Muestra el resultado
    print("Los nulos que tenemos en el conjunto de datos son:")
    display(df_nulos_sorted)
    columnas_mayor_50 = df_nulos[df_nulos["% nulos"] > 50].index.tolist()

    # Imprimimos los nombres de las columnas
    print("Las columnas con más del 50% de valores nulos son:")
    for col in columnas_mayor_50:
        print(col)

    print("\n ------------------------------- \n")
    print(f"Los tipos de las columnas son:")
    display(pd.DataFrame(dataframe.dtypes, columns = ["tipo_dato"]))
    print("\n ------------------------------- \n")


    print("Los valores que tenemos para las columnas categóricas son: ")
    dataframe_categoricas = pd.DataFrame(dataframe.select_dtypes(include = "O"))
    display(pd.DataFrame(dataframe_categoricas.columns,columns=["columna"]))
    print("\n ------------------------------- \n")


    print("Los valores que tenemos para las columnas numéricas son: ")
    dataframe_numericas = pd.DataFrame(dataframe.select_dtypes(include = np.number))
    display(pd.DataFrame(dataframe_numericas.columns,columns=["columna"]))
    print("\n ------------------------------- \n")


    for col in dataframe_categoricas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        if dataframe[col].isnull().sum()>0:
            print(f"--->La columna {col.upper()} tiene valores nulos")
        df_counts = pd.DataFrame({"count": dataframe[col].value_counts(dropna=False),"porcentaje (%)": (dataframe[col].value_counts(dropna=False, normalize=True) * 100).round(3)})
        display(df_counts)
        print("\n ------------------------------- \n")
    
    print("_______________________________________________________")
    print("Los valores que tenemos para las columnas numéricas son: ")
    dataframe_numericas = pd.DataFrame(dataframe.select_dtypes(include =np.number))
    
    for col in dataframe_numericas.columns:
        print(f"La columna {col.upper()} tiene las siguientes valore únicos:")
        if dataframe[col].isnull().sum()>0:
            print(f"--->La columna {col.upper()} tiene valores nulos")
        df_counts = pd.DataFrame({"count": dataframe[col].value_counts(dropna=False),"porcentaje (%)": (dataframe[col].value_counts(dropna=False, normalize=True) * 100).round(3)})
        display(df_counts)
        print("\n ------------------------------- \n")

def detector_columnas_categoricas(df,valores_unicos):
    """
    Devuelve las columnas numéricas con menos de 10 valores únicos.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        valores_unicos: Número de valores

    Returns:
        list: Lista de nombres de columnas numéricas con menos de x valores únicos.
    """
    numerical_columns = df.select_dtypes(include=['number'])  # Seleccionar columnas numéricas
    columns_with_few_unique = [
        col for col in numerical_columns.columns if numerical_columns[col].nunique() <= valores_unicos
    ]
    return columns_with_few_unique


def plot_numericas(dataframe):
    df_num=dataframe.select_dtypes(include=np.number)
    num_filas=math.ceil(len(df_num.columns)/2)
    fig, axes=plt.subplots(nrows=num_filas, ncols=2,figsize=(15,10))
    axes=axes.flat

    for indice, columna in enumerate(df_num.columns):
        sns.histplot(x=columna, data=df_num, ax=axes[indice])
        axes[indice].set_title(columna)
        axes[indice].set_xlabel("")

    if len(df_num.columns)%2!=0:
        fig.delaxes(axes[-1])  
    else:
        pass

    plt.tight_layout()


def plot_categoricas(dataframe, paleta="mako", max_categories=10):
    # Seleccionar solo columnas categóricas
    df_cat = dataframe.select_dtypes(include="O")
    
    # Filtrar columnas con menos de `max_categories` categorías únicas
    filtered_columns = [col for col in df_cat.columns if dataframe[col].nunique() <= max_categories]
    df_cat = df_cat[filtered_columns]
    
    num_columnas = 2  # Fijar el número de columnas por fila
    num_filas = math.ceil(len(df_cat.columns) / num_columnas)
    
    # Ajustar el tamaño de los gráficos
    fig, axes = plt.subplots(nrows=num_filas, ncols=num_columnas, figsize=(14, num_filas * 5))
    axes = axes.flat

    for indice, columna in enumerate(df_cat.columns):
        # Contar valores por categoría (incluidos nulos)
        category_counts = df_cat[columna].value_counts(dropna=False)
        
        # Separar los valores nulos como categoría aparte
        null_counts = df_cat[columna].isnull().sum()  # Total de nulos en la columna
        null_series = pd.Series(null_counts, index=["Nulos"]) if null_counts > 0 else pd.Series()

        # Combinar conteo de categorías con los valores nulos
        combined_counts = pd.concat([category_counts, null_series])
        
        # Crear el gráfico
        sns.barplot(
            x=combined_counts.index.astype(str),  # Aseguramos que sean strings
            y=combined_counts.values, 
            ax=axes[indice], 
            palette=paleta
        )
        
        # Configuración del gráfico
        axes[indice].set_title(columna, fontsize=14, weight="bold")  # Títulos más visibles
        axes[indice].set_xlabel("")  # Ocultar etiquetas del eje X
        axes[indice].tick_params(axis='x', rotation=45, labelsize=10)  # Rotar etiquetas
        axes[indice].set_ylabel("Count", fontsize=12)

    # Eliminar ejes vacíos si el número de gráficos no es par
    for i in range(len(df_cat.columns), len(axes)):
        fig.delaxes(axes[i])
    
    # Ajustar el espaciado entre subgráficos
    plt.tight_layout(pad=3.0)
    plt.suptitle("Análisis de Variables Categóricas (Incluyendo Nulos, ≤ {} Categorías)".format(max_categories), 
                 fontsize=18, weight="bold", y=1.02)
    
    plt.show()

def ANOVA(df, categorical_column):
    """
    Realiza un ANOVA entre una variable categórica y una lista de variables numéricas.
    
    Parameters:
        df (pd.DataFrame): El DataFrame con los datos.
        categorical_column (str): El nombre de la columna categórica.
            
    Returns:
        dict: Resultados del ANOVA para cada variable numérica.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    results = {}
    for variable in numeric_columns:
        formula = f'{variable} ~ C({categorical_column})'  # Fórmula para ANOVA
        try:
            model = ols(formula, data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)  # Calcular ANOVA
            results[variable] = anova_table
        except Exception as e:
            results[variable] = f"Error al procesar {variable}: {e}"
    
    return results


def relacion_vr_categoricas_heatmap(dataframe, variable_respuesta, max_categories=10, cmap="Blues"):
    # Seleccionar columnas categóricas con menos de `max_categories` categorías únicas
    df_cat = dataframe.select_dtypes(include="object")
    filtered_columns = [col for col in df_cat.columns if dataframe[col].nunique() <= max_categories and col != variable_respuesta]

    if not filtered_columns:
        print(f"No hay columnas categóricas con menos de {max_categories} categorías para graficar.")
        return

    for columna in filtered_columns:
        # Crear una tabla de frecuencias cruzadas
        crosstab = pd.crosstab(dataframe[variable_respuesta], dataframe[columna])

        # Crear el mapa de calor
        plt.figure(figsize=(8, 5))
        sns.heatmap(crosstab, annot=True, fmt="d", cmap=cmap, cbar=False)

        # Configuración del gráfico
        plt.title(f"Frecuencias cruzadas entre '{variable_respuesta}' y '{columna}'", fontsize=14, weight="bold")
        plt.xlabel(columna, fontsize=12)
        plt.ylabel(variable_respuesta, fontsize=12)
        plt.tight_layout()
        plt.show()

def relacion_vr_categoricas_barras_agrupadas(dataframe, variable_respuesta, max_categories=10, palette="Set2"):
    # Seleccionar columnas categóricas con menos de `max_categories` categorías únicas
    df_cat = dataframe.select_dtypes(include="object")
    filtered_columns = [col for col in df_cat.columns if dataframe[col].nunique() <= max_categories and col != variable_respuesta]

    if not filtered_columns:
        print(f"No hay columnas categóricas con menos de {max_categories} categorías para graficar.")
        return

    for columna in filtered_columns:
        # Crear una tabla de frecuencias cruzadas
        crosstab = pd.crosstab(dataframe[variable_respuesta], dataframe[columna])

        # Crear el gráfico de barras agrupadas
        crosstab.plot(kind="bar", figsize=(8, 5), colormap=palette)
        
        # Configuración del gráfico
        plt.title(f"Frecuencias absolutas entre '{variable_respuesta}' y '{columna}'", fontsize=14, weight="bold")
        plt.xlabel(variable_respuesta, fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.legend(title=columna, bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


def relacion_vr_numericas_boxplot(dataframe, variable_respuesta, paleta="Set2"):
    # Seleccionar columnas numéricas
    df_num = dataframe.select_dtypes(include="number")
    
    if df_num.empty:
        print("No hay columnas numéricas para graficar.")
        return

    for columna in df_num.columns:
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            x=dataframe[variable_respuesta], 
            y=dataframe[columna], 
            palette=paleta
        )
        
        plt.title(f"Distribución de '{columna}' por '{variable_respuesta}'", fontsize=14, weight="bold")
        plt.xlabel(variable_respuesta, fontsize=12)
        plt.ylabel(columna, fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()

def relacion_vr_numericas_violinplot(dataframe, variable_respuesta, paleta="Set2"):
    # Seleccionar columnas numéricas
    df_num = dataframe.select_dtypes(include="number")
    
    if df_num.empty:
        print("No hay columnas numéricas para graficar.")
        return

    for columna in df_num.columns:
        plt.figure(figsize=(8, 5))
        sns.violinplot(
            x=dataframe[variable_respuesta], 
            y=dataframe[columna], 
            palette=paleta, 
            inner="box"
        )
        
        plt.title(f"Densidad y distribución de '{columna}' por '{variable_respuesta}'", fontsize=14, weight="bold")
        plt.xlabel(variable_respuesta, fontsize=12)
        plt.ylabel(columna, fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()
        plt.show()


def matriz_correlacion(dataframe):
    matriz_corr=dataframe.corr(numeric_only=True)
    mascara=np.triu(np.ones_like(matriz_corr,dtype=np.bool_))
    sns.heatmap(matriz_corr,
                annot=True,
                vmin=1,
                vmax=-1,
                mask=mascara,
                cmap="seismic")
    plt.figure(figsize=(10,15))
    plt.tight_layout()


def comparador_estaditicos(df_list, names=None):
    # Obtener las columnas en común entre todos los DataFrames
    common_columns = set(df_list[0].columns)
    for df in df_list[1:]:
        common_columns &= set(df.columns)
    common_columns = list(common_columns)

    # Lista para almacenar cada DataFrame descriptivo
    descriptive_dfs = []

    # Genera descripciones para cada DataFrame y las almacena
    for i, df in enumerate(df_list):
        desc_df = df[common_columns].describe().T  # Transpone y usa solo las columnas comunes
        desc_df['DataFrame'] = names[i] if names else f'DF_{i+1}'
        descriptive_dfs.append(desc_df)

    # Combina todos los DataFrames descriptivos en uno solo
    comparative_df = pd.concat(descriptive_dfs)
    comparative_df = comparative_df.set_index(['DataFrame', comparative_df.index])  # Índice jerárquico

    # Encuentra las diferencias por fila (compara cada estadística entre DataFrames)
    diff_df = comparative_df.groupby(level=1).apply(lambda x: x.nunique() > 1).any(axis=1)

    # Filtra solo las filas que tengan diferencias y verifica que los índices existen
    available_indices = comparative_df.index.get_level_values(1).unique()
    indices_with_diff = [index for index in diff_df[diff_df].index if index in available_indices]

    comparative_df_diff = comparative_df.loc[(slice(None), indices_with_diff), :]

    return comparative_df_diff

def plot_top5_and_bottom5(df, label_column):
    """
    Genera gráficos de barras mostrando el Top 5 y el Bottom 5 valores de cada columna numérica,
    usando una columna específica como etiquetas. Los gráficos se presentan lado a lado,
    cada uno con su propia escala en el eje Y.

    :param df: DataFrame con los datos.
    :param label_column: Nombre de la columna que se usará como etiquetas para las barras.
    """
    if label_column not in df.columns:
        raise ValueError(f"La columna '{label_column}' no existe en el DataFrame.")
    
    # Paleta de colores fija
    colors_top = ["#4CAF50", "#FFC107", "#03A9F4", "#E91E63", "#9C27B0"]
    colors_bottom = ["#FF5722", "#8BC34A", "#03A9F4", "#FFEB3B", "#795548"]

    # Iterar sobre las columnas numéricas del DataFrame
    for column in df.select_dtypes(include=['number']).columns:
        if column == label_column:
            continue  # Saltar la columna usada para etiquetas si es numérica

        # Obtener los 5 valores más altos y más bajos
        top5 = df.nlargest(5, column)
        bottom5 = df.nsmallest(5, column)

        # Crear gráficos lado a lado
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Gráfico de Top 5
        axes[0].bar(top5[label_column].astype(str), top5[column], color=colors_top[:len(top5)])
        axes[0].set_title(f'Top 5 valores en columna: {column}')
        axes[0].set_xlabel(label_column)
        axes[0].set_ylabel(column)
        axes[0].tick_params(axis='x', rotation=45)

        # Gráfico de Bottom 5
        axes[1].bar(bottom5[label_column].astype(str), bottom5[column], color=colors_bottom[:len(bottom5)])
        axes[1].set_title(f'Bottom 5 valores en columna: {column}')
        axes[1].set_xlabel(label_column)
        axes[1].tick_params(axis='x', rotation=45)

        # Ajustar diseño
        plt.tight_layout()
        plt.show()

# GESTION OUTIERS
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def separar_dataframe(dataframe):
    return dataframe.select_dtypes(include = np.number), dataframe.select_dtypes(include = "O")

def obtener_outliers_iqr(df, columna,n): #Se usa para ver los outiers por su IQR
    """
    Identifica los valores outliers en base al rango intercuartílico (IQR) de una columna.

    Parámetros:
    - df: DataFrame
    - columna: Nombre de la columna a analizar (str)
    - n: Valor multiplicador del IQR (Cuanto mayor sea menos datos te dara la funcion por que sera mas grande el rango de datos)
    
    Retorna:
    - DataFrame con los valores outliers
    """
    # Calcular el rango intercuartílico (IQR)
    Q1 = df[columna].quantile(0.25)  # Primer cuartil
    Q3 = df[columna].quantile(0.75)  # Tercer cuartil
    IQR = Q3 - Q1

    # Definir los límites
    limite_inferior = Q1 - n * IQR
    limite_superior = Q3 + n * IQR

    # Identificar las filas con valores outliers
    outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
    
    return outliers

def detectar_outliers(dataframe, color="orange", tamaño_grafica=(15,10)): #Genera boxplts de las variables numericas del df
    df_num = dataframe.select_dtypes(include=np.number)
    num_filas = math.ceil(len(df_num.columns) / 2)
    fig, axes = plt.subplots(nrows=num_filas, ncols=2, figsize=tamaño_grafica)
    axes = axes.flat

    # Configuración de los outliers en color naranja
    flierprops = dict(marker='o', markerfacecolor='orange', markersize=5, linestyle='none')

    for indice, columna in enumerate(df_num.columns):
        sns.boxplot(x=columna,
                    data=df_num,
                    ax=axes[indice],
                    color=color,
                    flierprops=flierprops)  # Aplica color naranja a los outliers
        axes[indice].set_title(f"Outliers de {columna}")
        axes[indice].set_xlabel("")

    # Eliminar el último subplot si el número de columnas es impar
    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()

def scatterplot_outliers(dataframe, combinaciones_variables, columnas_hue, palette="Set1", alpha=0.5):
    """
    Visualización mejorada de gráficos de dispersión para analizar outliers.
    """
    for col_hue in columnas_hue:
        num_combinaciones = len(combinaciones_variables)
        num_filas = math.ceil(num_combinaciones / 3)  # Ajustar automáticamente el número de filas
        fig, axes = plt.subplots(ncols=3, nrows=num_filas, figsize=(15, 5 * num_filas))
        axes = axes.flat

        for indice, tupla in enumerate(combinaciones_variables):
            sns.scatterplot(
                data=dataframe,
                x=tupla[0],
                y=tupla[1],
                ax=axes[indice],
                hue=col_hue,
                palette=palette,
                style=col_hue,
                alpha=alpha
            )
            axes[indice].set_title(f"{tupla[0]} vs {tupla[1]} (hue: {col_hue})", fontsize=10)
            axes[indice].tick_params(axis='x', rotation=45)

        # Ocultar ejes vacíos si sobran
        for i in range(len(combinaciones_variables), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(f"Scatterplots con hue: {col_hue}", fontsize=16)
        plt.tight_layout()
        plt.show()

def gestion_nulos_lof(df, col_numericas, list_neighbors, lista_contaminacion):
    """
    Aplica el algoritmo LOF (Local Outlier Factor) para detectar outliers en las columnas numéricas del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        col_numericas (list): Lista de nombres de columnas numéricas sobre las que aplicar LOF.
        list_neighbors (list): Lista de valores para el número de vecinos (`n_neighbors`).
        lista_contaminacion (list): Lista de valores para la tasa de contaminación (`contamination`).
    
    Returns:
        pd.DataFrame: DataFrame con nuevas columnas que indican outliers (-1) o inliers (1) para cada combinación de parámetros.
    """
    # Validar si las columnas numéricas existen en el DataFrame
    missing_columns = [col for col in col_numericas if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Las columnas {missing_columns} no están presentes en el DataFrame.")

    # Generar combinaciones de parámetros
    combinaciones = list(product(list_neighbors, lista_contaminacion))
    
    # Progresión y cálculo de LOF para cada combinación
    for neighbors, contaminacion in tqdm(combinaciones, desc="Aplicando LOF con diferentes parámetros"):
        # Inicializar el modelo LOF
        lof = LocalOutlierFactor(
            n_neighbors=neighbors, 
            contamination=contaminacion,
            n_jobs=-1
        )
        
        # Crear una nueva columna para la combinación de parámetros
        columna_nombre = f"outliers_lof_{neighbors}_{contaminacion}"
        df[columna_nombre] = lof.fit_predict(df[col_numericas])
    
    return df

def detectar_outliers_categoricos(dataframe, threshold=0.05):
    """
    Detecta valores categóricos raros (outliers) en variables categóricas basándose en su frecuencia en el DataFrame.
    
    Args:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        threshold (float): Umbral mínimo de frecuencia relativa para considerar un valor como no raro.
                           Valores por debajo de este umbral se consideran raros.
    
    Returns:
        dict: Un diccionario donde las claves son las columnas categóricas y los valores son listas de categorías raras.
    """
    # Seleccionar solo columnas categóricas
    columnas_categoricas = dataframe.select_dtypes(include='object').columns
    
    # Diccionario para almacenar los valores categóricos raros
    outliers_categoricos = {}
    
    for columna in columnas_categoricas:
        # Calcular la frecuencia relativa de cada categoría
        frecuencias = dataframe[columna].value_counts(normalize=True)
        
        # Filtrar las categorías con frecuencia menor al umbral
        valores_raros = frecuencias[frecuencias < threshold].index.tolist()
        
        # Almacenar los valores raros si existen
        if valores_raros:
            outliers_categoricos[columna] = valores_raros
    
    return outliers_categoricos

    # Filtrar el DataFrame
def filtrar_por_alguna_condicion(dataframe, condiciones):
    filtro = pd.Series(False, index=dataframe.index)  # Empezar con un filtro "falso"
    
    for columna, valores in condiciones.items():
        if columna in dataframe.columns:
            # Aplicar la condición para cada columna
            filtro |= dataframe[columna].isin(valores)
        else:
            print(f"Advertencia: La columna '{columna}' no está en el DataFrame.")
    
    # Aplicar el filtro al DataFrame
    return dataframe[filtro]

def generador_boxplots(df_list):
    # Filtra los DataFrames válidos
    df_list = [df for df in df_list if isinstance(df, pd.DataFrame)]
    
    if not df_list:
        print("Error: La lista no contiene DataFrames válidos.")
        return

    # Define los sufijos deseados
    sufijos_deseados = ('_stds', '_norm', '_minmax', '_robust')

    # Filtra las columnas de cada DataFrame para incluir solo las que tienen los sufijos deseados
    filtered_df_list = [df[[col for col in df.columns if col.endswith(sufijos_deseados)]] for df in df_list]

    # Configura la figura con una fila de subplots por DataFrame
    fig, axes = plt.subplots(nrows=len(filtered_df_list), ncols=max(len(df.columns) for df in filtered_df_list),
                             figsize=(6 * max(len(df.columns) for df in filtered_df_list), 4 * len(filtered_df_list)),
                             squeeze=False)  # Squeeze=False asegura una matriz 2D

    # Itera sobre cada DataFrame filtrado y cada columna
    for df_idx, df in enumerate(filtered_df_list):
        for col_idx, column in enumerate(df.columns):
            sns.boxplot(x=column, data=df, ax=axes[df_idx][col_idx])
            axes[df_idx][col_idx].set_title(f"DF {df_idx + 1} - {column}")

    # Oculta los ejes vacíos si hay menos columnas en algún DataFrame
    for df_idx, ax_row in enumerate(axes):
        for col_idx in range(len(filtered_df_list[df_idx].columns), axes.shape[1]):
            ax_row[col_idx].axis('off')

    # Ajuste de espaciado entre subplots
    plt.tight_layout()
    plt.show()

# FASE DE ENCODING
# -------------------------------------------------------------------------------------------------------------------------------------------------------

def detectar_orden_cat(df, var_res):
    """
    Detecta si las variables categóricas tienen un orden significativo en relación con una variable objetivo.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - var_res: Nombre de la variable respuesta (objetivo).

    Imprime los resultados de cada variable y muestra dos listas finales:
    - Categorías con orden significativo.
    - Categorías sin orden significativo.
    """
    
    # Listas para almacenar los resultados
    categorias_con_orden = []  # Variables categóricas que tienen un orden significativo
    categorias_sin_orden = []  # Variables categóricas que no tienen un orden significativo

    # Seleccionar las columnas categóricas del DataFrame
    lista_categoricas=df.select_dtypes(["object", "category"])

    # Iterar sobre cada columna categórica
    for categorica in lista_categoricas:
        print(f"Estamos evaluando la variable: {categorica.upper()}")
        
        # Crear una tabla cruzada entre la variable categórica y la variable respuesta
        df_cross_tab = pd.crosstab(df[categorica], df[var_res])
        display(df_cross_tab)  # Mostrar la tabla cruzada

        # Realizar la prueba de chi-cuadrado para evaluar independencia
        chi2, p, dof, expected = chi2_contingency(df_cross_tab)

        # Evaluar el p-valor para determinar si hay orden
        if p < 0.05:  # Si el p-valor es menor a 0.05, rechazamos la hipótesis de independencia
            print(f"La variable categórica {categorica.upper()} sí tiene orden\n")
            categorias_con_orden.append(categorica)  # Agregar a la lista de categorías con orden
        else:
            print(f"La variable categórica {categorica.upper()} no tiene orden\n")
            categorias_sin_orden.append(categorica)  # Agregar a la lista de categorías sin orden

    # Imprimir los resultados finales
    print("\n=== Resultados finales ===")
    print("Categorías con orden:")
    print(categorias_con_orden)  # Lista de categorías con orden significativo
    print("\nCategorías sin orden:")
    print(categorias_sin_orden)  # Lista de categorías sin orden significativo

def one_hot_encoding(dataframe, columns):
    """
    Realiza codificación one-hot en las columnas especificadas y guarda el encoder en un archivo .pkl.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.

    Returns:
        pd.DataFrame: DataFrame con codificación one-hot aplicada.
    """
    # Inicializar OneHotEncoder
    one_hot_encoder = OneHotEncoder(sparse_output=False)

    # Aplicar codificación one-hot
    trans_one_hot = one_hot_encoder.fit_transform(dataframe[columns])
    oh_df = pd.DataFrame(trans_one_hot, columns=one_hot_encoder.get_feature_names_out(columns))
    
    # Combinar el DataFrame original con el DataFrame codificado
    dataframe = pd.concat([dataframe.reset_index(drop=True), oh_df.reset_index(drop=True)], axis=1)

    # Eliminar las columnas originales
    dataframe.drop(columns=columns, inplace=True)
    encoder_path="../transformers/one_hot_encoder.pkl"
    # Guardar el encoder en un archivo .pkl
    with open(encoder_path, 'wb') as file:
        pickle.dump(one_hot_encoder, file)
    print(f"Encoder guardado en: {encoder_path}")
    return dataframe

def ordinal_encoding(dataframe, columns, categories):
    """
    Realiza codificación ordinal en las columnas especificadas y guarda el encoder en un archivo .pkl.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.
        categories (list of list): Lista de listas con las categorías en orden.

    Returns:
        pd.DataFrame: DataFrame con codificación ordinal aplicada.
    """
    # Inicializar OrdinalEncoder
    ordinal_encoder = OrdinalEncoder(categories=categories, dtype=float, handle_unknown="use_encoded_value", unknown_value=np.nan)

    # Codificar las columnas
    dataframe[columns] = ordinal_encoder.fit_transform(dataframe[columns])
    encoder_path="../transformers/ordinal_encoder.pkl"
    # Guardar el encoder en un archivo .pkl
    with open(encoder_path, 'wb') as file:
        pickle.dump(ordinal_encoder, file)
    print(f"Encoder guardado en: {encoder_path}")

    return dataframe


def label_encoding(dataframe, columns):
    """
    Realiza codificación label en las columnas especificadas y guarda el encoder en un archivo .pkl.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.

    Returns:
        pd.DataFrame: DataFrame con codificación label aplicada.
    """
    # Diccionario para guardar el LabelEncoder para cada columna
    label_encoders = {}

    # Aplicar codificación label
    for col in columns:
        label_encoder = LabelEncoder()
        dataframe[col] = label_encoder.fit_transform(dataframe[col])
        label_encoders[col] = label_encoder  # Guardar el encoder para esta columna
    encoder_path="../transformers/label_encoders.pkl"
    # Guardar los LabelEncoders en un archivo .pkl
    with open(encoder_path, 'wb') as file:
        pickle.dump(label_encoders, file)
    print(f"Encoders guardados en: {encoder_path}")

    return dataframe


def target_encoding(dataframe, columns, target):
    """
    Realiza codificación target en las columnas especificadas y guarda el encoder en un archivo .pkl.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.
        target (str): Nombre de la variable objetivo.

    Returns:
        pd.DataFrame: DataFrame con codificación target aplicada.
    """
    # Verificar que las columnas y el target existen en el DataFrame
    if not set(columns).issubset(dataframe.columns):
        raise ValueError(f"Algunas columnas de {columns} no están en el DataFrame.")
    if target not in dataframe.columns:
        raise ValueError(f"La variable objetivo '{target}' no está en el DataFrame.")

    # Inicializar TargetEncoder
    target_encoder = TargetEncoder(cols=columns)

    # Codificar las columnas
    dataframe[columns] = target_encoder.fit_transform(dataframe[columns], dataframe[target])
    encoder_path="../transformers/target_encoding.pkl"
    # Guardar el encoder en un archivo .pkl
    with open(encoder_path, 'wb') as file:
        pickle.dump(target_encoder, file)
    print(f"Encoder guardado en: {encoder_path}")

    return dataframe


def frequency_encoding(dataframe, columns):
    """
    Realiza codificación de frecuencia en las columnas especificadas y guarda los mapeos en un archivo .pkl.

    Args:
        dataframe (pd.DataFrame): DataFrame de pandas.
        columns (list): Lista de columnas a codificar.

    Returns:
        pd.DataFrame: DataFrame con codificación de frecuencia aplicada.
    """
    # Diccionario para almacenar los mapeos de frecuencia
    freq_mappings = {}

    # Aplicar codificación de frecuencia
    for col in columns:
        freq_map = dataframe[col].value_counts(normalize=True)
        freq_mappings[col] = freq_map  # Guardar el mapeo
        dataframe[col] = dataframe[col].map(freq_map)
    encoder_path="../transformers/freq_mappings.pkl"
    # Guardar los mapeos de frecuencia en un archivo .pkl
    with open(encoder_path, 'wb') as file:
        pickle.dump(freq_mappings, file)
    print(f"Mapeos de frecuencia guardados en: {encoder_path}")

    return dataframe

# BALANCEO 

def balancear_datos_con_smote(dataframe, variable_respuesta, ruta_archivo):
    """
    Aplica SMOTE para balancear las clases en la variable objetivo y guarda los datos balanceados.

    Parámetros:
        dataframe (pd.DataFrame): Dataset encodeado con desbalanceo en la variable objetivo.
        variable_respuesta (str): Nombre de la columna objetivo.
        ruta_archivo (str): Ruta del archivo donde se guardará el dataset balanceado en formato .pkl.
    
    Retorna:
        None: Guarda el dataset balanceado en un archivo .pkl.
    """
    # Separar características (X) y la variable objetivo (y)
    X = dataframe.drop(columns=[variable_respuesta])
    y = dataframe[variable_respuesta]
    
    # Aplicar SMOTE
    print("Distribución antes de SMOTE:", Counter(y))
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print("Distribución después de SMOTE:", Counter(y_balanced))
    
    # Combinar las características balanceadas con la variable objetivo
    df_balanced = pd.concat([pd.DataFrame(X_balanced, columns=X.columns), 
                             pd.DataFrame(y_balanced, columns=[variable_respuesta])], axis=1)
    
    # Guardar el dataset balanceado en un archivo .pkl
    with open(ruta_archivo, 'wb') as file:
        pickle.dump(df_balanced, file)
    print(f"Dataset balanceado guardado en: {ruta_archivo}")
