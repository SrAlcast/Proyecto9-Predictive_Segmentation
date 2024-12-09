# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otras utilidades
# -----------------------------------------------------------------------
import math
from tqdm import tqdm

# Para las visualizaciones
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesado y modelado
# -----------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler


# Sacar número de clusters y métricas
# -----------------------------------------------------------------------
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Modelos de clustering
# -----------------------------------------------------------------------
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering


# Para visualizar los dendrogramas
# -----------------------------------------------------------------------
import scipy.cluster.hierarchy as sch

from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
)

class Clustering:
    """
    Clase para realizar varios métodos de clustering en un DataFrame.

    Atributos:
        - dataframe : pd.DataFrame. El conjunto de datos sobre el cual se aplicarán los métodos de clustering.
    """
    
    def __init__(self, dataframe):
        """
        Inicializa la clase Clustering con un DataFrame.

        Params:
            - dataframe : pd.DataFrame. El DataFrame que contiene los datos a los que se les aplicarán los métodos de clustering.
        """
        self.dataframe = dataframe

    def elbow_method(self, cluster_range=range(1, 11)):
        """
        Aplica el método del codo para determinar el número óptimo de clusters.

        Params:
            - cluster_range : range. Rango de números de clusters a evaluar.

        Returns:
            - inertia : list. Lista de inercia para cada número de clusters.
        """
        inertia = []
        for k in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.dataframe)
            inertia.append(kmeans.inertia_)
        
        # Gráfico del método del codo
        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, inertia, marker='o')
        plt.title('Método del Codo')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Inercia')
        plt.show()
        
        return inertia

    def silhouette_method(self, cluster_range=range(2, 11)):
        """
        Calcula el coeficiente de silueta para diferentes números de clusters.

        Params:
            - cluster_range : range. Rango de números de clusters a evaluar.

        Returns:
            - silhouette_scores : list. Lista de puntajes de silueta para cada número de clusters.
        """
        silhouette_scores = []
        for k in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.dataframe)
            score = silhouette_score(self.dataframe, kmeans.labels_)
            silhouette_scores.append(score)
        
        # Gráfico del coeficiente de silueta
        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, silhouette_scores, marker='o')
        plt.title('Coeficiente de Silueta')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Silhouette Score')
        plt.show()
        
        return silhouette_scores

    def davies_bouldin_method(self, cluster_range=range(2, 11)):
        """
        Calcula el índice de Davies-Bouldin para diferentes números de clusters.

        Params:
            - cluster_range : range. Rango de números de clusters a evaluar.

        Returns:
            - db_scores : list. Lista de puntajes de Davies-Bouldin para cada número de clusters.
        """
        db_scores = []
        for k in tqdm(cluster_range):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(self.dataframe)
            score = davies_bouldin_score(self.dataframe, kmeans.labels_)
            db_scores.append(score)
        
        # Gráfico del índice de Davies-Bouldin
        plt.figure(figsize=(8, 5))
        plt.plot(cluster_range, db_scores, marker='o')
        plt.title('Índice de Davies-Bouldin')
        plt.xlabel('Número de Clusters')
        plt.ylabel('Davies-Bouldin Score')
        plt.show()
        
        return db_scores
    
    def visualizar_dendrograma(self, metodo="average"):
        """
        Genera y visualiza dendrogramas para el conjunto de datos utilizando diferentes métodos de distancias.

        Params:
            - metodo : str, optional, default: ["average", "complete", "ward"]. Lista de métodos para calcular las distancias entre los clusters. Cada método generará un dendrograma
                en un subplot diferente.

        Returns:
            None
        """
        sch.dendrogram(
            sch.linkage(self.dataframe, method=metodo),
            labels=self.dataframe.index, 
            leaf_rotation=90, leaf_font_size=4
        )
        plt.title(f'Dendrograma usando {metodo}')
        plt.xlabel('Muestras')
        plt.ylabel('Distancias')
        plt.show()    

    def modelo_kmeans(self, dataframe_original, num_clusters):
        """
        Aplica KMeans al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - num_clusters : int. Número de clusters a formar.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        kmeans = KMeans(n_clusters=num_clusters)
        km_fit = kmeans.fit(self.dataframe)
        labels = km_fit.labels_
        dataframe_original["clusters_kmeans"] = labels.astype(str)
        return dataframe_original,labels
    
    def modelo_balanced_kmeans_min(self, dataframe_original, num_clusters, size_min):
        """
        Aplica Balanced KMeans al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - num_clusters : int. Número de clusters a formar.
            - size_min : int. Tamaño mínimo permitido por cluster.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
            - labels : np.array. Etiquetas de los clusters.
        """
        # Crear modelo Balanced KMeans con solo tamaño mínimo
        kmeans = KMeansConstrained(
            n_clusters=num_clusters,
            size_min=size_min,
            size_max=len(dataframe_original),  # Sin límite superior
            random_state=42
        )
        # Ajustar el modelo a los datos
        km_fit = kmeans.fit(self.dataframe)
        labels = km_fit.labels_

        # Añadir etiquetas al DataFrame original
        dataframe_original["clusters_balanced_kmeans_min"] = labels.astype(str)
        return dataframe_original, labels
    
    def modelo_aglomerativo(self, num_clusters, metodo_distancias, dataframe_original):
        """
        Aplica clustering aglomerativo al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - num_clusters : int. Número de clusters a formar.
            - metodo_distancias : str. Método para calcular las distancias entre los clusters.
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        modelo = AgglomerativeClustering(
            linkage=metodo_distancias,
            distance_threshold=None,
            n_clusters=num_clusters
        )
        aglo_fit = modelo.fit(self.dataframe)
        labels = aglo_fit.labels_
        dataframe_original["clusters_agglomerative"] = labels.astype(str)
        return dataframe_original
    
    def modelo_divisivo(self, dataframe_original, threshold=0.5, max_clusters=5):
        """
        Implementa el clustering jerárquico divisivo.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - threshold : float, optional, default: 0.5. Umbral para decidir cuándo dividir un cluster.
            - max_clusters : int, optional, default: 5. Número máximo de clusters deseados.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de los clusters.
        """
        def divisive_clustering(data, current_cluster, cluster_labels):
            # Si el número de clusters actuales es mayor o igual al máximo permitido, detener la división
            if len(set(current_cluster)) >= max_clusters:
                return current_cluster

            # Aplicar KMeans con 2 clusters
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(data)
            labels = kmeans.labels_

            # Calcular la métrica de silueta para evaluar la calidad del clustering
            silhouette_avg = silhouette_score(data, labels)

            # Si la calidad del clustering es menor que el umbral o si el número de clusters excede el máximo, detener la división
            if silhouette_avg < threshold or len(set(current_cluster)) + 1 > max_clusters:
                return current_cluster

            # Crear nuevas etiquetas de clusters
            new_cluster_labels = current_cluster.copy()
            max_label = max(current_cluster)

            # Asignar nuevas etiquetas incrementadas para cada subcluster
            for label in set(labels):
                cluster_indices = np.where(labels == label)[0]
                new_label = max_label + 1 + label
                new_cluster_labels[cluster_indices] = new_label

            # Aplicar recursión para seguir dividiendo los subclusters
            for new_label in set(new_cluster_labels):
                cluster_indices = np.where(new_cluster_labels == new_label)[0]
                new_cluster_labels = divisive_clustering(data[cluster_indices], new_cluster_labels, new_cluster_labels)

            return new_cluster_labels

        # Inicializar las etiquetas de clusters con ceros
        initial_labels = np.zeros(len(self.dataframe))

        # Llamar a la función recursiva para iniciar el clustering divisivo
        final_labels = divisive_clustering(self.dataframe.values, initial_labels, initial_labels)

        # Añadir las etiquetas de clusters al DataFrame original
        dataframe_original["clusters_divisive"] = final_labels.astype(int).astype(str)

        return dataframe_original,final_labels

    def modelo_espectral(self, dataframe_original, n_clusters=3, assign_labels='kmeans'):
        """
        Aplica clustering espectral al DataFrame y añade las etiquetas de clusters al DataFrame original.

        Params:
            - dataframe_original : pd.DataFrame. El DataFrame original al que se le añadirán las etiquetas de clusters.
            - n_clusters : int, optional, default: 3. Número de clusters a formar.
            - assign_labels : str, optional, default: 'kmeans'. Método para asignar etiquetas a los puntos. Puede ser 'kmeans' o 'discretize'.

        Returns:
            - pd.DataFrame. El DataFrame original con una nueva columna para las etiquetas de clusters.
        """
        spectral = SpectralClustering(n_clusters=n_clusters, assign_labels=assign_labels, random_state=0)
        labels = spectral.fit_predict(self.dataframe)
        dataframe_original["clusters_spectral"] = labels.astype(str)
        return dataframe_original,labels
    
    def modelo_dbscan(self, dataframe_original, eps_values=[0.5, 1.0, 1.5], min_samples_values=[3, 2, 1]):
        """
        Aplica DBSCAN al DataFrame y añade las etiquetas de clusters al DataFrame original.
        Optimiza parámetros según la métrica de silueta, ignorando ruido.

        Params:
            - dataframe_original (pd.DataFrame): DataFrame original al que se añadirán las etiquetas de clusters.
            - eps_values (list of float, optional): Lista de valores para el parámetro eps.
            - min_samples_values (list of int, optional): Lista de valores para el parámetro min_samples.

        Returns:
            - pd.DataFrame: DataFrame original con una nueva columna "clusters_dbscan".
            - dict: Diccionario con los mejores parámetros y métricas.
        """
        # Escalar los datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.dataframe)

        # Variables para guardar el mejor resultado
        best_eps = None
        best_min_samples = None
        best_silhouette = -1  # Inicializar en -1 ya que el rango es [-1, 1]

        # Iterar sobre combinaciones de eps y min_samples
        for eps in eps_values:
            for min_samples in min_samples_values:
                # Aplicar DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data_scaled)

                # Ignorar casos con un solo cluster o solo ruido
                unique_labels = set(labels)
                if len(unique_labels) <= 1 or (len(unique_labels) == 2 and -1 in unique_labels):
                    continue

                # Calcular métrica de silueta, excluyendo el ruido (-1)
                mask = labels != -1
                if mask.sum() > 1:  # Al menos dos puntos deben ser considerados
                    silhouette = silhouette_score(data_scaled[mask], labels[mask])
                else:
                    silhouette = -1

                # Actualizar el mejor resultado si es necesario
                if silhouette > best_silhouette:
                    best_silhouette = silhouette
                    best_eps = eps
                    best_min_samples = min_samples

        # Aplicar DBSCAN con los mejores parámetros
        best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        best_labels = best_dbscan.fit_predict(data_scaled)

        # Añadir las etiquetas al DataFrame original
        dataframe_original["clusters_dbscan"] = best_labels

        # Diccionario con los mejores resultados
        resultados = {
            "best_eps": best_eps,
            "best_min_samples": best_min_samples,
            "best_silhouette": best_silhouette,
            "unique_clusters": len(set(best_labels)) - (1 if -1 in best_labels else 0),  # Excluir ruido
        }

        return dataframe_original, resultados,best_labels

    def calcular_metricas(self, labels: np.ndarray):
        """
        Calcula métricas de evaluación del clustering.
        """
        if len(set(labels)) <= 1:
            raise ValueError("El clustering debe tener al menos 2 clusters para calcular las métricas.")

        silhouette = silhouette_score(self.dataframe, labels)
        davies_bouldin = davies_bouldin_score(self.dataframe, labels)

        unique, counts = np.unique(labels, return_counts=True)
        cardinalidad = dict(zip(unique, counts))

        return pd.DataFrame({
            "silhouette_score": silhouette,
            "davies_bouldin_index": davies_bouldin,
            "cardinalidad": cardinalidad
        }, index = [0])

def graficar_clusters(dataframe, cluster_col='Cluster', figsize=(18, 6), palette="tab10"):
    """
    Genera gráficos de barras para variables categóricas (conteos) y numéricas (promedios) por cluster,
    con las variables ordenadas de mayor a menor.

    Parameters:
        - dataframe (pd.DataFrame): Conjunto de datos que incluye las etiquetas de clusters.
        - cluster_col (str): Nombre de la columna que contiene las etiquetas de clusters.
        - figsize (tuple): Tamaño base de cada gráfica.
        - palette (str): Paleta de colores para las barras (compatible con seaborn).

    Returns:
        - None
    """
    # Asegurarse de que la columna de clusters existe
    if cluster_col not in dataframe.columns:
        raise ValueError(f"La columna '{cluster_col}' no se encuentra en el DataFrame.")

    # Filtrar variables categóricas y numéricas
    categorical_cols = dataframe.select_dtypes(include=['object', 'category']).columns
    numeric_cols = dataframe.select_dtypes(include=['number']).columns

    # Configurar colores
    colors = sns.color_palette(palette, len(dataframe[cluster_col].unique()))

    # Gráficos de variables categóricas
    if len(categorical_cols) > 0:
        n_cols = 3
        n_rows = math.ceil(len(categorical_cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
        axes = axes.flatten()

        for idx, variable in enumerate(categorical_cols):
            # Calcular distribución de conteos por cluster
            distribucion = (
                dataframe.groupby([cluster_col, variable])
                .size()
                .unstack(level=0, fill_value=0)
            )
            # Ordenar las categorías de mayor a menor para cada cluster
            distribucion = distribucion.sort_values(by=distribucion.columns.tolist(), ascending=False)

            # Generar gráfico
            distribucion.plot(kind='bar', ax=axes[idx], color=colors, width=0.8)
            axes[idx].set_title(f'Conteos de {variable} por Cluster')
            axes[idx].set_xlabel(variable)
            axes[idx].set_ylabel('Conteo')
            axes[idx].legend(title='Cluster', loc='upper right')

        for idx in range(len(categorical_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    # Gráficos de variables numéricas
    if len(numeric_cols) > 0:
        n_cols = 3
        n_rows = math.ceil(len(numeric_cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows))
        axes = axes.flatten()

        for idx, variable in enumerate(numeric_cols):
            # Calcular los promedios por cluster
            promedios = dataframe.groupby(cluster_col)[variable].mean()

            # Ordenar los promedios de mayor a menor
            promedios = promedios.sort_values(ascending=False)

            # Generar gráfico
            promedios.plot(kind='bar', ax=axes[idx], color=colors, width=0.8)
            axes[idx].set_title(f'Promedio de {variable} por Cluster')
            axes[idx].set_xlabel('Cluster')
            axes[idx].set_ylabel('Promedio')
            axes[idx].grid(axis='y', linestyle='--', alpha=0.7)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()
def calcular_metricas(dataframe, cluster_col):
    """
    Calcula métricas de evaluación del clustering utilizando un DataFrame con una columna de clusters.

    Parameters:
        - dataframe (pd.DataFrame): DataFrame con variables numéricas y una columna de clusters.
        - cluster_col (str): Nombre de la columna que contiene las etiquetas de clusters.

    Returns:
        - pd.DataFrame: DataFrame con las métricas calculadas.
    """
    # Verificar que la columna de clusters existe
    if cluster_col not in dataframe.columns:
        raise ValueError(f"La columna '{cluster_col}' no se encuentra en el DataFrame.")

    # Separar las etiquetas de clusters y las características
    labels = dataframe[cluster_col]
    features = dataframe.drop(columns=[cluster_col])

    if len(set(labels)) <= 1:
        raise ValueError("El clustering debe tener al menos 2 clusters para calcular las métricas.")

    # Calcular métricas
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)

    # Calcular cardinalidad de los clusters
    unique, counts = np.unique(labels, return_counts=True)
    cardinalidad = dict(zip(unique, counts))

    # Crear DataFrame de resultados
    resultados = pd.DataFrame({
        "silhouette_score": [silhouette],
        "davies_bouldin_index": [davies_bouldin],
        "cardinalidad": [cardinalidad]
    })

    return resultados

def dividir_y_guardar_clusters(dataframe, cluster_col, output_prefix="cluster"):
    """
    Divide un DataFrame en varios archivos PKL según los valores de la columna de clusters.

    Params:
        - dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        - cluster_col (str): Nombre de la columna que contiene los clusters.
        - output_prefix (str): Prefijo para los nombres de los archivos PKL (por defecto: "cluster").

    Returns:
        - None. Los archivos PKL se guardan en el directorio actual.
    """
    # Verificar que la columna existe
    if cluster_col not in dataframe.columns:
        raise ValueError(f"La columna '{cluster_col}' no existe en el DataFrame.")
    
    # Obtener los valores únicos de clusters
    clusters = dataframe[cluster_col].unique()

    # Dividir y guardar cada cluster
    for cluster in clusters:
        df_cluster = dataframe[dataframe[cluster_col] == cluster]
        output_filename = f"../results/{output_prefix}_{cluster}.pkl"
        df_cluster.to_pickle(output_filename)
        print(f"Archivo guardado: {output_filename}")
