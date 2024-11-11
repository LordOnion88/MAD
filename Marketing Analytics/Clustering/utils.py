import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.cluster import KMeans
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid",palette="Set2")
pd.set_option('display.max_rows', None)


def Graficar_categoricas (variable, df):
    plt.figure(figsize=(10,8))
    ax = sns.countplot(x= variable,
              data=df
              )
    plt.xlabel(f'{variable}')
    plt.ylabel('Cantidad')
    plt.title(f'Cantidad de registros por {variable}')
    plt.tight_layout()
    plt.xticks(rotation = 90)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                (p.get_x() + p.get_width() / 2, p.get_height()), 
                ha='center', va='bottom')
    plt.show()

def elbow_plot(df: pd.DataFrame, n: int = 2, scaled_data: bool = True):
    """
    Función que plotea el número de clusters vs WCSS para determinar el número de clusters óptimo.

    :param df: pd.DataFrame con la data original.
    :param n: int, opcional. Número máximo de clusters a comparar. Default es 2.
    :param scaled_data: bool, opcional. Indica si los datos ya están escalados. Default es True.
    :raises ValueError: Si el DataFrame contiene valores NaN o si no se puede calcular KMeans.
    :return: None
    
    **Nota: Si scaled_data es False, los datos se escalan utilizando StandardScaler.**
    """
    if df.isnull().values.any():
        raise ValueError("El DataFrame contiene valores NaN. KMeans no funcionará correctamente.")

    if not scaled_data:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)
    else:
        df_scaled = df

    try:
        inertias = [KMeans(n_clusters=k, random_state=42).fit(df_scaled).inertia_ for k in range(1, n+1)]
    except Exception as e:
        raise ValueError(f"No se pudo calcular KMeans: {e}")

    data_plot = pd.DataFrame({"n_clusters": range(1, n+1), "WCSS": inertias})
    (
        px.line(
            data_plot,
            x="n_clusters",
            y="WCSS",
            title="Finding optimal number of clusters",
            template="plotly_white",
            markers=True
        )
        .update_xaxes(title_text="Number of clusters")
        .update_yaxes(title_text="WCSS")
    ).show()

def Coolinealidad(matriz, p, ax):
    sns.heatmap(matriz, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(f'Matriz de correlación {p}')

def cluster_biplot(pca, dataframe, col_clusters, comp1=1, comp2=2, size_text=8):
    """
    Esta función construye el biplot de un PCA y muestra los clusters en el plano factorial.

    :param pca: Objeto PCA que se usará para plotear los clusters.
    :param dataframe: pandas.DataFrame con el que se realizó el PCA.
    :param col_cluster: Lista o pd.Series con las etiquetas asignadas vía el método de cluster.
    :param comp1: Componente en el eje x.
    :param comp2: Componente en el eje y.
    :param size_text: int, opcional. Tamaño del texto para evitar overlapping (default: 10).
    :return: None
    """
    percent_var, length = pca.explained_variance_ratio_ * 100, len(pca.explained_variance_)
    X_scaled = StandardScaler().fit_transform(dataframe)
    
    tmp = dataframe.copy()
    if tmp.index.name == None:  # En caso que el DataFrame no tenga nombre para el índice.
        tmp.index.name = "Indice"
    pca_trans = pd.DataFrame(
        pca.transform(X_scaled), index=tmp.index,
        columns=[f"PC{str(comp)}" for comp in range(1, length + 1)]
    )
    
    tmp["cluster"] = col_clusters
    tmp["cluster"] = tmp["cluster"].astype("string")
    
    fig = px.scatter(tmp, x=pca_trans[f"PC{str(comp1)}"], y=pca_trans[f"PC{str(comp2)}"], color="cluster",
                     text=tmp.index, hover_name=tmp.index, template="plotly_white", symbol="cluster")
    
    fig.add_hline(y=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.add_vline(x=0, line_width=0.5, line_dash="dash", line_color="black")
    fig.update_layout(title="PCA-CLUSTER Biplot.")
    fig.update_xaxes(range=[min(pca_trans[f"PC{str(comp1)}"] - 0.35), max(pca_trans[f"PC{str(comp1)}"]) + 0.35],
                     title_text="Dim " + str(comp1) + " ({:.2f}%)".format(percent_var[comp1 - 1]))
    fig.update_yaxes(range=[min(pca_trans[f"PC{str(comp2)}"] - 0.35), max(pca_trans[f"PC{str(comp2)}"]) + 0.35],
                     title_text="Dim " + str(comp2) + " ({:.2f}%)".format(percent_var[comp2 - 1]))
    fig.show()

    return None