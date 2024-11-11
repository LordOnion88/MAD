import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from statsmodels.graphics.mosaicplot import mosaic

sns.set_theme(style="darkgrid",palette="Set2")


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