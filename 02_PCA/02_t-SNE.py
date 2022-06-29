# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:30:10 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn import datasets
import seaborn as sns

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


#####################################################################################
# read the data
dataset = datasets.load_iris()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.columns = ['X1', 'X2', 'X3', 'X4']
df['class'] = dataset.target
df['class'] = pd.Categorical(df['class'])    # set categorical variable

tab = df.iloc[:,0:4]
tab_cols = tab.columns
n_cols = len(tab_cols)


######################################################################################
# Apply t-SNE

# canonical range: 30-50
perp = 30
n_comp = 2

tsne = TSNE(n_components=n_comp, perplexity=perp, random_state=1234)
tab_tsne = tsne.fit_transform(tab.to_numpy())
tab_tsne = pd.DataFrame(tab_tsne, columns=['Comp'+str(a+1) for a in range(n_comp)])

fig, ax = plt.subplots(1,1)
sns.scatterplot('Comp1', 'Comp2', data=tab_tsne, hue=df['class'], ax=ax)
ax.set_title('t-SNE Projection')
fig.tight_layout()


















