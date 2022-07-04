# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 22:46:16 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel, shapiro, wilcoxon, levene, bartlett, f_oneway
from scipy.stats import t
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


##################################################################################
#Read Data
df = pd.DataFrame({'patient': np.repeat([1, 2, 3, 4, 5], 4),
                   'drug': np.tile([1, 2, 3, 4], 5),
                   'response': [30, 28, 16, 34,
                                14, 18, 10, 22,
                                24, 20, 18, 30,
                                38, 34, 20, 44, 
                                26, 28, 14, 30]})

df_orig = df.pivot(index='patient', columns='drug', values='response')
df_orig.T.plot()

df_orig.melt(ignore_index=False)


###################################################################################
# Repeated Measures test
# H0: mean(expi) = const for all i 
# H1: mean(expi) changes

mod_rm = AnovaRM(data=df, depvar='response', subject='patient', within=['drug']).fit()

print(mod_rm.summary())


#############
# Gaussianity assumption
shap_list = df.groupby('drug')['response'].apply(lambda x: shapiro(x)[1])
print('Shapiro Test by Group')
print(shap_list)


#############
# Homoschedasticity Assumption
X_g = df.groupby('drug')['response'].apply(np.array).reset_index(drop=True)

lev_test = levene(X_g[0], X_g[1], X_g[2], X_g[3])
print(f'Levene Test Homoschedasticity: {lev_test[1]}')








