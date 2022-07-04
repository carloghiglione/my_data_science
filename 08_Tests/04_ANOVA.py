# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 18:03:26 2022

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

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


#################################################################################
# Import Data
df_read = sm.datasets.get_rdataset("Moore", "carData", cache=True).data

group = df_read['fcategory'].values
X = df_read['fscore'].values
df = pd.DataFrame({'group': group})
df['X'] = X

n = df.shape[0]
g = df['group'].nunique()
n_list = df.groupby('group').count().values

fig, ax = plt.subplots(1,1)
sns.boxplot(x='group', y='X', data=df, ax=ax)
ax.set_title('Data by group')


##################################################################################
# ANOVA One Way
mod_lm = ols('X ~ group', data=df).fit()
table = sm.stats.anova_lm(mod_lm, typ=2) # Type=2 to see it as in R

print(table)


###############
# Gaussianity Assumption
shap_group = df.groupby('group').apply(lambda x: shapiro(x)[1])

print('\nShapiro Test on data by group')
print(shap_group)


###############
# Uniformity of Variance
X_g = df.groupby('group')['X'].apply(lambda x: np.array(x))

var_list = df.groupby('group')['X'].var()
t_lev = levene(X_g[0], X_g[1], X_g[2])
print(f'\nLevene Test Uniformity of Variance: {t_lev[1]}')



################
SS_res = table.iloc[1,0]
Sp = SS_res/(n - g)
alpha = 0.95
IC_list = []
p_val_list = []
for i in range(g):
    tau_i = np.mean(X_g[i]) - np.mean(X)
    rad = t.ppf(q=1 - 0.5*alpha/g, df=n-g)*np.sqrt(Sp/n_list[i])
    IC_i = {'inf': tau_i - rad, 'center': tau_i, 'sup': tau_i + rad}
    p_val_i = 2 * ( 1 - t.cdf(x = np.abs(tau_i)/np.sqrt(var_list[i]/n_list[i]), df=n_list[i]-1) )
    IC_list.append(IC_i)
    p_val_list.append(p_val_i)

IC_list = pd.DataFrame(IC_list)
print('\nIC Tau(i)')
print(IC_list)
print(f'H0: Tau(i)=0, pvalue={p_val_list}')


#################################################################################
# Anova with Scipy
f_oneway(X_g[0], X_g[1], X_g[2])













