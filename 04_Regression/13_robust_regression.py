# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 08:41:15 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import norm
from scipy import stats
from sklearn.model_selection import cross_val_score, KFold, train_test_split

plt.style.use('seaborn')
import warnings
warnings.filterwarnings('ignore')


###################################################################################
# Read data
df = pd.read_csv('data/house_price_train.csv', index_col=0)
y = df['SalePrice']
X = df.loc[:,'MSSubClass':'YrSold']
# X = df.loc[:,'MSSubClass':'SaleCondition_Partial']

n = X.shape[0]
r = X.shape[1]

# normalize
normalize = True
if normalize:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
    y = pd.Series(StandardScaler().fit_transform(y.values.reshape(-1,1)).reshape(-1,))
X = sm.add_constant(X)   

###################################################################################
# Robust Regression (vedi la pagina dedicata per info precise sui vari tipi)

# Huber    (dati oltre lo scale parameter hanno peso che decade con 1/x, entro peso 1)
t = 1.50
hub_mod = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=t))
mod_ris = hub_mod.fit()   # I can set different covariance formulas, H1 (default), H2, H3

# LTS (Least Trimmed Squares)   (dati oltre lo scale parameter hanno peso nullo, entro peso 1)
c = 2.0
lts_mod = sm.RLM(y, X, M=sm.robust.norms.TrimmedMean(c=c))
mod_ris = lts_mod.fit()   # I can set different covariance formulas, H1 (default), H2, H3

# TukeyBiweight    (dati oltre lo scale parameter hanno peso nullo, entro una specie di parabola)
c = 4.7
tub_mod = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight(c=c))
mod_ris = tub_mod.fit()   # I can set different covariance formulas, H1 (default), H2, H3

print(mod_ris.summary())

y_hat = mod_ris.predict(X)