# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 15:30:19 2022

@author: Ghiglione
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

plt.style.use('seaborn')

import warnings
warnings.filterwarnings('ignore')

#######################################################################
# Read the data

df = pd.read_csv('data\HousePricesInputVariables.csv')


#######################################################################
# Treat NANs by variable
na_cols = df.isnull().sum()/len(df)
na_cols = na_cols.sort_values(ascending=False)
print(na_cols)

fig, ax = plt.subplots(1,1)
sns.barplot(x=na_cols.index, y=na_cols, color='royalblue')
plt.xticks(rotation=90)
ax.set_title('% of NA per variable')
plt.tight_layout()

fig, ax = plt.subplots(1,1)
sns.barplot(x=na_cols.index[:10], y=na_cols[:10], color='royalblue')
plt.xticks(rotation=90)
ax.set_title('% of NA per variable')
plt.tight_layout()


thresh_col = 0.10
df = df.drop(columns=na_cols.index[na_cols > thresh_col])


########################################################################
# Treat NANs by row
na_rows = df.isnull().sum(axis=1)/len(df)
na_rows = na_rows.sort_values(ascending=False)
print(na_rows)

thresh_row = 0.50
df = df.drop(index=na_rows.index[na_rows > thresh_row])


########################################################################
# Fill NANs
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
    
for col in df.select_dtypes(exclude=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
    
    

#########################################################################
# Skewness and Logarithmic Transformation
skew_thresh = 1
numeric_feat = df.select_dtypes(include=np.number).columns
skewness = df[numeric_feat].apply(lambda x: np.abs(skew(x))).sort_values(ascending=False)


fig, ax = plt.subplots(1,1)
sns.barplot(x=skewness.index, y=skewness, color='royalblue', ax=ax)
plt.xticks(rotation=90)
ax.set_title('Absolute Skewness')
plt.tight_layout()


fig, ax = plt.subplots(1,1)
sns.barplot(x=skewness.index[:10], y=skewness[:10], color='royalblue', ax=ax)
plt.xticks(rotation=90)
ax.set_title('Absolute Skewness')
plt.tight_layout()


skewed_vars = skewness[(skewness > skew_thresh) & (df[numeric_feat].min() >= 0.0)].index
df[skewed_vars] = df[skewed_vars].apply(lambda x: np.log1p(x))



###########################################################################
# Correlation Analysis
corrmat = df.corr()#.abs()

fig, ax = plt.subplots(1,1)
sns.heatmap(corrmat, cmap='coolwarm', vmin=-1, vmax=1, annot=False, square=True, ax=ax)
plt.tight_layout()



##########################################################################
# One-Hot Encoding for Categorical Variables
df_one_hot = pd.get_dummies(df)


##########################################################################
# Numerical Encoding for Categorical Variables
df_num = df.copy()
df_num[df_num.columns[df_num.dtypes == 'object']] = df_num[df_num.columns[df_num.dtypes == 'object']].apply(lambda x: x.astype('category').cat.codes)


##########################################################################
# Save Data
df_one_hot.to_csv('data\house_price_ohe.csv')
df_num.to_csv('data\house_price_num.csv')





