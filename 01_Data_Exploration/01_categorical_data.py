# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 10:15:43 2022

@author: Ghiglione
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro

from sklearn import datasets
plt.style.use('seaborn')

import warnings 
warnings.filterwarnings("ignore")


########################################################
# Import the data
dataset = datasets.load_iris()

df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.columns = ['X1', 'X2', 'X3', 'X4']
df['class'] = dataset.target
df['class'] = pd.Categorical(df['class'])    # set categorical variable


########################################################
# Summary Statistics
print(df.describe())

print('\n Class count:')
print(df.groupby('class').size())

print('\n Class statistics: ' + 'mean')   # mean, median, std, corr, cov, var, mad, describe
print(df.groupby('class').mean())

shapiro_tests = []
for a in range(4):
    shapiro_tests.append({'variable': df.columns[a], 'p-value': shapiro(df.iloc[:,a]).pvalue})
print(pd.DataFrame(shapiro_tests))


#########################################################
# Boxplot

# Boxplot of data by variables for each class
fig, ax = plt.subplots(1,3, figsize=(10,4), sharey=True)
df.groupby('class').boxplot(ax = ax)
plt.tight_layout()

fig, ax = plt.subplots(1,3, figsize=(10,4), sharey=True)
df.groupby('class').boxplot(column=['X1','X2'], ax = ax)
plt.tight_layout()

# Boxplot of data by class of each variable
fig, ax = plt.subplots(2,2, sharey=True)
df.boxplot(by='class', ax=ax)
plt.tight_layout()

fig, ax = plt.subplots(1,2, figsize=(6,4), sharey=True)
df.boxplot(by='class', column=['X1','X2'], ax=ax)
plt.tight_layout()



############################################################
# Barplot

# Barplot of classes
fig, ax = plt.subplots(1,1, figsize=(8,4))
sns.countplot(df['class'], ax=ax)
ax.set_title('Class count')
plt.tight_layout()



# Barplot of summary statistics for each class
stat_group = df.groupby('class').mean()
fig, ax = plt.subplots(1,1, figsize=(8,4))
stat_group.plot(y=['X1','X2'], kind='bar', ax=ax)
ax.set_title('Statistics by group')
plt.tight_layout()

stat_group = df.groupby('class').mean()
fig, ax = plt.subplots(1,1, figsize=(8,4))
stat_group.plot(kind='bar', ax=ax)
ax.set_title('Statistics by group')
plt.tight_layout()



########################################################
# Correlation matrix plot

# Correlation among the continuous attributes (no division by class)
corrmat = df.corr()
fig, ax = plt.subplots(1,1)
sns.heatmap(corrmat, square=True, cmap='coolwarm', annot=True, ax=ax)
ax.set_title('Correlation')
plt.tight_layout()



#########################################################
# Historam

# Histogram of data
fig, ax = plt.subplots(1,1, figsize=(8,4))
sns.distplot(df['X1'], bins=15, kde=True, ax=ax)
ax.set_title('Single variable histogram')
plt.tight_layout()

fig, ax = plt.subplots(1,1, figsize=(8,4))
df.groupby('class')['X1'].plot(kind='density', ax=ax)
ax.set_title('Single variable histogram')
ax.legend()
plt.tight_layout()



###########################################################
# Scatterplot

# Scatterplot by group
fig, ax = plt.subplots(1,1, figsize=(8,6))
sns.scatterplot('X1', 'X2', data=df, hue='class', ax=ax)
ax.set_title('Scatterplot by class')
plt.tight_layout()



###########################################################
# Pairplot

sns.pairplot(df, hue='class', diag_kind='kde')   # kde, hist


















