# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:58:17 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import shap

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

plt.style.use('seaborn')
import warnings
warnings.filterwarnings("ignore")

 
#################################################################################
# Read Data
df = pd.read_csv('data/LoansNumerical.csv')
target = 'safe_loans'

y = df[target]
y = pd.Series([1 if yy == 1 else 0 for yy in y])
X = df[df.columns[df.columns != target]]

n = X.shape[0]
r = X.shape[1]

normalization = True
if normalization:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# data structure optimized for xgboost function
Xd = xgboost.DMatrix(X, label=y)
Xd.get_data()

# train the model  
# 'eta': lerning rate, 'max_depth': maximum depth of a tree,
# 'base_score': initial prediction score of all instances, 'lambda': L2 regularization
# 'objective': type of fitting, 'binary:logistic' for binary classification and returning probability for 1 class
# 'num_class': number of classes, needed for classification
# num_boost_round: number of trees

model = xgboost.train({
    'eta':0.05, 'max_depth':3, "lambda": 0, 'objective': 'binary:logistic'
}, Xd, num_boost_round=50)

p_hat = model.predict(Xd)
thresh = 0.5
y_hat = [1 if p > thresh else 0 for p in p_hat]
accuracy = accuracy_score(y, y_hat)
print('Accuracy:', round(accuracy,3))

# Confusion Matrix
cm_norm = confusion_matrix(y, y_hat)/len(y)

fig, ax = plt.subplots(1,1, figsize=(5,4))
sns.heatmap(cm_norm, square=True, cmap='Blues', annot=True, ax=ax)
ax.set_title('Confusion Matrix Norm, Random Forest')
plt.tight_layout()

 
# Feature Importance
fig, ax = plt.subplots(1,1)
xgboost.plot_importance(model, importance_type='gain', show_values=False, max_num_features=10, ax=ax)

###############################################################################
###############################################################################
# SHAP Analysis
# SHAP values are s.t. the sum of the rows equals to the difference between
# SHAP model output for that input (explainer.expected_value) and
# the output predicted by the model
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xd)

# SHAP summary plot
fig, ax = plt.subplots(1,1)
ax = shap.summary_plot(shap_values, X)


# Dependence Plot: Vertical dispersion of the data points represents interaction effects
# In 'interaction_index' I set the index versus whom I check interaction
# If interaction_index='auto' (default) it sets the feature with the strongest interaction

# Scatterplot of SHAP values for i-th variable vs its values to see the impact on the output
i = 15
shap.dependence_plot(i, shap_values, X, interaction_index=None)

fig, ax = plt.subplots(2,1)
ax[0].scatter(X.iloc[:,i], y)
ax[0].set_title(f'Output vs {X.columns[i]}')
ax[1].scatter(X.iloc[:,i], shap_values[:,i])
ax[1].set_title(f'SHAP vs {X.columns[i]}')
fig.tight_layout()


# Dependence plot for 'OverallQual' vs feature with the largest interaction
shap.dependence_plot(i, shap_values, X, interaction_index='auto')