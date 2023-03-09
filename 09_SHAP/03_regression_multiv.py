# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:52:56 2023

@author: carlo
"""

import xgboost
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns


###############################################################################
###############################################################################
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

# data structure optimized for xgboost function
Xd = xgboost.DMatrix(X, label=y)
Xd.get_data()

# train the model  
# 'eta': lerning rate, 'max_depth': maximum depth of a tree,
# 'base_score': initial prediction score of all instances, 'lambda': L2 regularization
# num_boost_round: number of trees
model = xgboost.train({
    'eta':0.05, 'max_depth':5, 'base_score': 0, "lambda": 0
}, Xd, num_boost_round=200)

# RMSE
print("Model error (RMSE) =", np.sqrt(np.mean( (y - model.predict(Xd))**2 )))


# show the first trained trees
first_tree = model.get_dump(with_stats=True)[0]
print('First trained trees stats')
print(first_tree)


# Feature Importance
fig, ax = plt.subplots(1,1)
xgboost.plot_importance(model, importance_type='gain', show_values=False, max_num_features=10, ax=ax)

 
###############################################################################
###############################################################################
# SHAP Analysis

# SHAP values are s.t. the sum of the rows equals to the difference between
# SHAP model output for that input (explainer.expected_value) and
# the output predicted by the model
pred = model.predict(Xd, output_margin=True)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xd)

# do the check
print('SHAP Values Check')
print(np.abs(shap_values.sum(1) + explainer.expected_value - pred).max())

# summary plot
fig, ax = plt.subplots(1,1)
ax = shap.summary_plot(shap_values, X)

# Regarding OverallQual feature I can comment that:
    # high values of the feature have positive impact on model output (positive SHAP)
    # low values of the feature have negative impact on model output (negative SHAP)

# SHAP interaction values (if no interaction, only diagonal values)
shap_interaction_values = explainer.shap_interaction_values(Xd)
print('SHAP Interaction Values')
print(shap_interaction_values[0])  # I just show for the first observation

# ensure the SHAP interaction values sum to the marginal predictions
print('SHAP Interaction Values Check')
print(np.abs(shap_interaction_values.sum((1,2)) + explainer.expected_value - pred).max())

# Dependence Plot: Vertical dispersion of the data points represents interaction effects
# In 'interaction_index' I set the index versus whom I check interaction
# If interaction_index='auto' (default) it sets the feature with the strongest interaction

# Scatterplot of SHAP values for 'OverallQual' vs its values to see the impact
# on the output as 'OverallQual' (i=3) varies
i = 6
shap.dependence_plot(i, shap_values, X, interaction_index=None)

fig, ax = plt.subplots(2,1)
ax[0].scatter(X.iloc[:,i], y)
ax[0].set_title(f'Output vs {X.columns[i]}')
ax[1].scatter(X.iloc[:,i], shap_values[:,i])
ax[1].set_title(f'SHAP vs {X.columns[i]}')
fig.tight_layout()

# plt.scatter(X['OverallQual'], shap_values[:,3])

# Dependence plot for 'OverallQual' vs feature with the largest interaction
shap.dependence_plot(i, shap_values, X, interaction_index='auto')