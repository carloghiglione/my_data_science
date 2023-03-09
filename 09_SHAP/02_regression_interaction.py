# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:50:40 2023

@author: carlo
"""

import xgboost
import numpy as np
import shap
import matplotlib.pyplot as plt

 
# simulate some binary data and a linear outcome with an interaction term
# note we make the features in X perfectly independent of each other to make
# it easy to solve for the exact SHAP values
N = 2000
X = np.zeros((N,5))
X[:1000,0] = 1
X[:500,1] = 1
X[1000:1500,1] = 1
X[:250,2] = 1
X[500:750,2] = 1
X[1000:1250,2] = 1
X[1500:1750,2] = 1
X[:125,3] = 1
X[250:375,3] = 1
X[500:625,3] = 1
X[750:875,3] = 1
X[1000:1125,3] = 1
X[1250:1375,3] = 1
X[1500:1625,3] = 1
X[1750:1875,3] = 1
X[:,:4] -= 0.4999 # we can't exactly mean center the data or XGBoost has trouble finding the splits
y = 2* X[:,0] - 3 * X[:,1] + 2 * X[:,1] * X[:,2]


# ensure the variables are not correlated (not mean centered)
print('Covariance')
print(np.cov(X.T))
print('Mean')
print(X.mean(0))


# data structure optimized for xgboost function
Xd = xgboost.DMatrix(X, label=y)
Xd.get_data()

 
# train the model  
# 'eta': lerning rate, 'max_depth': maximum depth of a tree,
# 'base_score': initial prediction score of all instances, 'lambda': L2 regularization
# num_boost_round: number of trees
model = xgboost.train({
    'eta':1, 'max_depth':3, 'base_score': 0, "lambda": 0
}, Xd, num_boost_round=1)

# RMSE
print("Model error (RMSE) =", np.sqrt(np.mean( (y - model.predict(Xd))**2 )))

# show the first trained trees
first_tree = model.get_dump(with_stats=True)[0]
print('First trained trees stats')
print(first_tree)

# # visualize the fitted decision tree
# fig, ax = plt.subplots(figsize=(30, 30))
# xgboost.plot_tree(model, ax=ax)
# plt.show()


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
shap.summary_plot(shap_values, X)


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

# Dependence plot for Feature 0
shap.dependence_plot(0, shap_values, X, interaction_index='auto')

# Dependence plot for Feature 1
shap.dependence_plot(1, shap_values, X)

# Dependence plot for Feature 2
shap.dependence_plot(2, shap_values, X)