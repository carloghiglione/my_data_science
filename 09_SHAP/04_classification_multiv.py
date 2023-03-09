# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 19:56:17 2023

@author: carlo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import numpy as np
import shap
from sklearn import datasets
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


plt.style.use('seaborn')
import warnings
warnings.filterwarnings("ignore")

#################################################################################
#################################################################################
# Read Data
iris = datasets.load_iris()
target = np.array(iris.target)

X = iris.data
y = target
n_classes = len(np.unique(y))

n = X.shape[0]
r = X.shape[1]

normalization = True
if normalization:
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=iris.feature_names)

# data structure optimized for xgboost function
Xd = xgboost.DMatrix(X, label=y)
Xd.get_data()


# train the model  
# 'eta': lerning rate, 'max_depth': maximum depth of a tree,
# 'base_score': initial prediction score of all instances, 'lambda': L2 regularization
# 'objective': type of fitting, 'multi:softmax' for multiclass classification and returning predicted class
#                               'multi:softprob' for multiclass classification and returning predicted probabilities of each class
# 'num_class': number of classes, needed for classification
# num_boost_round: number of trees
model = xgboost.train({
    'eta':0.05, 'max_depth':3, 'base_score': 0, "lambda": 0, 'objective': 'multi:softmax', 'num_class':n_classes
}, Xd, num_boost_round=50)

y_hat = model.predict(Xd)
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
xgboost.plot_importance(model, importance_type='gain', show_values=False, ax=ax)

###############################################################################
###############################################################################
# SHAP Analysis
# SHAP values are s.t. the sum of the rows equals to the difference between
# SHAP model output for that input (explainer.expected_value) and
# the output predicted by the model
pred = model.predict(Xd, output_margin=True)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(Xd)

# mean SHAP for each classified unit
fig, ax = plt.subplots(1,1)
ax = shap.summary_plot(shap_values, X)

# SHAP values for each classification
for i in range(n_classes):
    fig, ax = plt.subplots(1,1)
    ax = shap.summary_plot(shap_values[i], X)

 

# Dependence Plot: Vertical dispersion of the data points represents interaction effects
# In 'interaction_index' I set the index versus whom I check interaction
# If interaction_index='auto' (default) it sets the feature with the strongest interaction


# Scatterplot of SHAP values for a given class
check_class = 1
for i in range(r):
    shap.dependence_plot(i, shap_values[check_class], X, interaction_index=None)