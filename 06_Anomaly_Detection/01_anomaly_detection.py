# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 21:09:17 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import seaborn as sns
import pandas as pd
from scipy.stats import chi2

plt.style.use('seaborn')


#################################################################################################
# Generate the data
d = 4
def GenerateAnomalyDetectionExample(n=1000,d=d,r=0.01,or_=10,random_state=1234):
    """n number of samples
       d number of dimensions
       r outlier ratio
       or_ outlier range
    """
    num_samples = n
    num_dimensions = d
    outlier_ratio = r
    # number of "usual" data points
    num_inliers = int(num_samples * (1-outlier_ratio))
    # number of outliers
    num_outliers = num_samples - num_inliers
    np.random.seed(random_state)
    # Generate the normally distributed inliers (mean 0, sigma 1)
    X_standard = np.random.randn(num_inliers, num_dimensions)
    # Add outliers sampled from a random uniform distribution
    X_outliers = np.random.uniform(low=-or_, high=or_, size=(num_outliers, num_dimensions))
    X = np.r_[X_standard, X_outliers]
    # Generate labels, 1 for inliers and −1 for outliers
    labels = np.ones(num_samples, dtype=int)
    labels[-num_outliers:]= (-1)
    return X, labels

X,labels = GenerateAnomalyDetectionExample()


################################################################################################
# Elliptic Envelope MCD (Minimum Covariance Determinant)
contamination = 0.01
MCD = EllipticEnvelope(contamination=contamination, random_state=1234)
MCD.fit(X)

norm_out = MCD.predict(X)

fig, ax = plt.subplots(1,1)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=norm_out, palette='tab10', ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Outlier', 'Normal'])
ax.set_title('MCD Outlier Detection')
fig.tight_layout()

df = pd.DataFrame(X)
df['class'] = norm_out

sns.pairplot(df, hue='class', diag_kind='kde', palette='tab10')
plt.tight_layout()

mean = MCD.location_
cov = MCD.covariance_
conf = 0.9
rad = chi2(df=d).ppf(q=conf)

fig, ax = plt.subplots(1,1)
sns.distplot(MCD.dist_[norm_out==1], ax=ax, kde=True, label='Normal Data')
xx = np.arange(0, 20, 0.01)
ax.plot(xx, chi2(df=d).pdf(xx), label=f'Chi2({d})')
ax.legend()
ax.set_title('Distance of Norml Data')


################################################################################################
# Isolation Forest
n_estim = 100
contamination = 0.01  # can avoid to set it, is set automatically
iso_for = IsolationForest(n_estimators=n_estim, contamination=contamination, random_state=1234)
iso_for.fit(X)

norm_out = iso_for.predict(X)

fig, ax = plt.subplots(1,1)
sns.scatterplot(x=X[:,0], y=X[:,1], hue=norm_out, palette='tab10', ax=ax)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Outlier', 'Normal'])
ax.set_title('MCD Outlier Detection')
fig.tight_layout()

df = pd.DataFrame(X)
df['class'] = norm_out

sns.pairplot(df, hue='class', diag_kind='kde', palette='tab10')
plt.tight_layout()



###############################################################################################
# # One Class SVM  (not good as others)
# kernel = 'rbf' # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed
# contamination = 0.01
# one_class_SVM = OneClassSVM(kernel=kernel, nu=contamination)
# one_class_SVM.fit(X)

# norm_out = one_class_SVM.predict(X)

# fig, ax = plt.subplots(1,1)
# sns.scatterplot(x=X[:,0], y=X[:,1], hue=norm_out, palette='tab10', ax=ax)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, ['Outlier', 'Normal'])
# ax.set_title('MCD Outlier Detection')
# fig.tight_layout()

# df = pd.DataFrame(X)
# df['class'] = norm_out

# sns.pairplot(df, hue='class', diag_kind='kde', palette='tab10')
# plt.tight_layout()
