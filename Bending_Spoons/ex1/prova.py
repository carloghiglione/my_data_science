# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:52:59 2022

@author: Utente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def renewals(T, p):
    renewals = 0
    subscribes = 1
    while renewals < T and subscribes:
        subscribes = np.random.binomial(1, 1-p)
        renewals += subscribes
    return renewals
    

def generate_users(N, T, p):
    user_list = []
    for user in range(N):
        user_dict = {
            'user_id': user,
            'renewals': renewals(T, p),
            'T': T
            }
        user_list.append(user_dict)
    return user_list


np.random.seed(1234)
N = 50
T = 10
p_used = 0.7


user_list = generate_users(N, T, p_used)

tab = pd.DataFrame(user_list)

fig, ax = plt.subplots(1,1)
plt.hist(tab['renewals'])
ax.set_title(f'N: {N} | T: {T} | p: {p_used} ')



def prob_sing(renewals, p, T):
    if renewals < T:
        return ((1-p)**(renewals))*p 
    else:
        probs = [ ((1-p)**(k))*p for k in range(0,T) ]
        return np.sum(probs)


def likelihood(user_list, p, T):
    prob = 1
    for user in user_list:
        prob *= prob_sing(user['renewals'], p, T)
    return prob

grid_p = np.linspace(0,1,1000)
lik_p_grid = likelihood(user_list, grid_p, T)

fig, ax = plt.subplots(1,1)
ax.plot(grid_p, lik_p_grid)


# print(grid_p[np.argmax(lik_p_grid)])



####################
import os
real_df = pd.read_csv(os.path.join('data', 'data_subscriptions.csv'), index_col=0)
real_df = real_df.sort_values('renewals')

real_user_list = real_df.to_dict(orient='records')


N_real = real_df['N'].sum()


real_df['N_under'] = (real_df['N']/40).apply(int)
real_df['renewals'] = real_df['renewals'].apply(int)


user_list_real = []

for i in range(21):
    n_curr_ren = real_df[real_df['renewals']==i]['N_under'].values[0]
    for j in range(n_curr_ren):
        user_dict = {
            'renewals': i,
            'T': 20
            }
        user_list_real.append(user_dict)



# function finding the maximum likelihood estimator
def mle_p(obs_R, T):
    p_grid = np.linspace(0,1, 1000)
    lik_p_grid = likelihood(obs_R, p_grid, T)
    return p_grid[np.argmax(lik_p_grid)], lik_p_grid, p_grid


mle_p_real, lik_p_grid, p_grid = mle_p(user_list_real, 20)

fig, ax = plt.subplots(1,1)
ax.plot(p_grid, lik_p_grid, label='likelihood')
ax.axvline(x=mle_p_real, color='red', label='p MLE', linestyle='--')
# ax.axvline(x=p_real, color='green', label='p real', linestyle='--')
ax.set_title(f'MLE Estimator real data')
ax.set_xlabel('p')
ax.legend(loc='upper left')

print(f'MLE real p: {mle_p_real}')



############
tab_real = pd.DataFrame(user_list_real)
renewals_real = tab_real['renewals'].values


user_list_sim = generate_users(250, 20, mle_p_real)
tab_sim = pd.DataFrame(user_list_sim)
renewals_sim = tab_sim['renewals'].values


fig, ax = plt.subplots(1,1)
ax.hist(renewals_real, label='real data', alpha=0.5, bins=20)
ax.hist(renewals_sim, label='simul data', alpha=0.5, bins=20)
ax.legend()



