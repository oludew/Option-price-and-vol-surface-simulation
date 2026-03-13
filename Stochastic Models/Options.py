# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 20:02:24 2026

@author: User
"""
















#===========================TODO=========================================#
'''Maybe standarize the normal (even tough its form N(0,1))
add seed'''









import numpy as np
import math
import plotly.graph_objects as go
import webbrowser
import os
from scipy.stats import norm
from scipy.optimize import brentq


def bs_call_price(sigma, S, K, T, r):
    if T <= 0 or sigma <= 0: return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def find_iv(market_price, S, K, T, r):
    intrinsic_value = max(S - K * np.exp(-r * T), 0)
    if market_price <= intrinsic_value: return np.nan
    try:
    
        return brentq(lambda x: bs_call_price(x, S, K, T, r) - market_price, 1e-6, 3.0)
    except:
        return np.nan


n_paths = 10000      
n_steps = 1000
r = 0.05
sigma = 0.2
S0 = 100
l = 2.0             
box_nmbr = 3
m_dbl_exp = 0.05  
b_dbl_exp = 0.05    

K_values = np.linspace(80, 200, 10)
T_values = np.linspace(0.2, 2.0, 10)

prices_surface = np.zeros((len(T_values), len(K_values)))
iv_surface = np.zeros((len(T_values), len(K_values)))

for t_idx, T_val in enumerate(T_values):
    T = T_val
    dt = T / n_steps
    lam_dt = l * dt
    
    poisson_cdf = np.empty(box_nmbr+1)
    poisson_cdf[0] = np.exp(-lam_dt)
    for i in range(1, box_nmbr+1):
        poisson_cdf[i] = poisson_cdf[i-1] + (np.exp(-lam_dt) * (lam_dt**i))/math.factorial(i)
        
    u_matrix_nmbr_jumps = np.random.uniform(0, 1, (n_paths, n_steps+1))
    u_matrix_value_jumps = np.random.uniform(0, 1, (n_paths, n_steps+1))
    z_matrix = np.random.normal(0, 1, (n_paths, n_steps + 1))
    
    nmbr_jumps = np.digitize(u_matrix_nmbr_jumps, poisson_cdf)
    nmbr_jumps = np.clip(nmbr_jumps, 0, 3)
    
    def cdf_inv_dbl_exp(x):
        cond = x <= 0.5
        lhs = m_dbl_exp + b_dbl_exp * np.log(2 * x)
        rhs = m_dbl_exp - b_dbl_exp * np.log(2 * (1 - x))
        return np.where(cond, lhs, rhs)
        
    jump_values = cdf_inv_dbl_exp(u_matrix_value_jumps)
    k = np.exp(m_dbl_exp) / (1 - b_dbl_exp**2) - 1
    

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    for step in range(1, n_steps + 1):
        A = (r - l * k - 0.5 * sigma**2) * dt
        B = sigma * np.sqrt(dt) * z_matrix[:, step]
        C = jump_values[:, step] * nmbr_jumps[:, step]
        paths[:, step] = paths[:, step-1] * np.exp(A + B + C)
            
    ST = paths[:, -1]
    
    for k_idx, K_val in enumerate(K_values):
        payoffs_call = np.maximum(ST - K_val, 0)
        Price = np.exp(-r * T) * np.mean(payoffs_call)
        prices_surface[t_idx, k_idx] = Price

        iv_surface[t_idx, k_idx] = find_iv(Price, S0, K_val, T, r)

fig = go.Figure(data=[go.Surface(
    z=iv_surface, 
    x=K_values, 
    y=T_values,
)])

fig.update_layout(
    title='Powierzchnia Zmienności Implikowanej (Smile & Skew)',
    scene=dict(xaxis_title='Strike (K)', yaxis_title='Maturity (T)', zaxis_title='Implied Volatility')
)

file_path = 'iv_surface.html'
fig.write_html(file_path)
webbrowser.open('file://' + os.path.realpath(file_path))


fig = go.Figure(data=[go.Surface(
    z=prices_surface, 
    x=K_values, 
    y=T_values,
)])

fig.update_layout(
    title='Powierzchnia Ceny',
    scene=dict(xaxis_title='Strike (K)', yaxis_title='Maturity (T)', zaxis_title='Price')
)

file_path = 'price_surface.html'
fig.write_html(file_path)
webbrowser.open('file://' + os.path.realpath(file_path))


