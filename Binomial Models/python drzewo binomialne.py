# -*- coding: utf-8 -*-
#Binomial pricing model

import numpy as np

#parameters:
    
"""Starting stokc price S,strike price K, probability of moving up p, of moving down 1-p,
value of moving up u, value of moving down d, risk free rate r,  number of steps T
volatlity v
u=exp(v*sqrt(dt))
d=1/u
p=(exp(r*dt)-d)/(u-d)


this is slower version without using vectorisation. 

"""
S=100
K=110
T=4
v=0.05
r=0.02

def BinomialPricing(S,K,T,v,r,option_type: str):

    dt=1/T 
    u=np.exp(v*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    
    Stock_Prices=np.zeros((T+1,T+1))
    for i in range(T+1):
        for j in range(i+1):
            Stock_Prices[j,i]=S*(u**(i-j))*d**(j)
            
    Option_Prices=np.zeros((T+1,T+1))
    if (option_type=="C"):
        #obliczenie explicite cen opcji w ostatniej kolumnie
        for i in range(T+1):
            Option_Prices[i,T]=max((Stock_Prices[i,T])-K,0)
        #cofamy się ze wzoru f=e^-rT(pfu +(1-p)fd)
        for i in range(T-1,-1,-1):
            for j in range(i+1):
                Option_Prices[j,i]=np.exp(-r*dt)*(p*Option_Prices[j,i+1]+(1-p)*Option_Prices[j+1,i+1])
                
    if(option_type=="P"):
            for i in range(T+1):
                Option_Prices[i,T]=max(K-Stock_Prices[i,T],0)
            for i in range(T-1,-1,-1):
                for j in range(i+1):
                    Option_Prices[j,i]=np.exp(-r*dt)*(p*Option_Prices[j,i+1]+(1-p)*Option_Prices[j+1,i+1])
                    
    return Option_Prices[0,0]

print(BinomialPricing(100, 80, 4, 0.05, 0.02,"C"))