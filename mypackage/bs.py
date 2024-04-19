"""
Black Scholes Model

class:
    d1
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)
        
    d2
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)

    call
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)

    put
        spot(S, K, T, r, sigma)
        future(F, K, T, sigma)

    implied_volatility                   # 給定其他參數，找出IV
        call(F, K, T, r, call_price)
        put(F, K, T, r, put_price)
"""


import numpy as np
from scipy.optimize import bisect
from scipy.stats import norm


class d1:
    def spot(S, K, T, r, sigma):
        return ( np.log(S/K) + (r+0.5*sigma**2)*T ) / ( sigma*np.sqrt(T) )
    def future(F, K, T, sigma):
        return ( np.log(F/K) + (0.5*sigma**2)*T ) / ( sigma*np.sqrt(T) )

class d2:
    def spot(S, K, T, r, sigma):
        return d1.spot(S,K,T,r,sigma) - sigma * np.sqrt(T)
    def future(F, K, T, sigma):
        return d1.future(F, K, T, sigma) - sigma * np.sqrt(T)

class call:
    def spot(S, K, T, r, sigma):
        return norm.cdf(d1.spot(S,K,T,r,sigma)) * S - norm.cdf(d2.spot(S,K,T,r,sigma)) * K * np.exp(-r*T)

    def future(F, K, T, sigma):
        return norm.cdf(d1.future(F,K,T,sigma)) * F - norm.cdf(d2.future(F,K,T,sigma)) * K

class put:
    def spot(S, K, T, r, sigma):
        return norm.cdf(-d2.spot(S,K,T,r,sigma)) * K * np.exp(-r * T) - norm.cdf(-d1.spot(S,K,T,r,sigma)) * S

    def future(F, K, T, sigma):
        return norm.cdf(-d2.future(F,K,T,sigma)) * K - norm.cdf(-d1.future(F,K,T,sigma)) * F

class implied_volatility:
    def call(F, K, T, r, call_price):
        def func(iv_guess):
            return call.future(F, K, T, iv_guess) - call_price
        try:
            iv = bisect(func, 0.00001, 5)
            return iv
        except:
            return None
    
    def put(F, K, T, r, put_price):
        def func(iv_guess):
            return put.future(F, K, T, iv_guess) - put_price
        try:
            iv = bisect(func, 0.001, 5)
            return iv
        except:
            return None