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


# import
# import
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar

from scipy.optimize import bisect, minimize
from scipy.stats import norm, genextreme
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline, CubicSpline

import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from mypackage.bs import *
from mypackage.marketIV import *
import warnings
warnings.filterwarnings("ignore")

def DVF_function(date, oneday, call_strike, df_idxprice, df_futuresprice, expiration_date, IVname="mixIV", power=5):
    IV = oneday[IVname]
    F = oneday["F"].dropna().iloc[0]
    smallK = []
    for p in range(power):
        smallK.append( (oneday["K"] / 1000)**(p+1) )

    dfsmall = pd.DataFrame(smallK).T
    dvf_model = sm.OLS(IV, sm.add_constant( dfsmall )).fit()

    min_K = 0#int(min(oneday["K"]) * 0.8) # 測試!!!!!!!!!!
    max_K = max(oneday["K"])+1
    dK = 1 / 1000
    K_fine = np.arange(min_K/1000, max_K/1000, dK, dtype=np.float64)

    Kpower = []
    for p in range(power):
        Kpower.append( K_fine**(p+1) )
    df = pd.DataFrame(Kpower).T
    
    Vol_fine = dvf_model.predict( sm.add_constant(df) )
    smooth = pd.DataFrame([np.round(K_fine*1000, 0), Vol_fine], index=["K", IVname]).T
    smooth = add_other_info(date, oneday, smooth, call_strike, df_idxprice, df_futuresprice, expiration_date, IVname)
    return smooth

def UnivariateSpline_function(date, oneday, call_strike, df_idxprice, df_futuresprice, expiration_date, IVname="mixIV", power=3, s=None, w=None):
    oneday2 = oneday.query("K <= 100000")
    spline = UnivariateSpline(oneday2["K"], oneday2["mixIV"], k=power, s=s, w=w) #三次样条插值，s=0：插值函数经过所有数据点
    
    min_K = 0#int(min(oneday2["K"]) * 0.8) # 測試!!!!!!!!!!
    max_K = max(30000, max(oneday2["K"])) +1
    dK = 1
    K_fine = np.arange(min_K, max_K, dK, dtype=np.float64)
    Vol_fine = spline(K_fine)

    smooth = pd.DataFrame([K_fine, Vol_fine], index=["K", IVname]).T
    
    try:    # IV左邊有往下
        left_US = smooth.query(f"K < {oneday['K'].iloc[0]}")
        idx = left_US[left_US["mixIV"].diff() > 0].index[-1]
        smooth = smooth.loc[idx:].reset_index(drop=True)
    except: # IV左邊沒有往下
        pass

    smooth = add_other_info(date, oneday2, smooth, call_strike, df_idxprice, df_futuresprice, expiration_date, IVname)
    return smooth