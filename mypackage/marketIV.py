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
import warnings
warnings.filterwarnings("ignore")

def hello():
    print("hello this is marketIV module...")


# (0) read data
def read_data(expiration_date = "31DEC21"):
    formatted_date = datetime.strptime(expiration_date, '%d%b%y').strftime('%Y-%m-%d')
    call_strike = pd.read_csv(f"deribit_data/BTC-call/call_strike_{formatted_date}.csv", index_col="Unnamed: 0")
    put_strike = pd.read_csv(f"deribit_data/BTC-put/put_strike_{formatted_date}.csv", index_col="Unnamed: 0")
    df_idxprice = pd.read_csv(f"deribit_data/BTC-index/BTC_index_{formatted_date}.csv", index_col="Unnamed: 0")
    df_futuresprice = pd.read_csv(f"deribit_data/BTC-futures/ohlcv_hr/ohlcv_hr_{formatted_date}.csv", index_col="UTC")

    df_futuresprice.index = pd.to_datetime(df_futuresprice.index)

    print(f"* 到期日: {expiration_date}, {formatted_date}")
    print(f"* F時間範圍: {df_futuresprice.index[0].strftime('%Y-%m-%d')}, {df_futuresprice.index[-1].strftime('%Y-%m-%d')}")
    print(f"* option時間範圍: {call_strike.index[0]}, {call_strike.index[-1]}")

    return call_strike, put_strike, df_idxprice, df_futuresprice

# (2) oneday_function
def oneday_function(date, price_strike, df_idxprice, df_futuresprice, expiration_date, CPtype="C"):
    df_oneday = price_strike.loc[date][price_strike.loc[date] != 0]
    df_oneday = df_oneday.reset_index()
    df_oneday.columns = ["K", CPtype]
    df_oneday["K"] = df_oneday["K"].astype(int)
    df_oneday["S"] = df_idxprice["index_price"].loc[date]
    df_oneday["T"] = (pd.to_datetime(expiration_date, format='%d%b%y') - pd.to_datetime(date)).days/365
    df_oneday["F"] = df_futuresprice["close"].loc[df_futuresprice.index.date == pd.to_datetime(date).date()].iloc[-1]
    df_oneday["r"] = np.log(df_oneday["F"]/df_oneday["S"]) / df_oneday["T"] # 不影響 np.log(F/S)/T
    df_oneday["F_byS"] = df_oneday["S"] * np.exp(0.0148 * df_oneday["T"] )
    if CPtype == "C":
        df_oneday["IV"] = df_oneday.apply(lambda row: implied_volatility.call(row["F"], row["K"], row["T"], row["r"], row["C"]), axis=1)
    elif CPtype =="P":
        df_oneday["IV"] = df_oneday.apply(lambda row: implied_volatility.put(row["F"], row["K"], row["T"], row["r"], row["P"]), axis=1)
    
    # moneyness: 1/sqrt(T) * ln(K/F)    
    df_oneday["moneyness"] = 1/np.sqrt(df_oneday["T"]) * np.log(df_oneday["K"]/df_oneday["F"])
    
    return df_oneday            


# (3) mix_cp_function
def mix_cp_function(date, call_strike, put_strike, df_idxprice, df_futuresprice, expiration_date):
    # 1. Call 
    call_oneday = oneday_function(date, call_strike, df_idxprice, df_futuresprice, expiration_date, CPtype="C")
    call_oneday = call_oneday[~(call_oneday["C"] < 10)].reset_index(drop=True) # 排除10美金以下的call
    F = call_oneday["F"].dropna().iloc[0]

    # 2. Put
    put_oneday = oneday_function(date, put_strike, df_idxprice, df_futuresprice, expiration_date, CPtype="P")

    # Put-Call Parity : Call = Put + S - K*exp(-r*t) 
    put_oneday["C"] = put_oneday["P"] + put_oneday["S"] - put_oneday["K"] * np.exp(-put_oneday["r"]*put_oneday["T"])
    #put_oneday = put_oneday[~(put_oneday["K"] > F * 1.2)]

    # 3. 合併 Call & Put IV
    mix_cp = call_oneday.merge(put_oneday[["K","IV","C"]][(put_oneday["K"] < F * 1.2)] , 
                               on='K', how='outer').sort_values(by=["K"]).reset_index(drop=True)

    #   F 左右10% Call&Put 平均
    mix_cp["mixIV"] = mix_cp.query(f"K > {F * 0.8} & K < {F * 1.2}")[["IV_x","IV_y"]].mean(axis=1, skipna=True)

    #   價外 Call + Put
    mix_cp["Cotm"] = mix_cp["K"] > mix_cp["F"] # True/False
    mix_cp["otmIV"] = (mix_cp["IV_x"] * mix_cp["Cotm"]).fillna(0) + (mix_cp["IV_y"] * ~mix_cp["Cotm"]).fillna(0)
    mix_cp["mixIV"] = mix_cp["mixIV"].fillna(mix_cp["otmIV"])
    mix_cp["mixIV"] = mix_cp["mixIV"].replace(0, np.nan)

    
    mix_cp["C_x"] = mix_cp["C_x"].fillna(mix_cp["C_y"])
    mix_cp.rename(columns={"C_x":"C"}, inplace=True)

    mix_cp = mix_cp.dropna(subset=['mixIV']) 
    mix_cp = mix_cp[(mix_cp["mixIV"] < 2)].reset_index(drop=True) # 粗略去除極端IV

    return mix_cp, call_oneday, put_oneday

# (4) draw_IV_and_Call
def draw_IV_and_Call(date, smooth, call_oneday, put_oneday, ivname="mixIV"):
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(10, 3))

    # Fig1
    ax1.set_title(date + f": smooth IV(Call, Put)")
    ax1.scatter(call_oneday["K"], call_oneday["IV"], label="call iv", marker="o", color="mediumseagreen", s=10)
    ax1.scatter(put_oneday["K"], put_oneday["IV"], label="put iv", marker="o", color="lightcoral", s=10)
    ax1.plot(smooth["K"], smooth[ivname], alpha=0.8, label="smooth iv(DVF)", color="royalblue", zorder=2)
    ax1.plot([call_oneday["F"][0]]*2, [max(smooth[ivname]),min(smooth[ivname])-0.05],  ":", color="black", label=f"futures price")
    ax1.grid(linestyle='--', alpha=0.3)
    ax1.legend()

    # Fig2
    ax2.set_title("smooth call price")  
    ax2.plot(smooth["K"], smooth["C"], alpha=0.8, label="smooth call price", color="royalblue")
    ax2.scatter(call_oneday["K"], call_oneday["C"], label="call price", marker="o", color="mediumseagreen", s=10)
    ax2.scatter(put_oneday["K"], put_oneday["C"], label="call price(derived from put)", marker="o", color="lightcoral", s=10)
    ax2.grid(linestyle='--', alpha=0.3)
    ax2.legend()
    plt.show()

# (5) add_other_info 添加 S, T, F, C, r 資料
def add_other_info(date, oneday, smooth, call_strike, df_idxprice, df_futuresprice, expiration_date, IVname="mixIV"):
    smooth["S"] = oneday["S"].dropna().iloc[0]
    smooth["T"] = oneday["T"].dropna().iloc[0]
    smooth["F"] = oneday["F"].dropna().iloc[0]

    smooth["C"] = call.future(smooth["F"],smooth["K"],smooth["T"],smooth[IVname])
    smooth["r"] = np.log(smooth["F"]/smooth["S"]) / smooth["T"] # 不影響
    smooth["moneyness"] =  np.log(smooth["K"]/smooth["F"]) 
    return smooth

if __name__ == "__main__":
    print("YOOOO")
