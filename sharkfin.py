import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math


def bs_call(spot,strike,r,q,vol,t):
    d1 = ((math.log(spot/strike))+t*(r-q+vol**2/2))/(vol*math.sqrt(t))
    d2 = d1 - vol*math.sqrt(t)
    c = spot*(math.exp(-q*t))*norm.cdf(d1) - strike*(math.exp(-r*t))*norm.cdf(d2)
    return c


def bs_delta(spot,strike,r,q,vol,t):
    d1 = ((math.log(spot / strike)) + t * (r - q + vol ** 2 / 2)) / (vol * math.sqrt(t))
    return norm.cdf(d1)*math.exp(-q*t)

def uo_call(spot,K,H,r,q,vol,t,frequency):

    if frequency == 1: # 1 denotes observation frequency is every day
        H = H * math.exp(0.5826*vol*math.sqrt(1.0/365.0))
    elif frequency ==2: # 2 denotes observation frequency is every month
        H = H * math.exp(0.5826*vol*math.sqrt(1.0/12.0))
    elif frequency ==3: # 3 denotes observation frequency is every year
        H = H * math.exp(0.5826*vol*math.sqrt(1.0/1.0))
    else:
        H=H

    lamda=(r-q+vol**2/2)/(vol**2)

    x1=math.log(spot/H)/(vol*math.sqrt(t))+lamda*vol*math.sqrt(t)
    y1=math.log(H/spot)/(vol*math.sqrt(t))+lamda*vol*math.sqrt(t)
    y=math.log(H**2/(spot*K))/(vol*math.sqrt(t))+lamda*vol*math.sqrt(t)

    c_price=bs_call(spot,K,r,q,vol,t)
    c_ui=(spot*norm.cdf(x1)*math.exp(-q*t)-K*math.exp(-r*t)*norm.cdf(x1-vol*math.sqrt(t))-spot*math.exp(-q*t)*((H/spot)**(2*lamda))*(norm.cdf(-y)-norm.cdf(-y1))
          +K*math.exp(-r*t)*((H/spot)**(2*lamda-2))*(norm.cdf(-y+vol*math.sqrt(t))-norm.cdf(-y1+vol*math.sqrt(t))))
    c_uo=c_price-c_ui

    return c_uo

def ui_binary(spot,K,H,r,q,vol,t,frequency):

    if frequency == 1:
        H=H*math.exp(0.5826*vol*math.sqrt(1.0/365.0))
    elif frequency == 2:
        H=H*math.exp(0.5826*vol*math.sqrt(1.0/12.0))
    elif frequency == 3:
        H=H*math.exp(0.5826*vol*math.sqrt(1.0/1.0))
    else:
        H=H
        print "hehe"

    miu=(r-q-vol**2/2)/(vol**2)
    lamda_xi=math.sqrt(miu**2+2*r/(vol**2))
    z=math.log(H/spot)/(vol*math.sqrt(t))+lamda_xi*vol*math.sqrt(t)
    result=K*(((H/spot)**(miu+lamda_xi))*norm.cdf(-z)+((H/spot)**(miu-lamda_xi))*norm.cdf(-z+2*lamda_xi*vol*math.sqrt(t)))

    return result

def uo_pricer(spot,K,H,r,q,vol,t,frequency,cash):

    if frequency == 1:
        H = H * math.exp(0.5826 * vol * math.sqrt(1.0 / 365.0))
    elif frequency == 2:
        H = H * math.exp(0.5826 * vol * math.sqrt(1.0 / 12.0))
    elif frequency == 3:
        H = H * math.exp(0.5826 * vol * math.sqrt(1.0 / 1.0))
    else:
        H = H

    b=r-q
    eta=-1
    phi=1

    miu=(b-vol**2/2)/(vol**2)
    lamda=math.sqrt(miu**2+(2*r/vol**2))
    z=math.log(H/spot)/(vol*math.sqrt(t))+lamda*vol*math.sqrt(t)

    x1=math.log(spot/K)/(vol*math.sqrt(t))+(1+miu)*vol*math.sqrt(t)
    x2=math.log(spot/H)/(vol*math.sqrt(t))+(1+miu)*vol*math.sqrt(t)
    y1=math.log(H**2/(spot*K))/(vol*math.sqrt(t))+(1+miu)*vol*math.sqrt(t)
    y2=math.log(H/spot)/(vol*math.sqrt(t))+(1+miu)*vol*math.sqrt(t)

    A=phi*spot*math.exp((b-r)*t)*norm.cdf(phi*x1)-phi*K*math.exp(-r*t)*norm.cdf(phi*x1-phi*vol*math.sqrt(t))
    B=phi*spot*math.exp((b-r)*t)*norm.cdf(phi*x2)-phi*K*math.exp(-r*t)*norm.cdf(phi*x2-phi*vol*math.sqrt(t))
    C=phi*spot*math.exp((b-r)*t)*norm.cdf(eta*y1)*((H/spot)**(2*(miu+1)))-phi*K*math.exp(-r*t)*((H/spot)**(2*miu))*norm.cdf(eta*y1-eta*vol*math.sqrt(t))
    D=phi*spot*math.exp((b-r)*t)*norm.cdf(eta*y2)*((H/spot)**(2*(miu+1)))-phi*K*math.exp(-r*t)*((H/spot)**(2*miu))*norm.cdf(eta*y2-eta*vol*math.sqrt(t))
    F=cash*(((H/spot)**(miu+lamda))*norm.cdf(eta*z)+((H/spot)**(miu-lamda))*norm.cdf(eta*z-2*eta*lamda*vol*math.sqrt(t)))

    return A-B+C-D+F



if __name__ == '__main__':
    spot_price=100.0
    strike=3536.25/3273.27*100
    barrier=3536.25*1.1/3273.27*100
    barrier_shift=barrier*1.005
    vol_rate=0.2
    risk_free=0.044
    div=0.0189
    N=(barrier-strike-0.51*3536.25/3273.27)/(barrier*0.005)
    ttm = 231.0/365.0
    PRate= 0.79
    Notional=309481237.85
    cash=0.51/PRate


    spread1=bs_call(spot_price,strike,risk_free,div,vol_rate,ttm)-bs_call(spot_price,barrier,risk_free,div,vol_rate,ttm)

    spread2=N*(bs_call(spot_price,barrier,risk_free,div,vol_rate,ttm)-bs_call(spot_price,barrier_shift,risk_free,div,vol_rate,ttm))


    print "sharkfin_price"
    print (PRate*spread1-spread2)*Notional*245/365/100.0
    print spread1
    print spread2

    delta1=bs_delta(spot_price,strike,risk_free,div,vol_rate,ttm)-bs_delta(spot_price,barrier,risk_free,div,vol_rate,ttm)

    delta2=N*(bs_delta(spot_price,barrier,risk_free,div,vol_rate,ttm)-bs_delta(spot_price,barrier_shift,risk_free,div,vol_rate,ttm))


    print "delta1-delta2"
    print (PRate*delta1-delta2)


    uo_call1=uo_call(spot_price,strike,barrier,risk_free,div,vol_rate,ttm,1)


    uin1=ui_binary(spot_price,cash,barrier,risk_free,div,vol_rate,ttm,1)

    uo_hahah=uo_pricer(spot_price,strike,barrier,risk_free,div,vol_rate,ttm,1,cash)

    print uo_call1
    print uin1

    print "sharkfin price"
    print  (PRate*uo_call1-uin1)*Notional*245/365/100.0
    print  PRate*uo_call1-uin1

    print  uo_hahah*PRate

