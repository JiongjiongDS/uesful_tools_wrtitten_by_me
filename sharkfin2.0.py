import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

np.random.seed(20180829)

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

    if spot>=H:
        return 0
    else:

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

def generate_matrix(ob_time,N):
    result=np.random.randn(N,ob_time)
    return result

def KO_judge(x):
    return 0 in np.array(x)

def find_max(a,b):
    return a if a>=b else b

def stock_price(st,ob_time,vol,r,interval,div,rand_matrix,n):
    s_price=np.ones([n,1])
    s_price=s_price*st
    s_price=np.column_stack((s_price,rand_matrix))
    for i in range(ob_time):
        s_price[:,[i+1]]=s_price[:,[i]]*np.exp((r-div-0.5*vol**2)*interval+vol*sqrt(interval)*rand_matrix[:,[i]])

    result=np.delete(s_price,0,axis=1)
    return result

def uo_pricer_MC(spot,K,H,r,q,vol,frequency,days,simu_time,ttm,randm):
    if spot>=H:
        return 0
    else:
        ST=stock_price(spot,days,vol,r,frequency,q,randm,simu_time)
        Ko=np.where((ST>=H),0,1)
        hehe=np.apply_along_axis(KO_judge,1,Ko)
        Ko_index=list(np.where(hehe==True)[0])
        a=range(simu_time)
        Left_index=list(set(a)^set(Ko_index))
#        print Left_index
#        return ST
        df=DataFrame(ST)
        No_Ko=df.iloc[Left_index,:]
        STT=list(No_Ko[days-1])
        final_result=[x-K if x >K else 0 for x in STT]
        final_result=exp(-r*ttm)*np.array(final_result)

        return (np.sum(final_result)/simu_time)


if __name__ == '__main__':
    spot_price=100.0
    strike=100.0
    barrier=110.0
    vol_rate=0.2
    risk_free=0.0
    div=0.07
    Notional=50000000
    cash=0.0
    h=0.01
    x_axis=np.linspace(50.0,130.0,50)

    sharkfin_delta_approx_3M=[]
    sharkfin_gamma_approx_3M=[]
    sharkfin_vega_approx_3M=[]
    sharkfin_delta_MC_3M=[]
    sharkfin_vega_MC_3M=[]
    sharkfin_delta_approx_6M=[]
    sharkfin_gamma_approx_6M=[]
    sharkfin_vega_approx_6M=[]
    sharkfin_delta_MC_6M=[]
    sharkfin_vega_MC_6M=[]

    # American Sharkfin Delta Approximation 3 months
    for i in x_axis:
        rand=generate_matrix(63,300000)
        sharkfin_delta_approx_3M.append((Notional/100.0)*(uo_pricer(i+h,strike,barrier,risk_free,div,vol_rate,0.25,1,0)-uo_pricer(i-h,strike,barrier,risk_free,div,vol_rate,0.25,1,0))/(2*h))

    plt.figure()
    plt.plot(x_axis, sharfin_delta_approx, "r-", label="delta")
    plt.xlabel("spot_price")
    plt.ylabel("delta")
    plt.title("delta derived from central difference method,3 months")
    plt.grid(True)
    plt.legend()
    plt.show()
    # American Sharkfin Gamma Approximation 3 months
    for i in x_axis:
        A=uo_pricer(i+h,strike,barrier,risk_free,div,vol_rate,0.25,1,0)
        B=uo_pricer(i-h,strike,barrier,risk_free,div,vol_rate,0.25,1,0)
        C=uo_pricer(i,strike,barrier,risk_free,div,vol_rate,0.25,1,0)
        sharfin_gamma_approx.append((Notional/100.0)*(A+B-2*C)/(h**2))

    plt.figure()
    plt.plot(x_axis, sharfin_gamma_approx, "r-", label="gamma")
    plt.xlabel("spot_price")
    plt.ylabel("gamma")
    plt.title("gamma derived from central difference method,3 months")
    plt.grid(True)
    plt.legend()
    plt.show()
    # American Sharkfin Vega Approximation 3 months
    for i in x_axis:
        A = uo_pricer(i, strike, barrier, risk_free, div, vol_rate+h, 0.25, 1, 0)
        B = uo_pricer(i, strike, barrier, risk_free, div, vol_rate-h, 0.25, 1, 0)
        sharkfin_vega_approx.append((Notional / 100.0) * (A-B) / (h * 2))

    plt.figure()
    plt.plot(x_axis, sharkfin_vega_approx, "r-", label="vega")
    plt.xlabel("spot_price")
    plt.ylabel("vega")
    plt.title("vega derived from central difference method,3 months")
    plt.grid(True)
    plt.legend()
    plt.show()

    # American Sharkfin Delta Approximation 6 months
    sharfin_delta_approx = []
    sharfin_gamma_approx = []
    sharkfin_vega_approx = []
    for i in x_axis:
        sharfin_delta_approx.append((Notional / 100.0) * (
        uo_pricer(i + h, strike, barrier, risk_free, div, vol_rate, 0.5, 1, 0) - uo_pricer(i - h, strike, barrier,risk_free, div, vol_rate,0.5, 1, 0)) / (2 * h))

    plt.figure()
    plt.plot(x_axis, sharfin_delta_approx, "r-", label="delta")
    plt.xlabel("spot_price")
    plt.ylabel("delta")
    plt.title("delta derived from central difference method,6 months")
    plt.grid(True)
    plt.legend()
    plt.show()
    # American Sharkfin Gamma Approximation 6 months
    for i in x_axis:
        A = uo_pricer(i + h, strike, barrier, risk_free, div, vol_rate, 0.5, 1, 0)
        B = uo_pricer(i - h, strike, barrier, risk_free, div, vol_rate, 0.5, 1, 0)
        C = uo_pricer(i, strike, barrier, risk_free, div, vol_rate, 0.5, 1, 0)
        sharfin_gamma_approx.append((Notional / 100.0) * (A + B - 2 * C) / (h ** 2))

    plt.figure()
    plt.plot(x_axis, sharfin_gamma_approx, "r-", label="gamma")
    plt.xlabel("spot_price")
    plt.ylabel("gamma")
    plt.title("gamma derived from central difference method,6 months")
    plt.grid(True)
    plt.legend()
    plt.show()
    # American Sharkfin Vega Approximation 3 months
    for i in x_axis:
        A = uo_pricer(i, strike, barrier, risk_free, div, vol_rate + h, 0.5, 1, 0)
        B = uo_pricer(i, strike, barrier, risk_free, div, vol_rate - h, 0.5, 1, 0)
        sharkfin_vega_approx.append((Notional / 100.0) * (A - B) / (h * 2))

    plt.figure()
    plt.plot(x_axis, sharkfin_vega_approx, "r-", label="vega")
    plt.xlabel("spot_price")
    plt.ylabel("vega")
    plt.title("vega derived from central difference method,6 months")
    plt.grid(True)
    plt.legend()
    plt.show()







