import numpy as np
from scipy.stats import norm
from math import exp, sqrt
import random
import matplotlib.pyplot as plt
import datetime




# to fix one random matrix fix the path
def generate_matrix(ob_time,N):
    result=np.random.randn(N,ob_time)
    return result

# after we have the random matrix, we put it together with the stock initial price column, then we use GBM to compute the following price, later we delete the initial stock price
# as it is useless
def stock_price(st,ob_time,vol,r,time,div,rand_matrix,n):
    s_price=np.ones([n,1])
    s_price=s_price*st
    s_price=np.column_stack((s_price,rand_matrix))
    for i in range(ob_time):
        s_price[:,[i+1]]=s_price[:,[i]]*np.exp((r-div-0.5*vol**2)*time[i]+vol*sqrt(time[i])*rand_matrix[:,[i]])
    print rand_matrix
    print s_price
    result=np.delete(s_price,0,axis=1)
    print result
    return result

# the logic for knock out function is that from time 0 to 10,if one KO happens, then it is ko
def knock_out(s_array,ob_time,s0):
    for i in range(ob_time-1):
        if s_array[0,i]>=s0:
            return i+1
            break

    return False

# we need to make sure the knock in total number is 12 (including the final observation date)
def knock_in(s_array,ob_time,s0):
    sum_i=0
    for i in range(ob_time):
        if s_array[0,i]/s0>=0.86:
            sum_i=sum_i+1
    return sum_i

def autocall(S0,N,rand):
   #all_time=[35,28,28,28,35,28,35,28,28,35,28,35]
   trading_time=[25,19,20,15,23,18,25,19,20,25,14,25]
   trade_time=np.array(trading_time)
   trade_time=trade_time/float(252)
   now_time=[35,28,28,28,35,28,35,28,28,35,28,35]
   rate=0.03
   time=np.array(now_time)
   time=time/float(365)

   stock_matrix=stock_price(S0,len(trade_time),0.25,-0.035,trade_time,0,rand,N)
   
   #print stock_matrix
   payoff=[]
   principal=100
   for i in range(N):

       stock_list=stock_matrix[[i]]
       print stock_list
       num1=knock_out(stock_list,len(time),100)
       print num1
       if num1 :
           print "num1"
           print num1
           payoff.append(exp(-rate*np.sum(time[:num1]))*0.14*principal*np.sum(time[:num1]))
       else:
           if (stock_list[0,len(time)-1]/100)>=1:
               print "final ko"
               payoff.append(exp(-rate*371/365)*0.14*principal*371/365)
           else:
               num2=knock_in(stock_list,len(time),100)
               if num2 ==12:
                   print "num2"
                   print num2
                   payoff.append(exp(-rate * 371 / 365) * 0.14 * principal * 371/ 365)
               else:
                   print "kin"
                   print num2
                   payoff.append(exp(-rate * 371 / 365)*((stock_list[0,len(time)-1]/100)-1)* principal*371/365)
   print "payoff"
   print payoff
   print "average"
   print float(sum(payoff))/len(payoff)
   return float(sum(payoff))/len(payoff)

if __name__ == '__main__':


    x_axis = np.linspace(60, 140, 30)
    h=0.01
    ac_delta=[]
    starttime = datetime.datetime.now()
    for i in x_axis:
        rand_m = generate_matrix(12, 300000)
        print i
        A=autocall(i+0.5*h,300000,rand_m)
        B=autocall(i-0.5*h,300000,rand_m)
        ac_delta.append((A-B)/h)
        print A
        print B


    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds
    print  ac_delta
    plt.figure()
    plt.plot(x_axis, ac_delta, "r-", label="true delta")

    print x_axis
    print ac_delta
    plt.xlabel("spot price")
    plt.ylabel("delta")
    plt.title("delta derived from difference method")

    plt.grid(True)
    plt.legend()
    plt.show()


