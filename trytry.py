import numpy as np
from scipy.stats import norm
from math import exp, sqrt
import random
import matplotlib.pyplot as plt
import datetime


np.random.seed(20180721)

# to fix one random matrix fix the path
def generate_matrix(ob_time,N):
    result=np.random.normal(0,1,[N,ob_time])
    return result

# after we have the random matrix, we put it together with the stock initial price column, then we use GBM to compute the following price, later we delete the initial stock price
# as it is useless
def stock_price(st,ob_time,vol,r,time,div,rand_matrix,n):
    s_price=np.ones([n,1])
    s_price=s_price*st
    s_price=np.column_stack((s_price,rand_matrix))
    for i in range(ob_time):
        s_price[:,[i+1]]=s_price[:,[i]]*np.exp((r-div-0.5*vol**2)*time[i]+vol*sqrt(time[i])*rand_matrix[:,[i]])
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
   now_time=[15,35,28,35]
   rate=0.03
   time=np.array(now_time)
   time=time/float(365)

   stock_matrix=stock_price(S0,len(time),0.05,0.02,time,0,rand,N)
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
               payoff.append(exp(-rate*113/365)*0.14*principal*113/365)
           else:
               num2=knock_in(stock_list,len(time),100)
               if num2 ==4:
                   print "num2"
                   print num2
                   payoff.append(exp(-rate * 113 / 365) * 0.14 * principal * 113/ 365)
               else:
                   print "kin"
                   print num2
                   payoff.append(exp(-rate * 113 / 365)*((stock_list[0,len(time)-1]/100)-1)* principal * 113 / 365)
   print "payoff"
   print payoff
   print "average"
   print float(sum(payoff))/len(payoff)
   return float(sum(payoff))/len(payoff)

if __name__ == '__main__':


    x_axis = np.linspace(90, 110, 10)
    h=0.0001
    ac_delta=[]
    starttime = datetime.datetime.now()
    for i in x_axis:
        rand_m = generate_matrix(4, 500000)
        print rand_m

        print i
        A=autocall(i+h,500000,rand_m)
        B=autocall(i,500000,rand_m)
        ac_delta.append(-(A-B)/h)
        print A
        print B

        
    endtime = datetime.datetime.now()
    print (endtime - starttime).seconds
    plt.figure()
    plt.plot(x_axis, ac_delta, "r-", label="true delta")

    plt.xlabel("spot price")
    plt.ylabel("delta")
    plt.title("delta derived from difference method")

    plt.grid(True)
    plt.legend()
    plt.show()


