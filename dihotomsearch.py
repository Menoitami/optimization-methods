import numpy as np
def f(x): return 2 * (x ** 2) - 9 * x - 31
def df(x): return 4 * x - 9

def f1(x):
    return x**2

def df1(x):
    return 2*x

def bsearch(f,df,interval,tol):

    a = interval[0]
    b= interval[1]
    neval=0
    coords=[]
    if df(a)*df(b)>0:
         raise ValueError("Производные f(a) и f(b) должны иметь разные знаки")
    

    while (np.abs(a-b) > tol) and (np.abs(df(a)) > tol):
        x1 = (a+b)/2
        if df(x1)>0:
            b=x1
        else:
            a=x1
        neval+=1
        coords.append(x1)

    
    answer_ = [x1, f(x1), neval, coords]
    return answer_


print(bsearch(f1,df1, [-2,10], 0.001))