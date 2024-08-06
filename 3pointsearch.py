import numpy as np
def f(x): return 2 * (x ** 2) - 9 * x - 31
def df(x): return 4 * x - 9

def f1(x):
    return x**2

def df1(x):
    return 2*x

def _3pointsearch(f,df,interval,tol):

    a = interval[0]
    b= interval[1]

    if df(a)*df(b)>0:
         raise ValueError("Производные f(a) и f(b) должны иметь разные знаки")

    neval=0
    coords=[]
    xm = (a+b)/2
    lk= np.abs(b-a)

    while (lk > tol):
        x1= a + lk/4
        x2= b - lk/4
        f1= f(x1)
        fm= f(xm)
        f2 = f(x2)
        if f1<fm:
            b=xm
            xm=x1
        elif f1>=fm and fm<=f2:
            a= x1
            b=x2
        else:
            a=xm
            xm=x2
        lk = np.abs(b-a)
        neval+=1
        coords.append(xm)

    
    answer_ = [xm, f(xm), neval, coords]
    return answer_


print(_3pointsearch(f,df, [-2,10], 0.001))