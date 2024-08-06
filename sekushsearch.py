import numpy as np

def f(x): return x**2 - 10*np.cos(0.3*np.pi*x) - 20
def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)
def ddf(x): return 2 + 0.9*(np.pi**2)*np.cos(0.3*np.pi*x)
import numpy as np


def sekushsearch(f,df, interval,tol):
# NSEARCH searches for minimum using Newton method
# 	answer_ = nsearch(tol,x0)
#   INPUT ARGUMENTS
# 	tol - set for bot range and function value
#	x0 - starting point
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of x values found during optimization 

    a= interval[0]
    b= interval[1]

    if df(a)*df(b)>0:
        raise ValueError("df must have different signs ")


    
    coords = []
    neval = 1
    
    while abs(b-a) > tol:
        dfb= df(b)
        xk = b - (dfb*(b-a))/(dfb -df(a))
        
        if df(xk)>0:
            b= xk
        else:
            a= xk

        neval+=1
        coords.append(xk)

    return  [xk, f(xk), neval, coords]


interval = [-2, 5]
tol = 1e-5
xmin, fmin, neval, coords = sekushsearch(f,df, interval, tol)
print("xmin:", xmin)
print("fmin:", fmin)
print("Number of evaluations:", neval)