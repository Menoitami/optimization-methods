import numpy as np
import sys
from numpy.linalg import norm

# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v


# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v


# F_ROSENBROCK is a Rosenbrock function
# 	v = F_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v


# DF_ROSENBROCK is a Rosenbrock function derivative
# 	v = DF_ROSENBROCK(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v

def gsearch(f,interval,tol):

    F = (1+5**0.5)/2

    a = interval[0]
    b = interval[1]

    k=2

    x1=b-(b-a)/F #lambda
    x2=a+(b-a)/F

    f1=f(x1)
    f2=f(x2)

    xmin=0
    
    coord =  [[x1,x2, a, b]]
    fmin=0

    while (b-a)>tol :
        if f1>=f2:
            a=x1
            xmin=x2
            x1=x2
            x2=a+(b-a)/F
            fmin=f2
            f1=f2
            f2=f(x2)
        else:
            b=x2
            xmin=x1
            x2=x1
            x1=b-(b-a)/F
            fmin=f1
            f2=f1
            f1=f(x1)
        
        coord.append([x1,x2, a, b])
        k+=1

    answer_ = xmin
    return answer_

def sdsearch(f, df, x0, tol):

    al = 0.01
    kmax=1000
    neval=1
    coords=[x0]
    
    
    while True:

        gk= -df(x0)
        
        f1dim=lambda a:f(x0+ a*gk)
        al= gsearch(f1dim,[0,1], tol )

        newX= x0+al*gk
        deltaX=newX-x0

        if (norm(deltaX) < tol) or (neval > kmax):
            answer_ = [newX, f(newX), neval, coords]
            return answer_
        
        neval+=1
        coords.append(newX)
        x0=np.copy(newX)
    
