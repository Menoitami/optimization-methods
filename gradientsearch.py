import numpy as np
import sys
from numpy.linalg import norm

def sphere(X):
    x = X[0]
    y = X[1]
    v= x**2+y**2
    return v

# Производная (градиент) функции сферы
def sphere_grad(X):
    v = np.copy(X)
    v[0] = X[0]*2
    v[1] = X[1]*2
    return v


#F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def f(X):
    x = X[0]
    y = X[1]
    #Версия питона в codeboard не поддерживает метод библиотеки numpy float_power 
    v = (x**2 + y - 11)**2 + (x + y** 2 - 7)**2
    return v
    
# DF_HIMMELBLAU is a Himmelblau function derivative
# 	v = DF_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a derivative function value
def df(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return v


def grsearch(f,df,x0,tol):


    al = 0.01
    kmax=1000
    neval=1
    coords=[]
    coords.append(x0)
    
    while True:
        gk= -df(x0)
        newX= x0+al*gk
        deltaX=newX-x0
        if (norm(deltaX) < tol) or (neval > kmax):
            answer_ = [newX, f(newX), neval, coords]
            return answer_
        
        neval+=1
        coords.append(newX)
        x0=np.copy(newX)
    


print( grsearch(sphere, sphere_grad,[3,3], 0.000001)[0:2])