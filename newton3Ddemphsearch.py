import numpy as np
import sys
from numpy.linalg import norm
np.seterr(all='warn')


# F_HIMMELBLAU is a Himmelblau function
# 	v = F_HIMMELBLAU(X)
#	INPUT ARGUMENTS:
#	X - is 2x1 vector of input variables
#	OUTPUT ARGUMENTS:
#	v is a function value
def fH(X):
    x = X[0]
    y = X[1]
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
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
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)

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
    v = (1 - x)**2 + 100*(y - x**2)**2
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
    v[0] = -2 * (1 - x) + 200 * (y - x**2)*(- 2 * x)
    v[1] = 200 * (y - x**2)
    return v


def H(x, tol, df):
    deltaX = 0.1 * tol
    diff = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            x1 = np.array(x, dtype=float)
            x2 = np.array(x, dtype=float)
            x1[i] += deltaX
            x2[i] -= deltaX
            dx1 = df(x1)
            dx2 = df(x2)

            if i == j:
                diff[i, j] = (dx1 - dx2)[i] / (deltaX * 2)
            else:
                diff[i, j] = (dx1 - dx2)[j] / (deltaX * 2)

    return diff


def nsearch(f, df, x0, tol):
# NSEARCH searches for minimum using Newton method
# 	answer_ = nsearch(f, df, x0, tol)
#   INPUT ARGUMENTS
#   f  - objective function
#   df - gradient
# 	x0 - start point
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics

    k = 0
    kmax = 1000
    deltaX = [1000,1000]
    coords = []
    a = 0.5
    while (norm(deltaX) >= tol) and (k < kmax):
        gk = -df(x0)
        xk = x0 - a* np.linalg.lstsq(-H(x0, tol, df), gk)[0] 

        deltaX = x0 - xk
        x0 = np.copy(xk)
        k+=1
        coords.append(x0)


    answer_ = [x0, f(x0), k,  coords]
    return answer_


print(nsearch(fR,dfR,[-2,-2], 1e-5 )[0:2])