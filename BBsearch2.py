import numpy as np
import sys
from numpy.linalg import norm


def goldensectionsearch(f, interval, tol):
    golden_ratio = (1 + 5 ** 0.5) / 2

    a, b = interval
    x1 = b - (b - a) / golden_ratio
    x2 = a + (b - a) / golden_ratio

    f1 = f(x1)
    f2 = f(x2)

    while (b - a) > tol:
        if f1 >= f2:
            a = x1
            x1 = x2
            x2 = a + (b - a) / golden_ratio
            f1 = f2
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            x1 = b - (b - a) / golden_ratio
            f2 = f1
            f1 = f(x1)

    return (a + b) / 2



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


def bbsearch(f, df, x0, tol):

# BBSEARCH searches for minimum using stabilized BB1 method
# 	answer_ = bbsearch(f, df, x0, tol)
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
    al = 0.01
    kmax = 1000
    neval = 1
    coords = [x0]
    d = 0.1

    gk = -df(x0)
    f1dim = lambda a: f(x0 + a * gk)
    al = goldensectionsearch(f1dim, [0, 1], tol)
    print(al)
    
    dX = tol+100
    while  (norm(dX) >= tol) and (neval <= kmax):
        g0 = df(x0)
        x1 = x0 - al * g0
        g1 = df(x1)
        dX = x1 - x0
        dG = g1 - g0

        al1 = np.dot(dX.T, dG) / np.dot(dG.T, dG)
        al2 = d / norm(g1)

        # al = min(al1, al2)
        al = al1 if (al1<al2) else al2
        x0 = x1
        coords.append(x0)
        neval += 1
        
    return [x1, f(x1), neval, coords]




print("Rosenbrock function:")
x0 = np.array([[2], [-1]])
tol = 1e-9
[xmin, f, neval, coords] = bbsearch(fR, dfR, x0, tol)  # r - функция Розенброка
print(xmin, f, neval)