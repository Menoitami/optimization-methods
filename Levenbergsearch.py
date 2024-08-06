import numpy as np
from numpy.linalg import norm

np.seterr(all='warn')

def fH(X):
    x = X[0]
    y = X[1]
    v = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    return v

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.zeros_like(X)
    v[0] = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    v[1] = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    return v

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x)**2 + 100 * (y - x**2)**2
    return v

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.zeros_like(X)
    v[0] = -2 * (1 - x) + 200 * (y - x**2) * (-2 * x)
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

def levenberg_marquardt(f, df, x0, tol):
    neval = 0
    kmax = 1000
    deltaX = np.array([1000, 1000])
    coords = []
    a = 100
    nu = 3

    while (norm(deltaX) >= tol) and (neval < kmax):
        gk = df(x0)
        Gesse = H(x0, tol, df)
        I = np.eye(len(x0))
        
      
        deltaX = -np.linalg.lstsq(Gesse + a * I, gk)[0]
        
        xk = x0 + deltaX

        if f(xk) < f(x0):
            a /= nu

        x0 = xk
        neval += 1
        coords.append(x0)

    answer_ = [x0, f(x0), neval, coords]
    return answer_

print(levenberg_marquardt(fR, dfR, [-2, -2], 1e-5)[:3])
