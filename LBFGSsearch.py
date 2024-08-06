import numpy as np
from numpy.linalg import norm

# Himmelblau function
def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v

# Himmelblau function derivative
def dfH(X):
    x = X[0]
    y = X[1]
    v = np.copy(X)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
    return v

# Line search using Wolfe conditions
def wolfsearch(f,df, x, p, grad, c1=1e-4, c2=0.9):
    alpha = 1
    phi0 = f(x)
    dphi0 = np.dot(grad, p)
    while True:
        new_x = x + alpha * p
        if f(new_x) > phi0 + c1 * alpha * dphi0 or (f(new_x) >= f(x) and alpha > 1):
            alpha *= 0.5
        elif abs(np.dot(df(new_x), p)) <= -c2 * dphi0:
            return alpha
        elif np.dot(df(new_x), p) >= 0:
            alpha *= 0.5
        else:
            alpha *= 1.1
    return alpha

# Two-loop recursion for L-BFGS
def findp(grad, s_stored, y_stored, m):
    q = grad
    alpha = np.zeros(m)
    rho = np.zeros(m)
    for i in range(m - 1, -1, -1):
        rho[i] = 1.0 / np.dot(y_stored[i], s_stored[i])
        alpha[i] = rho[i] * np.dot(s_stored[i], q)
        q = q - alpha[i] * y_stored[i]
    r = q
    gamma = np.dot(s_stored[-1], y_stored[-1]) / np.dot(y_stored[-1], y_stored[-1])
    r = gamma * r
    for i in range(m):
        beta = rho[i] * np.dot(y_stored[i], r)
        r = r + s_stored[i] * (alpha[i] - beta)
    return r

# L-BFGS optimization
def L_BFGSsearch(f,df, x0, max_it, m):
    d = len(x0)
    grad = df(x0)
    x = np.array(x0)
    x_store = [x0]

    y_stored = []
    d_stored = []
    p = -grad
    alpha = wolfsearch(f,df, x, p, grad)
    d_stored.append(alpha * p)
    grad_old = grad
    x = x + alpha * p
    grad = df(x)
    y_stored.append(grad - grad_old)
    m_ = 1
    neval = 1
    x_store.append(x)

    while norm(grad) > 1e-5:
        if neval > max_it:
            print('Maximum iterations reached!')
            break

        if neval < m:
            p = -findp(grad, np.array(d_stored), np.array(y_stored), m_)
            m_ += 1
        else:
            p = -findp(grad, np.array(d_stored), np.array(y_stored), m)
            d_stored.pop(0)
            y_stored.pop(0)



        alpha = wolfsearch(f,df, x, p, grad)
        d_stored.append(alpha * p)
        grad_old = grad
        x = x + alpha * p
        grad = df(x)
        y_stored.append(grad - grad_old)
        neval += 1
        x_store = np.append(x_store, [x], axis=0)
   
    
    return x, f(x), neval, x_store


print(L_BFGSsearch(fH,dfH, [1, 0], max_it=100, m=10)[0])