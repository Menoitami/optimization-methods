import numpy as np

def f1(x):
    return x**2

def df1(x):
    return 2*x



def armichosearch(f, df, x0, amax, c1,sign):
    alpha = amax
    phi = lambda x: f(x0 + x*sign)
    dphi = lambda x: df(x0 + x*sign)

    phi0 = phi(0)
    dphi0 = dphi(0)
    i = 1
    imax = 1000

    while i < imax:
        if phi(alpha) > phi0 + c1 * alpha * dphi0:
            alpha *= 0.5  # Уменьшаем значение alpha
        else:
            break
        i += 1
    print(i)
    return alpha


x = -130

c1 = 0.0001
amax = 80

sign=1

a = armichosearch(f1, df1, x, amax, c1, sign)

print(x + a*sign)
