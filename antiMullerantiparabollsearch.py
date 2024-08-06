import numpy as np

def f(x):
    return x**2 - 10*np.cos(0.3*np.pi*x) - 20

def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)

def inverse_parabolic_interpolation(f,df, interval, tol):
    neval = 0
    x0 = sum(interval) / 2
    coords = [interval[0], interval[1], x0]

    while abs(coords[-1] - coords[-2]) > tol:
        x2, x1, x0 = coords[-3], coords[-2], coords[-1]
        fx2, fx1, fx0 = df(x2), df(x1), df(x0)

        # Проверка на деление на ноль
        if (fx0 - fx1) == 0 or (fx0 - fx2) == 0 or (fx1 - fx2) == 0:
            break

        # Вычисление новой точки xk по формуле Лагранжа
        term0 = fx1 * fx2 / ((fx0 - fx1) * (fx0 - fx2))
        term1 = fx0 * fx2 / ((fx1 - fx0) * (fx1 - fx2))
        term2 = fx0 * fx1 / ((fx2 - fx0) * (fx2 - fx1))

        xk = x0 * term0 + x1 * term1 + x2 * term2

        # Проверка на некорректные значения
        if np.isnan(xk) or np.isinf(xk):
            break

        coords.append(xk)
        neval += 1

    xmin = coords[-1]
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    return answer_

interval = [-2, 3]
tol = 1e-5
xmin, fmin, neval, coords = inverse_parabolic_interpolation(f,df, interval, tol)
print("xmin:", xmin)
print("fmin:", fmin)
print("Number of evaluations:", neval)
