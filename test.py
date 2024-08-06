import numpy as np


def finite_difference_method2D(f, x, h=1e-4):
    derivative = (f(x + h) - f(x - h)) / (2 * h)
    return derivative


def numerical_gradient(f, X, h=1e-5):

    first = [X[0], 0]
    second = [0, X[1]]

    grad = np.zeros_like(X)

    firsthp = [X[0]+h, X[1]]
    firsthm = [X[0]-h, X[1]]
    secondhp = [X[0], X[1]+h]
    secondhm = [X[0], X[1]-h]

    grad[0] = (f(firsthp) - f(firsthm)) / (2 * h)
    grad[1] = (f(secondhp) - f(secondhm)) / (2 * h)

    return grad


# Функция f(x) = x^2
def f(x):
    return -2 * np.sin(np.sqrt(np.abs(x / 2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))



# Шаг разностной схемы
h = 0.01


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


x = [2.9,1.9]
print(dfH(x))
print(numerical_gradient(fH,x,h))