import numpy as np

def finite_difference_method(f, x, h):
    derivative = (f(x + h) - f(x - h)) / (2 * h)
    return derivative




def f1(x):
    return -2 * np.sin(np.sqrt(np.abs(x / 2 + 10))) - x * np.sin(np.sqrt(np.abs(x - 10)))


def df1(x):
    return finite_difference_method(f1, x, 0.01)

def f2(x):
    return x ** 2 - 10 * np.cos(0.5 * np.pi * x) - 110

def df2(x):
    return 2*x+5*np.pi*np.sin(np.pi*x/2)
