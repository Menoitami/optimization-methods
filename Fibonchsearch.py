import numpy as np

def f(x): 
    return (x - 3)**2 - 3*x + x**2 - 40

def fibonacci_numbers_until(threshold):
    fib = [0, 1]
    while fib[-1] <= threshold:
        fib.append(fib[-1] + fib[-2])
    return fib

def determine_n_for_fibonacci(a, b, tol):
    threshold = np.abs(b - a) / tol
    fib = fibonacci_numbers_until(threshold)
    return len(fib) - 1, fib

def fibonacci_search(f, interval, tol):
    a = interval[0]
    b = interval[1]
    
    n, fib = determine_n_for_fibonacci(a, b, tol)
    print(n)
    
    
    x1 = a + fib[n - 2] / fib[n] * (b - a)
    x2 = a + fib[n - 1] / fib[n] * (b - a)
    
    f1 = f(x1)
    f2 = f(x2)
    
    neval = 2
    coords = [[x1, x2, a, b]]
    
    for k in range(1, n): # за n шагов
        if f1 > f2:
            a = x1
            x1 = x2
            x2 = a + fib[n-k-1]/fib[n-k]  * (b - a)
            f1 = f2
            f2 = f(x2)
        else:
            b = x2
            x2 = x1
            x1 = a + fib[n-k-2]/fib[n-k]  * (b - a)
            f2 = f1
            f1 = f(x1)
            
        neval += 1
        coords.append([x1, x2, a, b])
    
    xmin = (a + b) / 2
    fmin = f(xmin)
    answer_ = [xmin, fmin, neval, coords]
    
    return answer_

interval = [0, 10]
tol = 1e-5
xmin, fmin, neval, coords = fibonacci_search(f, interval, tol)
print("xmin:", xmin)
print("fmin:", fmin)
print("Number of evaluations:", neval)

