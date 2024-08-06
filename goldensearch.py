import numpy as np
def f(x): return (x - 3)**2- 3*x + x**2 - 40

def gsearch(f ,interval,tol):
# GOLDENSECTIONSEARCH searches for minimum using golden section
# 	[xmin, fmin, neval] = GOLDENSECTIONSEARCH(f,interval,tol)
#   INPUT ARGUMENTS
# 	f is a function
# 	interval = [a, b] - search interval
# 	tol - set for bot range and function value
#   OUTPUT ARGUMENTS
# 	xmin is a function minimizer
# 	fmin = f(xmin)
# 	neval - number of function evaluations
#   coords - array of statistics,  coord[i][:] =  [x1,x2, a, b]

    #PLACE YOUR CODE HERE
    F = (1+5**0.5)/2
    
    a = interval[0]
    b = interval[1]
    
    neval = 2
    x1 = b - abs(b-a)/F
    x2 = a + abs(b-a)/F
    
    f1 = f(x1)
    f2 = f(x2)
    
    xmin = 0
    fmin = 0
    coord = [ [x1,x2,a,b]]
    
    while abs(a-b) > tol:
        if f1>=f2:
            a=x1
            xmin = x2
            fmin = f2
            x1 = x2
            x2 = a+abs(b-a)/F
            f1 = f2
            f2 = f(x2)
        else:
            b=x2
            xmin=x1
            fmin = f1
            x2=x1
            x1=b-abs(b-a)/F
            f2 = f1
            f1 = f(x1)
        coord.append([x1,x2,a,b])
    neval+=1
    
    answer_ = [xmin, fmin, neval, coord]
    return answer_


interval = [0, 10]
tol = 1e-5
xmin, fmin, neval, coords = gsearch(f, interval, tol)
print("xmin:", xmin)
print("fmin:", fmin)
print("Number of evaluations:", neval)