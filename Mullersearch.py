import numpy as np

def f(x): return x**2 -  10*np.cos(0.3*np.pi*x) - 20
def df(x): return 2*x + 3*np.pi*np.sin(0.3*np.pi*x)

def g(X):
    if len(X)==2:
        R=(df(X[1])-df(X[0]))/(X[1]-X[0])
    
    if len(X)==3:
       R=(g([X[1],X[2]])-g([X[0],X[1]]))/(X[2]-X[0])
    
    return R

def w(X):
    return g([X[-2], X[-1]])+g([X[-3], X[-1]])- g([X[-3], X[-2]])
    

def Mullersearch(f,df,interval,tol):
# SSEARCH searches for minimum using secant method
#   answer_ = ssearch(interval,tol)
#   INPUT ARGUMENTS
#   interval = [a, b] - search interval
#   tol - set for bot range and function value
#   OUTPUT ARGUMENTS
#   answer_ = [xmin, fmin, neval, coords]
#   xmin is a function minimizer
#   fmin = f(xmin)
#   neval - number of function evaluations
#   coords - array of x values found during optimization    

  #PLACE YOUR CODE HERE
   
    
    neval=0
    x0=sum(interval)/2
    coords=[interval[0], interval[1], x0]
    diff = 1000
    
    while (np.abs(diff)) > tol and (np.abs(coords[-3] - coords[-2])) > tol:
        diff = df(coords[-1])
        wc= w(coords)
        s= np.sqrt(wc**2-4*diff*g([coords[-3],coords[-2],coords[-1]]))
        denominatorPlus=wc+s
        denominatorMinus=wc-s
        if abs(denominatorPlus)>abs(denominatorMinus):
            xnew=coords[-1]-(2*diff/denominatorPlus)
        else:
            xnew=coords[-1]-(2*diff/denominatorMinus)
            
        coords.append(xnew)
        neval+=1
        
        
    xmin=coords[-1]
    fmin=f(coords[-1])
    answer_ = [xmin, fmin, neval, coords]
    return answer_


interval = [-2, 5]
tol = 1e-5
xmin, fmin, neval, coords = Mullersearch(f,df, interval, tol)
print("xmin:", xmin)
print("fmin:", fmin)
print("Number of evaluations:", neval)