import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def   plot_3d_function_and_gradient(f, df, x_range, y_range, resolution=50):
    """
    Строит 3D модель функции и её градиента.

    :param f: Функция, принимающая массив numpy и возвращающая скаляр.
    :param df: Производная функции, принимающая массив numpy и возвращающая массив numpy.
    :param x_range: Кортеж, определяющий диапазон значений по оси X (min, max).
    :param y_range: Кортеж, определяющий диапазон значений по оси Y (min, max).
    :param resolution: Количество точек на оси X и Y.
    """
    
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Вычисление значений функции
    Z = np.array([f(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    
    # Вычисление градиентов
    dX = np.array([df(np.array([x, y]))[0] for x, y in zip(np.ravel(X), np.ravel(Y))])
    dY = np.array([df(np.array([x, y]))[1] for x, y in zip(np.ravel(X), np.ravel(Y))])
    dX = dX.reshape(X.shape)
    dY = dY.reshape(Y.shape)
    
    # Построение графиков
    fig = plt.figure(figsize=(14, 6))
    
    # График функции
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_title('Функция')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # График градиентов
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(X, Y, Z, dX, dY, 0, length=0.1, normalize=True, color='r')
    ax2.set_title('Градиенты')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.show()


def fSphere(X):
    return np.sum(X**2)

def dfSphere(X):
    return 2 * X

def fH(X):
    x = X[0]
    y = X[1]
    v = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2
    return v

def dfH(X):
    x = X[0]
    y = X[1]
    v = np.array([0, 0], dtype=float)
    v[0] = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    v[1] = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)

    return v

def fR(X):
    x = X[0]
    y = X[1]
    v = (1 - x) ** 2 + 100 * (y - x ** 2) ** 2
    return v

def dfR(X):
    x = X[0]
    y = X[1]
    v = np.array([0, 0], dtype=float)
    v[0] = -2 * (1 - x) + 200 * (y - x ** 2) * (- 2 * x)
    v[1] = 200 * (y - x ** 2)
    return v

# Six-hump camel
def fSHC(X):
    x1 = X[0]
    x2 = X[1]
    v = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2
    return v
# Six-hump camel Gradient
def dfSHC(X):
    x1 = X[0]
    x2 = X[1]
    v = np.array([0, 0], dtype=float)
    v[1] = x1 - 8*x2+16*(x2**3) 
    v[0] = 8*x1 - (42*(x1**3))/5 + 2*(x1**5)+x2
    return v

# SUM OF DIFFERENT POWERS FUNCTION
def fSDP(xx):
    d = len(xx)
    total_sum = 0
    for ii in range(d):
        xi = xx[ii]
        new = abs(xi)**(ii + 2)
        total_sum += new
    return total_sum

def dfSDP(xx):
    d = len(xx)
    v = np.array([0, 0], dtype=float)
    v[0] = 3 * abs(xx[0])**(3 - 1) * np.sign(xx[0])
    v[1]= 4 * abs(xx[1])**(4 - 1) * np.sign(xx[1])
    return v



# styblinski–tang
def fST(xx):

    d = len(xx)
    total_sum = 0

    for ii in range(d):
        xi = xx[ii]
        new = xi**4 - 16 * xi**2 + 5 * xi
        total_sum += new

    y = total_sum / 2

    return y

def dfST(xx):

    x0 = xx[0]
    x1 = xx[1]
    v = np.array([0, 0], dtype=float)
    v[0] = 0.5 * (4 * x0**3 - 32 * x0 + 5)
    v[1] = 0.5 * (4 * x1**3 - 32 * x1 + 5)

    return v

#branin
def fBran(xx, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    x1 = xx[0]
    x2 = xx[1]

    term1 = a * (x2 - b*x1**2 + c*x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)

    y = term1 + term2 + s

    return y

def dfBran(xx, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    term1 = 2 * a * (x2 - b*x1**2 + c*x1 - r) * (-2*b*x1 + c)
    term2 = - s * (1 - t) * np.sin(x1)

    v[0] = term1 + term2

    v[1] = 2 * a * (x2 - b*x1**2 + c*x1 - r)

    return v

#camel3
def fc3(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = 2 * x1**2
    term2 = -1.05 * x1**4
    term3 = x1**6 / 6
    term4 = x1 * x2
    term5 = x2**2

    y = term1 + term2 + term3 + term4 + term5

    return y

def dfc3(xx):
    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    v[0] = 4*x1 - 4.2*x1**3 + x1**5 + x2
    v[1] = x1 + 2*x2

    return v
#mccormick

def fmc(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = np.sin(x1 + x2)
    term2 = (x1 - x2)**2
    term3 = -1.5 * x1
    term4 = 2.5 * x2

    y = term1 + term2 + term3 + term4 + 1

    return y

def dfmc(xx):
    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    v[0] = np.cos(x1 + x2) + 2 * (x1 - x2) - 1.5
    v[1] = np.cos(x1 + x2) - 2 * (x1 - x2) + 2.5
    return v

#zakharov
def fz(xx):
    d = len(xx)
    sum1 = sum(xi**2 for xi in xx)
    sum2 = sum(0.5 * i * xi for i, xi in enumerate(xx, start=1))
    y = sum1 + sum2**2 + sum2**4
    return y

def dfz(xx):
    x1 = xx[0]
    x2 = xx[1]
    sum2 = 0.5 * (1*x1 + 2*x2)
    v = np.array([0, 0], dtype=float)
    v[0] = 2*x1 + 2*sum2
    v[1] = 2*x2 + 2*sum2
    return v

# booth
def fboo(xx):


    x1 = xx[0]
    x2 = xx[1]

    term1 = (x1 + 2*x2 - 7)**2
    term2 = (2*x1 + x2 - 5)**2

    y = term1 + term2

    return y

def dfboo(xx):
    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    v[0] = 10*x1 + 8*x2 - 34
    v[1] = 8*x1 + 10*x2 - 38

    return v
# matyas
def fmat(xx):
    x1 = xx[0]
    x2 = xx[1]
    term1 = 0.26 * (x1**2 + x2**2)
    term2 = -0.48 * x1 * x2
    y = term1 + term2
    return y

def dfmat(xx):


    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    v[0] = 0.52 * x1 - 0.48 * x2
    v[1] = 0.52 * x2 - 0.48 * x1

    return v

# bukin n 6
def fbukin(xx):


    x1 = xx[0]
    x2 = xx[1]
    

    term1 = 100 * np.sqrt(np.abs(x2 - 0.01*x1**2))
    term2 = 0.01 * np.abs(x1 + 10)

    y = term1 + term2

    return y


def dfbukin(xx):

    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    term1_derivative = 100 * (0.5 * x1 / np.sqrt(np.abs(x2 - 0.01 * x1**2)))
    term2_derivative = 0.01 * np.sign(x1 + 10)

    v[0] = term1_derivative + term2_derivative

    term3_derivative = 100 * ((x2 - 0.01 * x1**2) / np.sqrt(np.abs(x2 - 0.01 * x1**2)))

    v[1] = term3_derivative

    return v

#holder

def fholder(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
    y = -np.abs(fact1 * fact2)
    return y

def dfholder(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
    v = np.array([0, 0], dtype=float)
    v[0]= -np.sign(fact1 * fact2) * (np.cos(x1) * np.cos(x2) * fact2 + x1 / np.sqrt(x1**2 + x2**2) * np.sin(x1) * np.sin(x2) * fact2)
    v[1] = np.sign(fact1 * fact2) * (np.sin(x1) * np.sin(x2) * fact2 - x2 / np.sqrt(x1**2 + x2**2) * np.cos(x1) * np.cos(x2) * fact2)
    return v

# cross in tray
def fcit(xx):
    x1 = xx[0]
    x2 = xx[1]
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    y = -0.0001 * (np.abs(fact1 * fact2) + 1)**0.1
    return y

def dfcit(xx):


    x1 = xx[0]
    x2 = xx[1]
    v = np.array([0, 0], dtype=float)
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    sgn_fact1 = np.sign(fact1)

    v[0] = -0.0001 * 0.1 * sgn_fact1 * np.cos(x1) * np.sin(x2) * (np.abs(fact1 * fact2) + 1)**(-0.9) * fact2 * (100 / (2 * np.pi * np.sqrt(x1**2 + x2**2)))
    v[1] = -0.0001 * 0.1 * sgn_fact1 * np.sin(x1) * np.cos(x2) * (np.abs(fact1 * fact2) + 1)**(-0.9) * fact2 * (100 / (2 * np.pi * np.sqrt(x1**2 + x2**2)))

    return v
# eggholder

def egg(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1/2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    
    y = term1 + term2
    return y


def degg(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    sqrt_term1 = np.sqrt(np.abs(x2 + x1/2 + 47))
    sqrt_term2 = np.sqrt(np.abs(x1 - (x2 + 47)))
    
    # Partial derivative with respect to x1
    dy_dx1 = -np.cos(sqrt_term1) * 0.5 * (x1 + 2*x2 + 94) / sqrt_term1 - np.sin(sqrt_term2)
    
    # Partial derivative with respect to x2
    dy_dx2 = -np.cos(sqrt_term1) * 0.5 * (x1 + 2*x2 + 94) / sqrt_term1 - np.sin(sqrt_term1) * 0.5 * (x1 + 2*x2 + 47) / sqrt_term1
    
    return np.array([dy_dx1, dy_dx2])

#levy
def levy(xx):
    d = len(xx)
    w = 1 + (np.array(xx) - 1) / 4

    term1 = (np.sin(np.pi * w[0]))**2
    term3 = (w[d-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[d-1]))**2)
    
    sum_terms = 0
    for ii in range(d-1):
        wi = w[ii]
        new_term = (wi - 1)**2 * (1 + 10 * (np.sin(np.pi * wi + 1))**2)
        sum_terms += new_term

    y = term1 + sum_terms + term3
    return y

def dflevy(xx):
    x1, x2 = xx[0], xx[1]
    w1 = 1 + (x1 - 1) / 4
    w2 = 1 + (x2 - 1) / 4
    
    # Partial derivative w.r.t x1
    dy_dx1 = (1/2) * (
        2 * np.pi * np.sin(np.pi * w1) * np.cos(np.pi * w1) +
        2 * (w1 - 1) * (1 + 10 * np.sin(np.pi * w1 + 1)**2) +
        20 * np.pi * (w1 - 1)**2 * np.sin(np.pi * w1 + 1) * np.cos(np.pi * w1 + 1)
    )
    
    # Partial derivative w.r.t x2
    dy_dx2 = (1/2) * (
        2 * (w2 - 1) * (1 + 10 * np.sin(np.pi * w2 + 1)**2) +
        20 * np.pi * (w2 - 1)**2 * np.sin(np.pi * w2 + 1) * np.cos(np.pi * w2 + 1) +
        2 * (w2 - 1) * (1 + np.sin(2 * np.pi * w2)**2) +
        4 * np.pi * (w2 - 1)**2 * np.sin(2 * np.pi * w2) * np.cos(2 * np.pi * w2)
    )
    
    return np.array([dy_dx1, dy_dx2])

# ackley
def ackley(xx, a=20, b=0.2, c=2*math.pi):
    d = len(xx)

    sum1 = 0.0
    sum2 = 0.0
    for x in xx:
        sum1 += x**2
        sum2 += math.cos(c * x)

    term1 = -a * math.exp(-b * math.sqrt(sum1 / d))
    term2 = -math.exp(sum2 / d)

    y = term1 + term2 + a + math.exp(1)

    return y


def dfackley(xx, a=20, b=0.2, c=2*np.pi):
    d = len(xx)
    
    sum1 = 0.0
    sum_cos = 0.0
    for x in xx:
        sum1 += x**2
        sum_cos += np.cos(c * x)
    
    sum_sqrt = np.sqrt(sum1 / d)
    
    df_dx = [0.0] * d
    for i in range(d):
        df_dx[i] = (-2 * a * xx[i] / d) * np.exp(-b * sum_sqrt) \
                   - b * xx[i] / (np.sqrt(d) * sum_sqrt) * np.exp(-b * sum_sqrt) * (sum1 / d)**(-0.5) \
                   - (1 / d) * np.sin(c * xx[i]) * np.exp(sum_cos / d)

    return df_dx



def numerical_gradient(f, X, h=1e-4):
    grad = [0.0] * len(X)
    
    for i in range(len(X)):
        X_hp = X.copy()
        X_hm = X.copy()
        X_hp[i] += h
        X_hm[i] -= h
        
        grad[i] = (f(X_hp) - f(X_hm)) / (2 * h)
        
    return grad

def numerical_gradient_4th_order(f, X, h=1e-4):
    grad = [0.0] * len(X)
    
    for i in range(len(X)):
        X_hp2 = X.copy()
        X_hp1 = X.copy()
        X_hm1 = X.copy()
        X_hm2 = X.copy()
        
        X_hp2[i] += 2 * h
        X_hp1[i] += h
        X_hm1[i] -= h
        X_hm2[i] -= 2 * h
        
        grad[i] = (-1/12*f(X_hp2)+2/3*f(X_hp1)- 2/3*f(X_hm1)+1/12*f(X_hm2))/h
        
    return grad

def complex_function(X):
    x = X[0]
    y = X[1]
    return np.exp(x * y) + x / y + np.sqrt(x)

def complex_gradient(X):
    x = X[0]
    y = X[1]
    grad = np.zeros(2)
    grad[0] = y * np.exp(x * y) + 1 / y + 1 / (2 * np.sqrt(x))
    grad[1] = x * np.exp(x * y) - x / (y**2)
    return grad


# plot_3d_function_and_gradient(fbukin, dfbukin, x_range=(-10, 10), y_range=(-10, 10))

x= [2,5]
print(complex_gradient(x))
print(numerical_gradient_4th_order(complex_function, x))
