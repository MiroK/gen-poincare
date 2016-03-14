from numpy.polynomial.legendre import leggauss
from numpy import sin, cos, pi, sqrt
import numpy as np


def get_basis(N):
    fs = [lambda x: np.ones_like(x)/sqrt(2)]
    fs.extend([lambda x, k=k: sin(k*pi/2*x) for k in range(1, N, 2)])
    fs.extend([lambda x, k=k: cos(k*pi/2*x) for k in range(2, N, 2)])
    return fs

def project(f, basis):
    xq, wq = leggauss(50)
    return np.array([np.sum(wq*f(xq)*v(xq)) for v in basis])

def as_function(c):
    N = len(c)
    def foo(x):
        ans = np.ones_like(x)/sqrt(2)*c[0]
        for i, k in enumerate(range(1, N, 2), 1):
            ans += c[i]*sin(k*pi/2*x)
        for j, k in enumerate(range(2, N, 2), i+1):
            ans += c[j]*cos(k*pi/2*x)
        return ans
    return foo

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f0 = lambda x: x**2

    N = 32
    basis = get_basis(N)
    F = project(f0, basis)
    f = as_function(F)
    
    x = np.linspace(-1, 1, 1000)
    plt.figure()
    plt.plot(x, f0(x), label='f0')
    plt.plot(x, f(x), label='f')
    plt.legend(loc='best')
    plt.show()
