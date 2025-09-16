import numpy as np

def RLS(func, X, Y, forget_factor, d):
    p = len(Y)
    w = np.matrix(np.zeros(p))
    A = d * np.matrix(np.identity(p))

    print(np.multiply(w,X))

    for n in range(0, p):
        x = np.transpose(np.matrix(X[n]))
        y = func(x)
        z = forget_factor**(-1) * A * x
        a = float((1 + x.T*z)**(-1))
        w = w + (y - a * float(x.T * (w + y * z)))
        A -= a * z * z.T