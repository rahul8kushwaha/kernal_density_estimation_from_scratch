import numpy as np
class KernalDensityEstimation:
    def __init__(self, X, h):
        self.x = X
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.h = h
        self.V = h**self.D
    def K(self, x):
        k = np.absolute(np.subtract(self.x, x))/self.h
        k[k>0.5]=100
        k[k<=0.5]=1
        k[k==100]=0
        k = k.max(axis=1).sum()
        return k
    def probability(self, x):
        p_x = self.K(x)/(self.N*self.V)
        return p_x
    def get_probability(self, x):
        if x.ndim==1:
            x = x.reshape(-1, 1)
        return np.apply_along_axis(lambda a: self.probability(a),1,x)

## sample from normal distribution
# X = np.random.randn(1000000).reshape(-1,1)
# KDE = KernalDensityEstimation(X, 0.01)
# x = np.linspace(-1,1,100)

## sample from beta distribution
# X = np.random.beta(2, 5, 1000000).reshape(-1,1)
# KDE = KernalDensityEstimation(X, 0.01)
# x = np.linspace(0,1,100)

# import matplotlib.pyplot as plt
# plt.plot(x, KDE.get_probability(x))
# plt.show()
