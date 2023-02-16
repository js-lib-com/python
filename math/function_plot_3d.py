from numpy import exp
from pylab import arange, meshgrid
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def function(x, y):
    # return (1 - (x ** 2 + y ** 3)) * exp(-(x ** 2 + y ** 2) / 2)
    return x ** 2 + y ** 2


x = arange(-3.0, 3.0, 0.1)
y = arange(-3.0, 3.0, 0.1)

X, Y = meshgrid(x, y)
Z = function(X, Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       cmap=cm.RdBu, linewidth=0, antialiased=False)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
