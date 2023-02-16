from numpy import exp, arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, title, show


def function(x, y):
    return (1 - (x ** 2 + y ** 3)) * exp(-(x ** 2 + y ** 2) / 2)


x = arange(-3.0, 3.0, 0.01)
y = arange(-3.0, 3.0, 0.01)

X, Y = meshgrid(x, y)
Z = function(X, Y)

im = imshow(Z, cmap=cm.RdBu)
cset = contour(Z, arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
colorbar(im)

title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
show()
