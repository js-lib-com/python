import numpy as np
import matplotlib.pylab as graph
import scipy.interpolate as interp

y = [8928, 9073, 9133, 9204, 9213, 9242, 9286, 9267, 9257, 9279, 9266, 9272, 9283, 9279, 9249, 9347, 9311, 9313, 9326,
     9338, 9310, 9336, 9338, 9343, 9334, 9317, 9356, 9279, 9337, 9308]
x = range(0, len(y))

spline_function = interp.make_interp_spline(x, y)
spline_x = np.linspace(0, len(y), 500)
spline_y = spline_function(spline_x)

graph.plot(x, y, 'o')
graph.plot(spline_x, spline_y)
graph.show()
