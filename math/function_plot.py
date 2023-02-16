import matplotlib.pyplot as plot
import numpy as np

x = np.linspace(-3, 3, 10000)
y = x ** 2
y = x * np.sin(x ** 2)
# y = np.sin(x)
# y = 1.0 / (1.0 + np.exp(-x))
# y = np.sin(x)
# y = x ** 3 - 12 * x + 2

fig = plot.figure()
ax = fig.add_subplot(1, 1, 1)

ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position("zero")
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

# ax.xaxis.set_ticks_position("bottom")
# ax.yaxis.set_ticks_position("left")

plot.plot(x, y, 'r')
plot.show()
