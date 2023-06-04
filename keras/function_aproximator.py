import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plot


def function(name, x):
    print(f"Use function {name}.")
    match (name.lower()):
        case "squared_root":
            return np.sqrt(x)

        case "cubic_root":
            return np.cbrt(x)

        case "squared":
            return np.power(x, 2)

        case "cubic":
            return np.power(x, 3)

        case "fall_time":
            return np.sqrt(x / 9.81)

        case "sine":
            return np.sin(x)

        case "cosine":
            return np.cos(x)

        case "squared_sinus":
            return np.sin(np.power(x, 2))

        case "multiplied_square_sinus":
            # x sin(x^2)
            return np.multiply(x, np.sin(np.power(x, 2)))

        case "polynomial":
            # x^2 + x
            return np.add(np.power(x, 2), x)

        case "poly_trigonometric":
            # 0.4x^2 + 0.3x sin(15x) + 0.05 cos(50x) + 0.2
            a = np.multiply(0.4, np.power(x, 2))
            b = np.multiply(np.multiply(0.3, x), np.sin(np.multiply(15, x)))
            c = np.multiply(0.05, np.cos(np.multiply(50, x)))
            d = 0.2
            return np.add(np.add(a, b), np.add(c, d))


def function_domain():
    return range(domain[0] * samples, domain[1] * samples)


domain = (-3, 3)
samples = 100

print("Define the model architecture.")
model = Sequential()
model.add(Dense(1000, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))

print("Compile the model.")
model.compile(loss='mean_squared_error', optimizer='adam')

# print(model.weights)

print("Generate training data.")
x_train = np.linspace(domain[0], domain[1], samples)
y_train = function("multiplied_square_sinus", x_train)

print("Train the model.")
model.fit(x_train, y_train, epochs=1000, verbose=0)

print("Create plot.")
figure, axes = plot.subplots()
axes.spines["left"].set_position("center")
axes.spines["bottom"].set_position("center")
axes.spines["top"].set_color("none")
axes.spines["right"].set_color("none")

# plot.scatter(x_train, y_train, 1)
plot.plot(x_train, y_train, 'b')

x_predict = [x / samples for x in function_domain()]
y_predict = [model.predict(np.array([x / samples]), verbose=0)[0] for x in function_domain()]
plot.plot(x_predict, y_predict, 'r')

plot.show()
