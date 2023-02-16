import numpy as np

learning_rate = 0.001

def function(x):
#    return x ** 2
    return x * np.sin(x ** 2)


def derivative(x):
#    return 2 * x
    return np.sin(x ** 2) + 2 * (x ** 2) * np.cos(x ** 2)


def gradient_descent(domain, x):
    gradient = derivative(x)
    steps = 0

    while domain[0] < x < domain[1]:
        delta = -learning_rate * gradient
        gradient = derivative(x + delta)
        print(f"x:{x}, delta:{delta}, gradient:{gradient}")

        # to detect that minimum was reached we may need to use 'if gradient == 0'
        # because of precision limitations use 'abs(gradient) < 1e-9' instead
        if abs(gradient) < 1e-9:
            break

        x += delta
        steps += 1
        if steps > 1000:
            break

    return x, steps


minim_x, steps = gradient_descent([-3, 3], 1.48)
minim_value = function(minim_x)

print()
print(f"steps:{steps} minim_x:{minim_x} minim_value:{minim_value}")
