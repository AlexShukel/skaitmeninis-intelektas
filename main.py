import random
import matplotlib.pyplot as plt
import numpy as np 

random.seed(21)

def random_coord(lower, upper):
    return random.uniform(lower, upper)

def generate_points(lower, upper):
    return list(map(lambda _: [random_coord(lower, upper), random_coord(lower, upper)], [None for _ in range(10)]))

A = generate_points(0, 0.5)
B = generate_points(0.5, 1)

def plot_points(weight_sets):
    fig, ax = plt.subplots()
    ax.scatter(list(map(lambda x: x[0], A)), list(map(lambda x: x[1], A)), color="red", marker=".", label="Klasė A")
    ax.scatter(list(map(lambda x: x[0], B)), list(map(lambda x: x[1], B)), color="blue", marker=".", label="Klasė B")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Tiesiškai atskiriamos klasės")

    x_values = np.linspace(0.1, 0.9, 100)
    i = 1
    colors = ["green", "brown", "black"]
    for weights in weight_sets:
        y_values = - (weights[1] / weights[2]) * x_values - (weights[0] / weights[2])
        ax.plot(x_values, y_values, linewidth=2, color=colors[i - 1], label=f"Skiriančioji tiesė {i}")
        i += 1

    ax.legend()
    plt.show()

# Weights w0, w1, w2, where w0 is a bias
def generate_random_weights():
    lower = -5
    upper = 5
    return [random.uniform(lower, upper), random.uniform(lower, upper), random.uniform(lower, upper)]

def neuron(point, weights):
    return 1 * weights[0] + point[0] * weights[1] + point[1] * weights[2]

def threshold_activation(a):
    if a >= 0:
        return 1

    return 0

def sigmoid_activation(a):
    return 1 / (1 + np.exp(-a))

def evaluate(point, activation, weights):
    return round(activation(neuron(point, weights)))

def are_weights_valid(activation, weights):
    global A, B

    return all(evaluate(point, activation, weights) == 0 for point in A) and all(evaluate(point, activation, weights) == 1 for point in B)

def find_valid_weights(activation):
    weights = generate_random_weights()

    while not are_weights_valid(activation, weights):
        weights = generate_random_weights()

    return weights

# Find 3 sets of weights using threshold_activation
random.seed(55)
w11 = find_valid_weights(threshold_activation)
random.seed(69)
w12 = find_valid_weights(threshold_activation)
random.seed(96)
w13 = find_valid_weights(threshold_activation)

# Find 3 sets of weights using sigmoid_activation
w21 = find_valid_weights(sigmoid_activation)
w22 = find_valid_weights(sigmoid_activation)
w23 = find_valid_weights(sigmoid_activation)

plot_points([w11, w12, w13])
print(w11, w12, w13)
print(w21, w22, w23)
