import random
import matplotlib.pyplot as plt
import matplotlib.markers as markers

random.seed(21)

def random_coord(lower, upper):
    return random.uniform(lower, upper)

def generate_points(lower, upper):
    return list(map(lambda _: [random_coord(lower, upper), random_coord(lower, upper)], [None for _ in range(10)]))

A = generate_points(0, 0.5)
B = generate_points(0.5, 1)

def plot_points():
    fig, ax = plt.subplots()
    ax.scatter(list(map(lambda x: x[0], A)), list(map(lambda x: x[1], A)), color="red", marker=".", label="Klasė A")
    ax.scatter(list(map(lambda x: x[0], B)), list(map(lambda x: x[1], B)), color="blue", marker=".", label="Klasė B")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Tiesiškai atskiriamos klasės")

    ax.legend()
    plt.show()

plot_points()
