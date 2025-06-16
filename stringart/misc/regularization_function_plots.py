import matplotlib.pyplot as plt
import numpy as np


# smooth regularizer
def f1(x):
    return 4 * x * (1 - x)


# abs regularizer
def f2(x):
    return -np.abs(x - 0.5)


x = np.linspace(0, 1, 500)


def plot_func(x, y, filename):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(x, y, color="black")
    ax.grid(True, color="black", linestyle="--", linewidth=0.5)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")

    plt.tight_layout(pad=0)
    plt.show()
    fig.savefig(filename, dpi=300, bbox_inches="tight", pad_inches=0.1, facecolor="white")
    plt.close(fig)


plot_func(x, f1(x), "../../docs/assets/smooth_regularizer_graph.svg")
plot_func(x, f2(x), "../../docs/assets/abs_regularizer_graph.svg")
