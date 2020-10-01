import matplotlib.pyplot as plt
import numpy as np

def step(x):
    return np.where(x < 0, 0, 1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hyperbolic(x):
    return 2*sigmoid(2*x) - 1

def relu(x):
    return np.maximum(0, x)

def leakyrelu(x, alpha):
    return np.maximum(alpha*x,x)

def elu(x, alpha):
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)

def make_plot(x, y, name):
    plt.plot(x, y, linewidth=1.5)
    plt.title(name)
    plt.grid()
    plt.axis([-5, 5, -1.5, 1.5])
    #plt.plot((-5, 5), (0, 0), 'k-')
    #plt.plot((0, 0), (-1.5, 1.5), 'k-')
    plt.savefig(name[:6], dpi=300)
    plt.clf()


if __name__ == '__main__':
    x = np.linspace(-6, 6, 800)

    y = step(x)
    make_plot(x, y, "Step Function")

    y = sigmoid(x)
    make_plot(x, y, "Sigmoid Function")

    y = hyperbolic(x)
    make_plot(x, y, "Hyperbolic Tangent Function")

    y = relu(x)
    make_plot(x, y, "ReLU Function")

    y = leakyrelu(x, 0.1)
    make_plot(x, y, "Leaky ReLU Function with a = 0.01")

    y = elu(x, 1)
    make_plot(x, y, "ELU Function with a = 1")



