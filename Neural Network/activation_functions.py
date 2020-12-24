import numpy as np
import matplotlib.pylab as plt

def identity(x):
    return x


def stepFunc(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return (x > 0) * x


def softmax(x):
    c = np.max(x)
    exp = np.exp(x-c) # Prevent Overflow
    expSum = np.sum(exp)
    return exp / expSum


def plotFunc(func, ax):
    x = np.arange(-10, 10, 0.1)
    y = func(x)
    ax.plot(x, y)


if __name__ == "__main__":
    x = np.array([0.3, 2.9, 4.0])
    print(softmax(x))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Activation functions')
    plotFunc(stepFunc, ax1)
    plotFunc(sigmoid, ax2)
    plotFunc(ReLU, ax3)
    plotFunc(identity, ax4)
    plt.show()