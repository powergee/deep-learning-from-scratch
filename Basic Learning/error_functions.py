import numpy as np

def sumSquareError(y, t):
    return 0.5 * np.sum((y-t)**2)


def crossEntropyError(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7
    batchSize = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batchSize


if __name__ == "__main__":
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
    print("y =", y)
    print("Sum Square Error:", sumSquareError(y, t))
    print("Cross Entropy Error:", crossEntropyError(y, t))
    print()

    y = np.array([0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0])
    print("y =", y)
    print("Sum Square Error:", sumSquareError(y, t))
    print("Cross Entropy Error:", crossEntropyError(y, t))