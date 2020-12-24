import numpy as np
import sys, os, pickle
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from activation_functions import *

def getData():
    (xTrain, tTrain), (xTest, tTest) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return xTest, tTest


def initNetwork():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

if __name__ == "__main__":
    x, t = getData()
    network = initNetwork()
    batchSize = 100
    count = 0
    for i in range(0, len(x), batchSize):
        xBatch = x[i:i+batchSize]
        yBatch = predict(network, xBatch)
        p = np.argmax(yBatch, axis=1) # 확률이 가장 높은 원소의 인덱스를 얻는다.
        count += np.sum(p == t[i:i+batchSize])

    print("Accuracy:" + str(float(count) / len(x) * 100) + " %")