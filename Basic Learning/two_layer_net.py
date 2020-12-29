import sys, os
import numpy as np
sys.path.append(os.pardir)
from shared.activation_functions import *
from shared.error_functions import *
from dataset.mnist import load_mnist


def getNumerGrad(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for i in range(x.size):
        tmp = x[i]

        # f(x+h) 계산
        x[i] = tmp + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[i] = tmp - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2*h)
        x[i] = tmp

    return grad


class TwoLayerNet:
    def __init__(self, inputSize, hiddenSize, outputSize, weightInitStd = 0.01):
        self.params = {}
        self.params["W1"] = weightInitStd * np.random.randn(inputSize, hiddenSize)
        self.params["b1"] = np.zeros(hiddenSize)
        self.params["W2"] = weightInitStd * np.random.randn(hiddenSize, outputSize)
        self.params["b2"] = np.zeros(outputSize)

    def predict(self, x):
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        input1 = np.dot(x, W1) + b1
        output1 = sigmoid(input1)
        input2 = np.dot(output1, W2) + b2
        output2 = softmax(input2)

        return output2
    
    def getLoss(self, input, label):
        output = self.predict(input)
        return crossEntropyError(output, label)
    
    # 추측한 결과들 중 정답인 것들의 비율을 반환
    def getAccuracy(self, input, label):
        output = self.predict(input)
        outMax = np.argmax(output, axis=1)
        labMax = np.argmax(label, axis=1)

        return np.sum(outMax == labMax) / float(input.shape[0])

    def getNumerGrad(self, input, label):
        lossW = lambda W: self.getLoss(input, label)

        grads = {}
        grads['W1'] = getNumerGrad(lossW, self.params['W1'])
        grads['b1'] = getNumerGrad(lossW, self.params['b1'])
        grads['W2'] = getNumerGrad(lossW, self.params['W2'])
        grads['b2'] = getNumerGrad(lossW, self.params['b2'])

        return grads


if __name__ == "__main__":
    net = TwoLayerNet(784, 100, 10)
    print(net.params['W1'].shape)
    print(net.params['b1'].shape)
    print(net.params['W2'].shape)
    print(net.params['b2'].shape)