import numpy as np
import sys, os
from error_functions import *
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(xTrain, tTrain), (xTest, tTest) = load_mnist(normalize=True, one_hot_label=True)

trainSize = xTrain.shape[0]
batchSize = 10
batchMask = np.random.choice(trainSize, batchSize)
xBatch = xTrain[batchMask]
tBatch = tTrain[batchMask]

# predict...