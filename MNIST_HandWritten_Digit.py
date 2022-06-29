# Handwritten digit classification using SVD.
from numpy import *
from scipy.linalg import *
#from mnist import MNIST
from mlxtend.data import loadlocal_mnist
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

#PLEASE PUT PHAT OF MNIST FILE TO WORK
train_image, train_label = loadlocal_mnist(images_path="/content/drive/MyDrive/Colab Notebooks/data/train-images-idx3-ubyte", labels_path="/content/drive/MyDrive/Colab Notebooks/data/train-labels-idx1-ubyte")
test_image, test_label = loadlocal_mnist(images_path="/content/drive/MyDrive/Colab Notebooks/data/t10k-images-idx3-ubyte", labels_path="/content/drive/MyDrive/Colab Notebooks/data/t10k-labels-idx1-ubyte")
train_image = np.asarray(train_image)
train_label = np.asarray(train_label)
test_image = np.asarray(test_image)
test_label = np.asarray(test_label)
values = {}


def resizeVec (inputMat, dim):
    result = np.tile(0, (dim, dim))
    for i in range(0, dim):
        for j in range(0, dim):
            result[i, j] = inputMat[(dim*i)+j]
    return result
            

def columnVec (inputMat, dim):
    result = np.arange(dim**2)
    for i in range(0, dim):
        for j in range(0, dim):
            result[(dim*j)+i] = inputMat[i, j]
    return result

def addColumn(inputMat, inColumn, index):
    for i in range(0, inputMat.shape[0]):
        inputMat[i, index] = inColumn[i]
    return inputMat

def translate(val, minimum, maximum):
    valRange = maximum-minimum
    scaled = float(val - minimum)/float(valRange)
    return scaled

for j in range (0, 10):
    values[j] = np.tile(0, (train_image.shape[1], np.bincount(train_label)[j]))

for j in range(0, 10):
    column = 0
    for i in range(0, len(train_image[:, 0])):
        if(train_label[i] == j):
            addColumn(values[j], columnVec(resizeVec(train_image[i], 28), 28), column)
            column += 1



def evalProc(basis):
    tables = []
    for i in range(0, 10):
        val = U, S, V = scipy.linalg.svd(values[i][:, 0:basis], False)
        tables.append(val)

    id1 = np.identity(784)
    rtnVal = []

    for i in range(0, 500):
        toUse = columnVec(resizeVec(test_image[i], 28), 28)
        results = []

        n = 0
        v = 10000
        
        for j in range(0, 10):
            val = scipy.linalg.norm(np.dot((id1 - np.dot(tables[j][0], tables[j][0].T)), toUse), 2)
            results.append(val)
            if (val < v):
                n = j
                v = val

        if (n == test_label[i]):
            rtnVal.append(True)
        else:
            rtnVal.append(False)

    return rtnVal

bases = []
percentile = []
print("Testing on Basis:")
for i in range(1, 51):
    if (i%2 == 0):    
        l = evalProc(i)
        numTests = 0
        correct = 0
        for j in range(0,len(l)):
            numTests += 1
            if (l[j]):
                correct += 1
        bases.append(i)
        percentile.append(correct/numTests)

plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')
plt.xlabel("Number of Basis Images")
plt.ylabel("Classification Percentage")    
plt.plot(bases, percentile, marker='', color=palette(i), linewidth=1, alpha=0.9, label=0)
plt.show()