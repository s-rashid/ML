import numpy as np
import random
import tensorflow as tf
import time


class HopfieldNetwork(object):
    def hebbian(self):
        self.W = np.zeros([self.num_neurons, self.num_neurons])
        print("Number of Neurons: ", self.num_neurons)

        for image_vector, _ in self.train_dataset:
            self.W += np.outer(image_vector, image_vector) / self.num_neurons
        np.fill_diagonal(self.W, 0)


    def activate(self, vector):
        changed = True
        while changed:
            changed = False
            indices = list(range(0, len(vector)))
            random.shuffle(indices)

            new_vector = [0] * len(vector)

            for i in range(0, len(vector)):
                neuron_index = indices.pop()

                s = self.compute_sum(vector, neuron_index)
                new_vector[neuron_index] = 1 if s >= 0 else -1
                changed = not np.allclose(vector[neuron_index], new_vector[neuron_index], atol=1e-3)

            vector = new_vector

        return vector

    def compute_sum(self, vector, neuron_index):
        s = 0
        for pixel_index in range(len(vector)):
            pixel = vector[pixel_index]
            if pixel > 0:
                s += self.W[neuron_index][pixel_index]

        return s

    def __init__(self, train_dataset, mode='hebbian'):
        self.train_dataset = train_dataset
        self.num_training = 10
        self.num_neurons = len(self.train_dataset[0][0])

        self._modes = {
            "hebbian": self.hebbian,
        }

        self._modes[mode]()


def flatten(grid):
    out = np.matrix(grid)
    return np.asarray(out.flatten(), dtype=np.int32)[0]

def toImage(inMatrix, labeled = True):
    result = ""
    if labeled:
        result = "Labled: " + str(inMatrix[1]) + '\n'
        inMatrix = inMatrix[0]
    dim = 28

    for i in range(0, dim):
        for j in range(0, dim):
            if(inMatrix[(i*dim) + j] == -1): result += ' .'
            else: result += '**'
        result += '\n'
    return result

def setSign (digit):
    for i in range(digit.size):
        digit[i] = 1 if digit[i] > 0 else -1
    return digit



start = time.time()

mnist = tf.keras.datasets.mnist.load_data()
mnistTrain, mnistTest = mnist[0], mnist[1]
train = []
test = []

trainingSize = 10
count = 0

for i in range(len(mnistTest[0])):
    label = mnistTest[1][i]
    if (label == 1 or label == 5):
        test.append((setSign(flatten(mnistTest[0][i])), label))
        count += 1
        if (count >= 20):
            count = 0
            break


for s in range(1, 10):
    for i in range(len(mnistTrain[0])):
        label = mnistTrain[1][i]
        if (label == 1 or label == 5):
            train.append((setSign(flatten(mnistTrain[0][i])), label))
            count += 1
            if (count >= trainingSize*s):
                break
         
    hebbianNet = HopfieldNetwork(train_dataset=train, mode="hebbian")
    print("Time to train: ", time.time() - start)
    start = time.time()

    minimum = []
    for t in test:
        out = hebbianNet.activate(t[0])
        norm = np.linalg.norm(t[0] - out)
        print ("Norm:", norm)
        print (toImage(t))
        print (toImage(out, labeled=False))
        if (norm==0):
            minimum.append(t)
    
    accuracy = 0
    accuracy = accuracy / len(test)

    print ("For hebbian network trained on: ", trainingSize*s, " images, an accuracy of: ", accuracy, " was obtained on a P/N of: ", (trainingSize*s/784))