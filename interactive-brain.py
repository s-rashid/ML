import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BRAIN = 'assets/brain1-v2.pickle'
# BRAIN2 = 'assets/brain2-v1.pickle'
net = pickle.load(open(BRAIN, 'rb'))

print ('Inspecting %s' % BRAIN)
print ('Mean accuracy:', net.mean_accuracy)
print ('Accuracy per training fold:', net.plot)

images, labels = pickle.load(open('.cache/train-images-idx3-ubyte.pickle', 'rb')), pickle.load(
    open('.cache/train-labels-idx1-ubyte.pickle', 'rb'))
test_images, test_labels = pickle.load(open('.cache/t10k-images-idx3-ubyte.pickle', 'rb')), pickle.load(
    open('.cache/t10k-labels-idx1-ubyte.pickle', 'rb'))
images = [image.flatten() for image in images]

test_images = [image.flatten() for image in test_images]


def show(index):
    plt.imshow(np.array(images[index]).reshape(28, 28))
    plt.show()


def test2(index):
    return net.identify(test_images[index]), test_labels[index]


def test(index, plt=False):
    if plt:
        show(index)
    return net.identify(images[index]), labels[index]


def accuracy():
    plot = {"Accuracy": net.accuracy_list}
    print ('mean_accuracy', net.mean_accuracy)
    fig, ax = plt.subplots()
    errors = pd.DataFrame(plot)
    errors.plot(ax=ax)
    plt.show()


def test_some(start, end):
    accuracy = 0.
    for i in range(start, end):
        predicted, output = test(i)
        accuracy += int(predicted == output)
    accuracy /= (end - start)
    return accuracy


def test2_some(start, end):
    accuracy = 0.
    for i in range(start, end):
        predicted, output = test2(i)
        accuracy += int(predicted == output)
    accuracy /= (end - start)
    return accuracy
