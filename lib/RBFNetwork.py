import random
from numpy import mean, zeros, argmax, std
from numpy.linalg import norm
from scipy.cluster.vq import kmeans
from sklearn.model_selection import KFold

from Neuron import Neuron

TOLERANCE = 0.01
MAX_ITERS = 3

class RBFNetwork(object):
    def __init__(self, data, k=20, output=10, learning_rate=0.125):
        self.learning_rate = learning_rate
        self.hidden_layer = []
        self.output_layer = [0] * output
        self.k = k
        self.init_hidden_layer(k, data, output, learning_rate)

    def init_hidden_layer(self, k, data, output, learning_rate):
        centroids, distortion = kmeans(data, k)
        clusters = compute_clusters(centroids, data)
        empty_clusters = list()

        for i in range(0, len(clusters)):
            if len(clusters[i]) == 0:
                empty_clusters.append(i)

        for i in range(0, len(centroids)):
            if i in empty_clusters:
                continue

            sigma = compute_sigma(centroids[i], clusters[i])
            beta = compute_beta(sigma)
            self.hidden_layer.append(Neuron(output, beta, centroids[i], leaning_rate=learning_rate))

    def train(self, tset, tlabels):
        accuracy_list = []
        self.mean_accuracy = 0
        gym = zip(tset, tlabels)
        random.shuffle(gym)

        kfold = KFold(n_splits=10)
        count = 0
        for training_indices, testing_indices in kfold.split(gym):
            training_set = [gym[i] for i in training_indices]
            testing_set = [gym[i] for i in testing_indices]

            for neuron in self.hidden_layer:
                neuron.reset_weights()

            prev_accuracy = 0.0
            biggest_accuracy = 0.0
            iters = 0
            for i in range(0, 100):
                for image, label in training_set:
                    self.feed_input(image)
                    self.forward_propagate()
                    self.back_propagate(self.target_label_as_vector(label))

                num_correct = 0.
                for test_image, test_label in testing_set:
                    prediction = self.identify(test_image)
                    num_correct += int(prediction == test_label)


                accuracy = num_correct / len(testing_set)
                biggest_accuracy = accuracy if accuracy > biggest_accuracy else biggest_accuracy
                iters = iters + 1 if accuracy - prev_accuracy <= TOLERANCE else 0
                prev_accuracy = accuracy

                print ("\n %.2f accuracy on iteration %s" % (accuracy, i))
                print ("Completed {}/3 non-improving iterations".format(iters))

                if iters == MAX_ITERS:
                    break

            accuracy_list.append(biggest_accuracy)
            self.mean_accuracy = mean(accuracy_list)
            count += 1

            print ("{} / {} folds completed.".format(count, kfold.get_n_splits()))
            print ("{0:.2f} converged accuracy".format(self.mean_accuracy))

        self.accuracy_list = accuracy_list
        self.plot = {"Accuracy": accuracy_list}
        self.confidence_interval = (
            self.mean_accuracy - 3 * std(self.accuracy_list),
            self.mean_accuracy + 3 * std(self.accuracy_list)
        )

    def feed_input(self, vector):
        # Pass the input vector to each neuron in the network
        for neuron in self.hidden_layer:
            neuron.update_output(vector)

    def forward_propagate(self):
        """
        Propagate the input vector forward through the neural network.
        """
        for i in range(0, len(self.output_layer)):
            output = 0

            # Loop through each Neuron in the hidden layer
            for neuron in self.hidden_layer:
                output += neuron.weights[i] * neuron.output

            # Update summation for output classifier
            self.output_layer[i] = output

    def back_propagate(self, label_vector):
        """
        Update hidden layer neuron weights based on output error.
        """
        for i in range(0, len(self.output_layer)):
            last_neuron_error = label_vector[i] - self.output_layer[i]

            # Update each neuron's correction value for the weight pointing at the specified output cell
            for neuron in self.hidden_layer:
                neuron.update_correction(last_neuron_error, i)

        # Apply corrections to each neuron
        for neuron in self.hidden_layer:
            neuron.apply_corrections()

    def target_label_as_vector(self, target_label=0):
        target_vector = zeros(len(self.output_layer))
        target_vector[target_label] = 1
        return target_vector

    def identify(self, image_vector):
        self.feed_input(image_vector)
        self.forward_propagate()
        return argmax(self.output_layer)


def find_closest_centroid(vector, centroids):
    centroid = 0
    smallest_distance = norm(vector - centroids[0]) ** 2

    for i in range(1, len(centroids)):
        dist = norm(vector - centroids[i]) ** 2

        if dist < smallest_distance:
            smallest_distance = dist
            centroid = i

    return centroid


def compute_clusters(centroids, data):
    clusters = [list() for i in centroids]

    # Sort vector's into their clusters
    for vector in data:
        c = find_closest_centroid(vector, centroids)
        clusters[c].append(vector)

    return clusters


def compute_sigma(centroid, data):
    #Computes sigma as per http://mccormickml.com/2013/08/15/radial-basis-function-network-rbfn-tutorial/
    dist = 0.0
    for vector in data:
        dist += norm(vector - centroid)

    return dist / len(data)


def compute_beta(sigma):
    return 1.0 / (2 * (sigma ** 2))
