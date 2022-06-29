#!/usr/bin/env python
"""
The purpose of this file is to perform the elbow method of finding the correct K value
for use within the RBF Network.

Note: This code will take a very long time to execute
"""
import json
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm
from scipy.cluster.vq import kmeans, whiten
from a2q2 import dataDownload, extract, loadPickle
from q2.lib.RBFNetwork import compute_clusters


def main():
    with(open('config.json', 'r')) as f:
        config = json.load(f)

    train_filename_gz = dataDownload(config['train']['images'], 9912422)
    train_pickle = extract(train_filename_gz)
    train_data = loadPickle(train_pickle)

    input_training = train_data.reshape(60000, 784)[:500]
    distance_list = []
    whitened = whiten(input_training)
    k_values = range(1, 401)
    for i in k_values:
        print ("Performing K-Means clustering with K = {}".format(i))
        centroids, distortion = kmeans(whitened, i)
        clusters = compute_clusters(centroids,input_training)

        total_dist = 0
        # Compute distance from each cluster's points to their centroid
        for j in range(0, len(clusters)):
            centroid_dist = 0

            for vector in clusters[j]:
                centroid_dist += norm(vector - centroids[j]) ** 2

            total_dist += centroid_dist

        print ("Objective centroid distance of {} observed".format(total_dist))
        distance_list.append(total_dist)

    plot = {"Distances": distance_list}
    fig, ax = plt.subplots()
    errors = pd.DataFrame(plot)
    errors.plot(ax=ax)
    plt.show()


if __name__ == '__main__':
    main()
