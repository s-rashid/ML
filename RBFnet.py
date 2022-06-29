#!/usr/bin/env python
import array
import functools
import gzip
import json
import operator
import os
import struct
import sys

import numpy as np
from scipy.cluster.vq import whiten
from six.moves import cPickle as pickle
from six.moves.urllib.request import urlretrieve

from q2.lib.RBFNetwork import RBFNetwork

DATA_TYPES = {
    0x08: 'B',  # unsigned byte
    0x09: 'b',  # signed byte
    0x0b: 'h',  # short (2 bytes)
    0x0c: 'i',  # int (4 bytes)
    0x0d: 'f',  # float (4 bytes)
    0x0e: 'd'
}  # double (8 bytes)

LEARNING_RATE = .25
K = 30
CACHE_DIR = '.cache'
URL = 'http://yann.lecun.com/exdb/mnist/'

last_percent_reported = None


# This is a simple example with one layer, it's not sufficient
# Just to kind of get started
def main():
    with(open('config.json', 'r')) as f:
        config = json.load(f)

    train_filename_gz = dataDownload(config['train']['images'], 9912422)
    train_labels_gz = dataDownload(config['train']['labels'], 28881)
    train_pickle = extract(train_filename_gz)
    train_labels_pickle = extract(train_labels_gz)
    train_data = loadPickle(train_pickle)
    train_labels = loadPickle(train_labels_pickle)

    net = loadPickle(config['brain']['filename'])

    if net:
        print ('Using already existing neural network from %s' % config['brain']['filename'])
    else:
        input_training = train_data.reshape(60000, 784).astype(float)

        # Regularize data
        for i in range(0, len(input_training)):
            image_vector = input_training[i]
            input_training[i] = [j / 255.0 for j in image_vector]  # [pixel / (255.0 / 2) - 1 for pixel in image_vector]
        # input_training = whiten(input_training)

        # Initialize the RBF network
        net = RBFNetwork(input_training, k=K, learning_rate=LEARNING_RATE)

        # Train the network
        net.train(input_training, train_labels)

        # Save the network to a pickle file
        pickleFile(config['brain']['filename'], net)


def download_progress_hook(count, block_size, total_size):
    global last_percent_reported

    percent = int(count * block_size * 100 / total_size)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def dataDownload(filename, expected_bytes, force=False):
    dest_filename = os.path.join(CACHE_DIR, filename)
    if force or not os.path.exists(dest_filename):
        filename, _ = urlretrieve(URL + filename, dest_filename, reporthook=download_progress_hook)

    return dest_filename


def extract(filename, force=False):
    with gzip.open(filename, 'rb') as fd:
        pickle_file = filename.replace('.gz', '.pickle')
    return pickleFile(pickle_file, parse_idx(fd), force)


def pickleFile(filename, data, force=False):
    if not os.path.exists(filename) or force:
        with open(filename, 'wb') as pf:
            pickle.dump(data, pf, pickle.HIGHEST_PROTOCOL)
    return filename


def parse_idx(fd):
    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))
    return np.array(data).reshape(dimension_sizes)


def loadPickle(f):
    try:
        with open(f, 'rb') as tp:
            return pickle.load(tp)
    except IOError:
        return None


if __name__ == '__main__':
    main()
