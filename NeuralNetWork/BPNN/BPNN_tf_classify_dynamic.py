
"""
dynamiic BPNN
bigV
"""

import tensorflow as tf
import numpy as np


def make_layer(inputs, in_size, out_size, activate=None):
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    basis = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    result = tf.matmul(inputs, weights) + basis
    if activate is None:
        return result
    else:
        return activate(result)


class BPNeuralNetwork:
    def __init__(self, session):
        self.session = session
        self.loss = None
        self.optimizer = None
        self.input_n = 0
        self.hidden_n = 0
        self.hidden_size = []
        self.output_n = 0
        self.input_layer = None
        self.hidden_layers = []
        self.output_layer = None
        self.label_layer = None

    def setup(self, layers):
        # set size args
        if len(layers) < 3:
            return
        self.input_n = layers[0]
        self.hidden_n = len(layers) - 2  # count of hidden layers
        self.hidden_size = layers[1:-1]  # count of cells in each hidden layer
        self.output_n = layers[-1]

        # build network
        self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])
        self.label_layer = tf.placeholder(tf.float32, [None, self.output_n])
        # build hidden layers
        in_size = self.input_n
        out_size = self.hidden_size[0]
        self.hidden_layers.append(make_layer(self.input_layer, in_size, out_size, activate=tf.nn.relu))
        for i in range(self.hidden_n-1):
            in_size = out_size
            out_size = self.hidden_size[i+1]
            inputs = self.hidden_layers[-1]
            self.hidden_layers.append(make_layer(inputs, in_size, out_size, activate=tf.nn.relu))
        # build output layer
        self.output_layer = make_layer(self.hidden_layers[-1], self.hidden_size[-1], self.output_n)

    def train(self, cases, labels, limit=100, learn_rate=0.05):
        self.loss = tf.reduce_mean(tf.reduce_sum(tf.square((self.label_layer - self.output_layer)), reduction_indices=[1]))
        self.optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss)

        self.session.run(tf.initialize_all_variables())
        for i in range(limit):
            self.session.run(self.optimizer, feed_dict={self.input_layer: cases, self.label_layer: labels})

    def predict(self, case):
        return self.session.run(self.output_layer, feed_dict={self.input_layer: case})


def runTest():
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_data = np.array([[0, 1, 1, 0]]).transpose()
    test_data = np.array([[0, 1]])
    with tf.Session() as session:
        model = BPNeuralNetwork(session)
        model.setup([2, 10, 5, 1])
        model.train(x_data, y_data)
        print(model.predict(test_data))




if __name__ == '__main__':
    runTest()
