import numpy as np
import math
from scipy.optimize import minimize
import cPickle as pickle
from random import shuffle
import load_data
import pdb
import utils
import sys
from optparse import OptionParser

np.set_printoptions(precision=2, suppress=True)


def one_hot(i, size):
    o = [0.0] * size
    o[int(i)] = 1.0
    return o


def un_hot(l):
    return l  # .index(1.0)


def dump(nn, location):
    f = file(location, 'wb')
    pickle.dump(nn, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()


def load(location):
    f = file(location, 'rb')
    nn = pickle.load(f)
    f.close()
    return nn


def sigmoid(a):
    d = - a
    d = 1.0 + np.exp(d)
    d = 1.0 / d
    return d.reshape(np.shape(a))


def sigmoid_prime(z):
    return z * (1.0 - z)


def add_bias_input(vec, bias=1.0):
    vec = np.append(vec, [bias])
    return vec.reshape(np.size(vec), 1)


def remove_bias(delta):
    return delta[:-1]


def safe_log(l, p):
    if abs(l - p) == 1.0:
        return -100.0
    elif l == 1:
        return math.log(p)
    else:
        return math.log(1.0 - p)


class HiddenLayer():
    def __init__(self, n_inputs, n_outputs, w=1):
        self.n_inputs = n_inputs + 1  # +1 for bias
        self.n_outputs = n_outputs
        if w == 1:
            ep = math.sqrt(6.0) / math.sqrt(self.n_inputs + self.n_outputs)
        elif w == 2:
            ep = 1.0 / math.sqrt(self.n_inputs)
        elif w == 3:
            ep = 0.01
        else:
            ep = math.sqrt(6.0) / math.sqrt(self.n_inputs + self.n_outputs)
        self.W = np.random.uniform(-ep, ep, (self.n_outputs, self.n_inputs))
        # self.W = np.zeros((self.n_outputs, self.n_inputs))

    def func_dot(self, x, w):
        assert np.shape(x) == np.shape(w)
        return np.dot(x.T, w)[0, 0]

    def func_mult(self, m1, m2):
        assert np.shape(m1) == np.shape(m2)
        return m1 * m2

    def func_z(self, x, w):
        a = self.func_dot(x, w)
        z = sigmoid(a)
        return z

    def func_gprime(self, x, w):
        z = self.func_z(x, w)
        gz = sigmoid_prime(z)
        return gz

    def weight_update(self, z_current, delta_from_next):
        delta_from_next = remove_bias(delta_from_next)
        z_current = add_bias_input(z_current)
        x1 = np.reshape(delta_from_next, (self.n_outputs, 1))
        x2 = np.reshape(z_current, (1, self.n_inputs))
        return np.multiply(x1, x2)

    def get_a(self, x_input):
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, 1)
        a = np.dot(self.W, x_input)
        assert np.shape(a) == (self.n_outputs, 1)
        return a

    def get_z(self, x_input):
        assert np.shape(x_input) == (self.n_inputs - 1, 1)
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, 1)
        a = np.dot(self.W, x_input)
        z = sigmoid(a)
        assert np.shape(z) == (self.n_outputs, 1)
        return z

    def get_zprime(self, x_input):
        x_input = add_bias_input(x_input)
        assert np.shape(x_input) == (self.n_inputs, 1)
        a = np.dot(self.W, x_input)
        z = sigmoid(a)
        gz = sigmoid_prime(z)
        assert np.shape(gz) == (self.n_outputs, 1)
        return gz

    def get_delta(self, zprime_current, delta_from_next):
        delta_from_next = remove_bias(delta_from_next)
        zprime_current = add_bias_input(zprime_current, bias=1.0)
        assert np.shape(delta_from_next) == (self.n_outputs, 1)
        assert np.shape(zprime_current) == (self.n_inputs, 1)
        p_test = np.dot(self.W.T, delta_from_next)
        delta_current = p_test * zprime_current
        assert np.shape(delta_current) == (self.n_inputs, 1)
        return delta_current

    def update(self, learning_rate, w_update):
        self.W += learning_rate * w_update


class OutputLayer(HiddenLayer):
    def __init__(self, n_inputs, n_outputs, w=1):
        HiddenLayer.__init__(self, n_inputs, n_outputs, w)

    def get_delta_at_final(self, prediction, target_at_output):
        assert np.shape(prediction) == (self.n_outputs, 1)
        assert np.shape(target_at_output) == (self.n_outputs, 1)
        return prediction - target_at_output

    def get_delta(self, zprime_current, delta_from_next):
        zprime_current = add_bias_input(zprime_current, bias=1.0)
        assert np.shape(delta_from_next) == (self.n_outputs, 1)
        assert np.shape(zprime_current) == (self.n_inputs, 1)
        p_test = np.dot(self.W.T, delta_from_next)
        delta_current = p_test * zprime_current
        assert np.shape(delta_current) == (self.n_inputs, 1)
        return delta_current

    def weight_update(self, z_current, delta_from_next):
        z_current = add_bias_input(z_current)
        x1 = np.reshape(delta_from_next, (self.n_outputs, 1))
        x2 = np.reshape(z_current, (1, self.n_inputs))
        return x1 * x2


class Network():
    def __init__(self, lmbda=None, topology=None, data=None, w=1):
        self.layers = []
        self.lmbda = lmbda
        # self.data = data
        if topology is not None:
            self.topology = topology
            self.layers = self.make_layers(topology, w)
        if data is not None:
            self.N = float(len(data))

    def size_bytes(self):
        tot_bytes = 0
        for l in self.layers:
            tot_bytes += l.W.nbytes
        return tot_bytes

    def make_layers(self, topology, w=1):
        layers = []
        for idx, (t_inp, t_out) in enumerate(zip(topology, topology[1:])):
            if idx == len(topology[1:]) - 1:
                # self.layers.append(OutputLayer(t_inp, t_out))
                layers.append(OutputLayer(t_inp, t_out, w))
            else:
                # self.layers.append(HiddenLayer(t_inp, t_out))
                layers.append(HiddenLayer(t_inp, t_out, w))
        return layers

    def predict(self, data):
        predictions = []
        for d, l in data[:]:
            # print d, l
            z = d
            for idx, layer in enumerate(self.layers):
                if idx == len(self.layers) - 1:
                    # this is a output layer
                    prediction = layer.get_z(z)
                    predictions.append(prediction)
                else:
                    z = layer.get_z(z)
        return predictions

    def get_representation(self, x_input, layer=0):
        """
        :returns output of the layer (after sigmoid)
        :param x_input:
        :param layer:
        :return:
        """
        z = self.layers[layer].get_z(x_input)
        return z

    def get_network_weights(self):
        linear_weights = np.asarray([])
        for l in self.layers:
            length = np.shape(l.W)[0] * np.shape(l.W)[1]
            linear_weights = np.append(linear_weights, l.W.reshape(length, 1))
        return linear_weights

    def set_network_weights(self, weights):
        st = 0
        for l in self.layers:
            end = st + (np.shape(l.W)[0] * np.shape(l.W)[1])
            segment = weights[st:end]
            new_w = segment.reshape(np.shape(l.W))
            l.W = new_w
            st = end

    def convert_weights_to_layers(self, weights):
        layers = self.make_layers(self.topology)
        st = 0
        for l in layers:
            end = st + (np.shape(l.W)[0] * np.shape(l.W)[1])
            segment = weights[st:end]
            new_w = segment.reshape(np.shape(l.W))
            l.W = new_w
            st = end
        return layers

    def get_cost(self, weights, data, display=True):
        # print 'getting cost...'
        N = float(len(data))
        reg = (self.lmbda / (2.0 * N)) * np.sum(weights ** 2)
        # reg = (self.lmbda / self.N) * np.sum(np.abs(weights))
        # self.set_network_weights(weights)
        layers = self.convert_weights_to_layers(weights)
        cost = 0.0
        for d, l in data[:]:
            z = d
            for idx, layer in enumerate(layers):
                if idx == len(layers) - 1:
                    # this is a output layer
                    prediction = layer.get_z(z)
                    prediction[prediction >= 1.0] = 1.0 - np.finfo(float).eps  # to avoid nan showing up
                    prediction[prediction <= 0.0] = 0.0 + np.finfo(float).eps
                    l1p = -l * np.log(prediction)
                    l2p = -(1.0 - l) * np.log((1.0 - prediction))
                    lcost = np.sum(l1p + l2p)
                    cost += lcost * (1.0 / float(N))
                else:
                    z = layer.get_z(z)
        if display:
            sys.stderr.write('cost:' + str(float(cost + reg)) + '\n')
        return cost + reg

    def get_gradient(self, weights, data, display=False):
        # print 'getting grad...'
        N = float(len(data))
        reg = (self.lmbda / N) * weights
        # self.set_network_weights(weights)
        layers = self.convert_weights_to_layers(weights)
        accumulate_deltas = [np.zeros(np.shape(layer.W)) for layer in layers]
        for d, l in data[:]:
            z_list = [None] * (len(layers) + 1)
            zp_list = [None] * (len(layers) + 1)
            delta_list = [None] * (len(layers) + 1)
            z_list[0] = d
            zp_list[0] = d

            for idx, layer in enumerate(layers):
                z_next = layer.get_z(z_list[idx])
                z_next_prime = layer.get_zprime(zp_list[idx])
                z_list[idx + 1] = z_next
                zp_list[idx + 1] = z_next_prime

            for idx in reversed(range(len(layers))):
                layer = layers[idx]
                if isinstance(layer, OutputLayer):
                    delta = layer.get_delta_at_final(z_list[idx + 1], np.asarray(l))
                    delta_list[idx + 1] = delta
                    delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                    delta_list[idx] = delta
                else:
                    delta = layer.get_delta(zp_list[idx], delta_list[idx + 1])
                    delta_list[idx] = delta

            for idx, layer in enumerate(layers):
                theta = accumulate_deltas[idx]
                theta += layer.weight_update(z_list[idx], delta_list[idx + 1]) * (1.0 / float(N))
                accumulate_deltas[idx] = theta

        linear_deltas = np.asarray([])
        for a in accumulate_deltas:
            length = np.shape(a)[0] * np.shape(a)[1]
            linear_deltas = np.append(linear_deltas, a.reshape(length, 1))
        linear_deltas += reg
        return linear_deltas

    def train(self, data, init_weights=None, tol=0.1):
        if init_weights is None:
            init_weights = self.get_network_weights()
        t1 = minimize(self.get_cost, init_weights, method='L-BFGS-B', jac=self.get_gradient,
                      args=(data, ),
                      tol=tol)

        self.set_network_weights(t1.x)
        return t1.x

    def train_earlystop(self, data, init_weights=None, tol=0.0001, maxfun=5):
        if init_weights is None:
            sys.stderr.write('using network weights...\n')
            init_weights = self.get_network_weights()
        t1 = minimize(self.get_cost, init_weights, method='L-BFGS-B', jac=self.get_gradient,
                      args=(data, ),
                      tol=tol, options={'maxfun': maxfun})

        return t1.x, t1.fun

    def train_sgd(self, data, momentum=0.0, init_weights=None, epochs=100):
        weights = self.get_network_weights()
        ids = range(len(data))
        lr = 0.01
        for _ in range(epochs):
            prev_grad = np.zeros_like(self.get_gradient(weights, data[0:1]))
            sys.stderr.write('epoch: ' + str(_) + '\n')
            shuffle(ids)
            for d_id in ids:
                grad = self.get_gradient(weights, data[d_id:d_id + 1])
                g = grad + (momentum * prev_grad)
                weights -= lr * g
                prev_grad = grad
        self.set_network_weights(weights)
        return weights

    def train_adagrad(self, data, init_weights=None, maxfun=5):
        if init_weights is None:
            sys.stderr.write('using network weights...\n')
            weights = self.get_network_weights()
        else:
            weights = init_weights
        ids = range(len(data))
        eta0 = 1.0
        sum_square_grad = np.zeros(np.shape(weights))
        I = 1.0
        for _ in range(maxfun):
            shuffle(ids)
            for d_id in ids:
                grad = self.get_gradient(weights, data[d_id:d_id + 1])
                sum_square_grad += (grad ** 2)
                eta_t = eta0 / np.sqrt(I + sum_square_grad)
                weights -= np.multiply(eta_t, grad)
        self.set_network_weights(weights)
        return weights


def make_data(data):
    labels = data[:, 0]
    n_outputs = np.shape(np.unique(labels))[0]
    data = data[:, 1:]
    row_sums = data.sum(axis=1)
    # data = data / row_sums[:, np.newaxis]
    n_inputs = len(data[0])
    one_hot_labels = [one_hot(l, n_outputs) for l in labels]
    data = [(list(d), l) for d, l in zip(data, one_hot_labels)]
    data = [(np.reshape(x, (len(x), 1)), np.reshape(y, (len(y), 1)))
            for x, y in data]
    return data, labels, n_inputs, n_outputs


if __name__ == '__main__':
    # script here
    opt = OptionParser()
    # insert options here
    opt.add_option('-d', dest='data_file', default='')
    opt.add_option('-r', dest='split', default='')
    opt.add_option('-o', dest='optimize', default='')
    opt.add_option('-s', dest='nn_size', default='')
    opt.add_option('-w', dest='weight_init', default='')
    (options, _) = opt.parse_args()
    if options.data_file == '' or options.split == '' or options.optimize == '' or options.nn_size == '' or options.weight_init == '':
        sys.stderr.write(
            'Usage: python neural_networks.py -d [c/i/m1/m2/m3] -r [train test ratio] -o [sgd/sgdm/adagrad/lbfgs] -s [s/m/l] -w [1/2/3]\n')
        exit(-1)
    else:
        pass
    if options.data_file == 'c':
        test_data, training_data = load_data.load_congress_data(float(options.split))
    elif options.data_file == 'i':
        test_data, training_data = load_data.load_iris(float(options.split))
    elif options.data_file == 'm1':
        test_data, training_data = load_data.load_monks(1)
    elif options.data_file == 'm2':
        test_data, training_data = load_data.load_monks(2)
    elif options.data_file == 'm3':
        test_data, training_data = load_data.load_monks(3)
    else:
        sys.stderr.write(
            'Usage: neural_networks.py -d [c/i/m1/m2/m3] -r [train test ratio] -o [sgd/sgdm/adagrad/lbfgs] -s [s/m/l] -w [1/2/3]\n')
        exit(-1)

    w_opt = int(options.weight_init)

    test_data, test_labels, test_inputs, test_outputs = make_data(test_data)
    data, labels, n_inputs, n_outputs = make_data(training_data)
    if options.nn_size == 's':
        nn = Network(0.001, [n_inputs, int(n_inputs / 2), n_outputs], data, w_opt)
    elif options.nn_size == 'm':
        nn = Network(0.001, [n_inputs, int(n_inputs) + 4, n_outputs], data, w_opt)
    elif options.nn_size == 'l':
        nn = Network(0.001, [n_inputs, int(n_inputs) + 4, int(n_inputs) + 4, n_outputs], data, w_opt)
    else:
        sys.stderr.write(
            'Usage: neural_networks.py -d [c/i/m1/m2/m3] -r [train test ratio] -o [sgd/sgdm/adagrad/lbfgs] -s [s/m/l] -w [1/2/3]\n')
        exit(-1)

    init_weights = nn.get_network_weights()

    if options.optimize == 'sgd':
        final_weights = nn.train_sgd(data, momentum=0.0, epochs=100)
    elif options.optimize == 'sgdm':
        final_weights = nn.train_sgd(data, momentum=0.1, epochs=20)
    elif options.optimize == 'adagrad':
        final_weights = nn.train_adagrad(data)
    else:
        final_weights = nn.train(data, tol=0.0001)

    nn.set_network_weights(final_weights)
    ps = nn.predict(test_data)
    ls = [l for d, l in test_data]
    for l, p in zip(ls, ps):
        ns = np.array([1.0 if i == max(p) else 0.0 for i in p])
        ns = np.reshape(ns, (len(ns), 1))
        print ','.join([str(i) for i in l.tolist()]), ','.join([str(i) for i in ns.tolist()])
