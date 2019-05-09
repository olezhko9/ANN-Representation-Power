import numpy as np
import matplotlib.pyplot as plt
from logic_gates.logic_gates_dataset import logic_dataset

np.random.seed(2628856917)
print(np.random.get_state()[1][0])


class Layer:

    def __init__(self, n_input, n_neurons, activation=None, weights=None, bias=None):

        self.weights = weights if weights is not None else np.random.rand(n_input, n_neurons)
        self.activation = activation
        self.bias = bias if bias is not None else np.random.rand(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None

    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self._apply_activation(r)
        return self.last_activation

    def _apply_activation(self, r):
        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return np.tanh(r)

        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        return r

    def apply_activation_derivative(self, r):
        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)

        return r


class NeuralNetwork:
    def __init__(self):
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def predict(self, X):
        return self.feed_forward(X)

    def backpropagation(self, X, y, learning_rate):
        output = self.feed_forward(X)

        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.apply_activation_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.apply_activation_derivative(layer.last_activation)

        for i in range(len(self._layers)):
            layer = self._layers[i]
            input_to_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_to_use.T * learning_rate

    def train(self, X, y, learning_rate, max_epochs):
        mses = []

        for i in range(max_epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)
            if i % 50 == 0:
                mse = np.mean(np.square(y - self.feed_forward(X)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        return mses

    @staticmethod
    def accuracy(y_pred, y_true):
        return (y_pred == y_true).mean()


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_layer(Layer(3, 15, 'tanh'))
    # nn.add_layer(Layer(12, 6, 'tanh'))
    nn.add_layer(Layer(15, 1, 'tanh'))

    # X = np.array([
    #     [0, 0, 0],
    #     [0, 0, 1],
    #     [0, 1, 0],
    #     [0, 1, 1],
    #     [1, 0, 0],
    #     [1, 0, 1],
    #     [1, 1, 0],
    #     [1, 1, 1],
    # ])

    # y = np.array([[0], [0], [0], [0], [0], [0], [0], [1]])

    X = logic_dataset["X"]
    y = logic_dataset["y"]

    errors = nn.train(X, y, 0.01, 10000)
    y_pred = np.round(nn.predict(X)).astype(dtype='int8')
    print(y_pred)
    print('Accuracy: %.2f%%' % (nn.accuracy(y_pred, y) * 100))


    plt.plot(errors)
    plt.title('Changes in MSE')
    plt.xlabel('Epoch (every 10th)')
    plt.ylabel('MSE')
    plt.show()
