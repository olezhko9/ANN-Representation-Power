import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(2019)


class LogicNeuralNetwork:
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, learning_rate=0.001, max_iterations=1000,
                 activation=None):
        # number of nodes in layers
        self.ni = n_input_neurons + 1  # +1 for bias
        self.nh = n_hidden_neurons
        self.no = n_output_neurons
        self.lr = learning_rate
        self.max_iterations = max_iterations
        self.activation = activation

        # initialize node-activations
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # initialize node weights to random vals
        self.wi = np.random.uniform(-0.2, 0.2, (self.ni, self.nh)).astype(dtype='float64')
        self.wo = np.random.uniform(-1.0, 1.0, (self.nh, self.no)).astype(dtype='float64')

    def feed_forward(self, inputs):
        if len(inputs) != self.ni - 1:
            print('incorrect number of inputs')

        for i in range(self.ni - 1):
            self.ai[i] = inputs[i]

        for j in range(self.nh):
            hidden_weight = 0.0
            for i in range(self.ni):
                hidden_weight += (self.ai[i] * self.wi[i][j])
            self.ah[j] = self.apply_activation(hidden_weight)

        for k in range(self.no):
            output_weight = 0.0
            for j in range(self.nh):
                output_weight += (self.ah[j] * self.wo[j][k])
            self.ao[k] = output_weight
        return self.ao

    def apply_activation(self, r):
        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return np.tanh(r)

        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        if self.activation == 'relu':
            return max(0, r)

        return r

    def apply_activation_derivative(self, r):

        if self.activation is None:
            return r

        if self.activation == 'tanh':
            return 1 - r ** 2

        if self.activation == 'sigmoid':
            return r * (1 - r)

        if self.activation == 'relu':
            return r

        return r

    def back_propagation(self, targets):
        # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        # calc output deltas
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] += self.lr * change

        # calc hidden deltas
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = error * self.apply_activation_derivative(self.ah[j])

        # update input weights

        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] += self.lr * change

        # calc combined error
        # 1/2 for differential convenience & **2 for modulus
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.ao[k]) ** 2
        return error

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])
        print('')

    def test(self, patterns):
        predicted_labels = []
        for p in patterns:
            inputs = p
            predicted_proba = self.feed_forward(inputs)
            predicted_labels.append(predicted_proba[0])
        return predicted_labels

    def train(self, X_train, y_train, plot_error=False):
        print_freq = 20
        errors = []
        for i in range(self.max_iterations):
            error = 0.0
            for j in range(len(X_train)):
                inputs = [X_train[j]]
                targets = [y_train[j]]
                self.feed_forward(inputs)
                error += self.back_propagation(targets)

            if (i+1) % print_freq == 0:
                print("It - %d: error = %.5f" % (i+1, error))
                errors.append(error)

        if plot_error:
            plt.plot(errors)
            plt.title('Changes in MSE')
            plt.xlabel('Epoch (every %d-th)' % print_freq)
            plt.ylabel('MSE')
            plt.show()
        return True

    @staticmethod
    def accuracy(test_data, pred_data):
        if len(test_data) != len(pred_data):
            print("array sizes do not match")

        return (np.array(test_data) == np.array(pred_data)).mean() * 100


def f(x):
    return np.sin(x)
    # return np.sqrt(1 + 4*x + 12*x**2)
    # return x * np.sin(x * 0.5) * np.cos(x / 0.5)
    # return (1+x)**x


if __name__ == "__main__":

    np.seterr(all='raise', over='raise')
    x_start, x_end = -5, 5

    nn = LogicNeuralNetwork(1, 10, 1, learning_rate=0.01, activation="tanh", max_iterations=1000)

    x_values = np.linspace(x_start, x_end, 1000)
    y_values = np.array([f(x) for x in x_values])

    nn.train(x_values, y_values, plot_error=True)

    y_pred = [nn.test([[x]]) for x in x_values]

    plt.plot(x_values, y_values)
    plt.plot(x_values, y_pred)
    plt.show()


