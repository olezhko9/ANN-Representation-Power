import numpy as np
import matplotlib.pyplot as plt


class LogicNeuralNetwork:
    def __init__(self, n_input_neurons, n_hidden_neurons, n_output_neurons, learning_rate=0.001, activation=None):
        # number of nodes in layers
        self.ni = n_input_neurons + 1  # +1 for bias
        self.nh = n_hidden_neurons
        self.no = n_output_neurons
        self.lr = learning_rate
        self.activation = activation

        # initialize node-activations
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no

        # initialize node weights to random vals
        self.wi = np.random.uniform(-0.2, 0.2, (self.ni, self.nh))
        self.wo = np.random.uniform(-2.0, 2.0, (self.nh, self.no))

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
            self.ao[k] = self.apply_activation(output_weight)

        return self.ao

    def apply_activation(self, r):
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

    def back_propagation(self, targets):
        # dE/dw[j][k] = (t[k] - ao[k]) * s'( SUM( w[j][k]*ah[j] ) ) * ah[j]
        # calc output deltas
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = error * self.apply_activation_derivative(self.ao[k])

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
            inputs = p[0]
            predicted_proba = self.feed_forward(inputs)
            predicted_labels.append(int(round(predicted_proba[0])))
        return predicted_labels

    def train(self, patterns, max_iterations=1000, plot_error=False):
        errors = []
        for i in range(max_iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feed_forward(inputs)
                error += self.back_propagation(targets)

            if i % 10 == 0:
                errors.append(error)

        if plot_error:
            plt.plot(errors)
            plt.title('Changes in MSE')
            plt.xlabel('Epoch (every 10th)')
            plt.ylabel('MSE')
            plt.show()

def main():
    train_data = [
        # AND
        [[0, 0, 0], [0]],
        [[0, 0, 1], [0]],
        [[0, 1, 0], [0]],
        [[0, 1, 1], [1]],
        # OR
        [[1, 0, 0], [0]],
        [[1, 0, 1], [1]],
        [[1, 1, 0], [1]],
        [[1, 1, 1], [1]],
        # NOT
        [[-1, 0, 0], [1]],
        [[-1, 1, 1], [0]],
        # NOR
        [[-2, 0, 0], [1]],
        [[-2, 0, 1], [0]],
        [[-2, 1, 0], [0]],
        [[-2, 1, 1], [0]],
        # NAND
        [[-3, 0, 0], [1]],
        [[-3, 0, 1], [1]],
        [[-3, 1, 0], [1]],
        [[-3, 1, 1], [0]],
        # XOR
        # [[4, 0, 0], [0]],
        # [[4, 0, 1], [1]],
        # [[4, 1, 0], [1]],
        # [[4, 1, 1], [0]],
        # # NXOR
        # [[-4, 0, 0], [1]],
        # [[-4, 0, 1], [0]],
        # [[-4, 1, 0], [0]],
        # [[-4, 1, 1], [1]],
    ]
    
    logic_nn = LogicNeuralNetwork(3, 3, 1, learning_rate=0.1, activation="tanh")
    logic_nn.train(train_data, plot_error=True)
    pred_labels = logic_nn.test(train_data)

    for i in range(len(pred_labels)):
        print('Inputs:', train_data[i][0], '-->', 'Predicted', [pred_labels[i]], '\tTarget', train_data[i][1])
        
    logic_nn.weights()

if __name__ == "__main__":
    main()

