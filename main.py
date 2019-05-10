from nn import NeuralNetwork, Layer
from logic_gates.solver import LogicExpressionSolver
from logic_gates.dataset import logic_dataset
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    logic_nn = NeuralNetwork(seed=2628856917)
    logic_nn.add_layer(Layer(3, 15, 'tanh'))
    logic_nn.add_layer(Layer(15, 1, 'tanh'))

    X = logic_dataset["X"]
    y = logic_dataset["y"]

    errors = logic_nn.train(X, y, 0.01, 10000)
    y_pred = np.round(logic_nn.predict(X)).astype(dtype='int8')
    print('Accuracy: %.2f%%' % (logic_nn.accuracy(y_pred, y) * 100))

    plt.plot(errors)
    plt.title('Changes in MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.show()

    solver = LogicExpressionSolver(solver=logic_nn)
    exp = "(x xor y) pirse (x sheffer z)"

    check_dataset = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    def f(row):
        x, y, z = row
        return int(not ((not x and y or x and not y) or not (x and z)))

    print("input\t\ttrue\tpred")
    for row in check_dataset:
        print("%s\t%s\t\t%s" % (row, f(row), solver.solve(expression=exp, inputs=row, verbose=False)))

    def f(x):
        return x ** 3 + x ** 2 - x - 1
        # return np.cos(x)
        # return x * np.sin(x * 0.5) * np.cos(x / 0.5)
        # return np.sqrt(1 + 4 * x + 12 * x ** 2)
        # return np.exp(x)/np.cos(x)

    np.seterr(all='raise', over='raise')
    x_start, x_end = -(-4*np.pi*0 + np.pi)/2 + 1E-3, (-4*np.pi*0 + np.pi)/2 - 1E-3

    nn = NeuralNetwork(1111)

    nn.add_layer(Layer(1, 25, "tanh"))
    nn.add_layer(Layer(25, 16, "sigmoid"))
    nn.add_layer(Layer(16, 1, None))

    data_size = 1000
    x_values = np.array([[x] for x in np.arange(x_start, x_end, (x_end - x_start) / data_size)])
    y_values = np.array([f(x) for x in x_values])

    max_x = np.abs(x_values).max()
    max_y = np.abs(y_values).max()
    x_values /= max_x
    y_values /= max_y

    mse = nn.train(x_values, y_values, learning_rate=0.01, max_epochs=1000) * max_y

    y_pred = nn.predict([x_values])[0]
    y_pred *= max_y
    x_values = x_values * max_x
    y_values = y_values * max_y

    plt.plot(x_values, y_values)
    plt.plot(x_values, y_pred)
    plt.savefig('approximation.png')
    plt.show()

    print(mse[-1])
    plt.plot(mse)
    plt.savefig('mse.png')
    plt.show()
