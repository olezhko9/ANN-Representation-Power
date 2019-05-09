from logic_gates.nn import NeuralNetwork, Layer
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
    plt.xlabel('Epoch (every 10th)')
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
