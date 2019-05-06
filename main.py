from logic_gates.nn import LogicNeuralNetwork
from logic_gates.solver import LogicExpressionSolver
from logic_gates_dataset import logic_dataset


if __name__ == '__main__':

    logic_nn = LogicNeuralNetwork(3, 12, 1, learning_rate=0.04, max_iterations=2000, activation="tanh")

    logic_nn.train(logic_dataset, plot_error=True)
    pred_labels = logic_nn.test([test[:-1] for test in logic_dataset])

    # for i in range(len(pred_labels)):
    #     print('Inputs:', logic_dataset[i][:-1], '-->', 'Predicted', [pred_labels[i]], '\tTarget', logic_dataset[i][-1])

    print("Model accuracy:", logic_nn.accuracy([logic_dataset[i][-1] for i in range(len(logic_dataset))], pred_labels), "%")

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
