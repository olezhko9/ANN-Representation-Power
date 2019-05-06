import re

logic_dict = {
    "and": 1,
    "or": 2,
    "xor": 3,
    "imply": 4,
    "not": -1,
    "nand": -2,
    "sheffer": -2,
    "nor": -3,
    "pirse": -3,
    "nxor": -4,
    "eq": -4,
}


class LogicExpressionSolver:
    def __init__(self, solver):
        self.solver = solver

    def solve(self, expression, inputs, verbose=False):
        self.verbose = verbose
        x, y, z = inputs
        expression = expression.lower()
        expression = re.sub(r'\bx\b', str(x), expression)
        expression = re.sub(r'\by\b', str(y), expression)
        expression = re.sub(r'\bz\b', str(z), expression)
        if self.verbose:
            print(expression)
        return self._eval(expression)

    def _eval(self, expression):
        if len(expression) == 1:
            return expression

        not_pattetn = re.compile('not \d')
        while not_pattetn.search(expression):
            val = int(not_pattetn.search(expression).group(0)[-1])
            pred = str(self.solver.test([[logic_dict["not"], val, val]])[0])
            expression = not_pattetn.sub(pred, expression, 1)
            if self.verbose:
                print(expression)

        par_pattern = re.compile('\((\d \w+ \d)\)')
        while par_pattern.search(expression):
            match = par_pattern.search(expression).group(1).split(' ')
            pred = str(self.solver.test([[logic_dict[match[1]], int(match[0]), int(match[2])]])[0])
            expression = par_pattern.sub(pred, expression, 1)
            if self.verbose:
                print(expression)
        
        and_pattern = re.compile('\d and \d')
        while and_pattern.search(expression):
            match = and_pattern.search(expression).group(0).split(' ')
            pred = str(self.solver.test([[logic_dict[match[1]], int(match[0]), int(match[2])]])[0])
            expression = and_pattern.sub(pred, expression, 1)
            if self.verbose:
                print(expression)

        any_pattern = re.compile('(?!\()\d \w+ \d(?!\))')
        while any_pattern.search(expression):
            match = any_pattern.search(expression).group(0).split(' ')
            pred = str(self.solver.test([[logic_dict[match[1]], int(match[0]), int(match[2])]])[0])
            expression = any_pattern.sub(pred, expression, 1)
            if self.verbose:
                print(expression)

        return self._eval(expression)
            

