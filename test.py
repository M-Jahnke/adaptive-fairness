import nlopt
from numpy import *

class Test:

    def __init__(self, a):
        self.a = a

    def myfunc(self, x, grad):
        if grad.size > 0:
            grad[0] = 0.0
            grad[1] = 0.5 / sqrt(x[1])
        return sqrt(x[1])
    def myconstraint(self, x, grad, a, b):
        if grad.size > 0:
            grad[0] = 3 * a * (a*x[0] + b)**2
            grad[1] = -1.0
        return (a*x[0] + b)**3 - x[1]

    def train(self):
        opt = nlopt.opt(nlopt.LD_MMA, 2)
        opt.set_lower_bounds([-float('inf'), 0])
        opt.set_min_objective(self.myfunc)
        opt.add_inequality_constraint(lambda x,grad: self.myconstraint(x,grad,2,0), 1e-8)
        opt.add_inequality_constraint(lambda x,grad: self.myconstraint(x,grad,-1,1), 1e-8)
        opt.set_xtol_rel(1e-4)
        x = opt.optimize([1.234, 5.678])
        minf = opt.last_optimum_value()
        print("optimum at ", x[0], x[1])
        print("minimum value = ", minf)
        print("result code = ", opt.last_optimize_result())

def main():
    test = Test(a='a')
    test.train()

if __name__ == "__main__":
    main()