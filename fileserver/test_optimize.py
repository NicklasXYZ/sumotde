from SALib.sample import sobol_sequence
from SALib.util import scale_samples
import numpy as np
from scipy.optimize import (
    OptimizeResult,
    basinhopping,
    differential_evolution,
)

def main():
    func_hist = []
    def func(x):
        fval = np.sum(np.cos(14.5 * x - 0.3) + (x + 0.2) * x)
        func_hist.append(fval)
        return fval

    minimizer_kwargs = {
        # "method": "COBYLA",
        "method": "SLSQP",
        "tol": 0.10,
        # "constraints":  {
        #     "type": "eq",
        #     "fun": self.gencon_penalty
        # },
    }
    # minimizer_kwargs = {
    #     "method": "L-BFGS-B",
    #     "bounds": self.bounds,
    #     "tol": 0.001,
    # }
    v = sobol_sequence.sample(10, 10)
    # scale_samples(v, problem={"bounds": self.bounds})
    new_sample = v[-1]
    result = basinhopping(
        func=func,
        x0=new_sample,
        niter=200,
        # niter=5,
        # T=0.10,
        # stepsize=1.0,
        minimizer_kwargs=minimizer_kwargs,
    )
    return result, func_hist

if "__main__" == __name__:
    res, hist = main()
    print("Result object            : ")
    print(res)
    print("Objective function values: ")
    for item in hist:
        print("OF Value: ", item)
