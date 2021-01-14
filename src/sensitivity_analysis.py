from train import main
import numpy as np
import itertools as it


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == "__main__":
    default_params = {"max_epochs": 1}
    NOF_STEPS = 10
    varying_params2vals = {
        ("lambda_od", "lambda_entropy"): [
            np.linspace(0.01, 0.09, NOF_STEPS),
            np.linspace(0.1, 1, NOF_STEPS),
        ],
        ("gamma_od", "gamma_entropy"): [
            np.linspace(0.8, 1.7, NOF_STEPS),
            np.linspace(1, 2, NOF_STEPS),
        ],
    }
    for varying_params, vals in varying_params2vals.items():
        p0, p1 = varying_params
        p0_vals, p1_vals = vals
        target_mat = np.zeros(vals[0].size, vals[1].size)
        sense_mat = np.zeros(vals[0].size, vals[1].size)
        for p0_ind, p0_val in enumerate(p0_vals):
            for p1_ind, p1_val in enumerate(p1_vals):
                params = {**default_params, p0: p0_val, p1: p1_val}
                args = Namespace(params)
                target_acc, sens_acc = main(args, return_accuracy=True)
                target_mat[p0_ind, p1_ind] = target_acc
                sens_mat[p0_ind, p1_ind] = sens_acc

    # ('lambda_od', 'gamma_entropy')
    # ('gamma_od', 'gamma_entropy'): [[],[]]
    # for param0, param1, vals0, vals1 in ('lambda_od', 'gamma_od'):
    #     pass
