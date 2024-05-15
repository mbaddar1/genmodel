from functional_tt_fabrique import Extended_TensorTrain, orthpoly
from tt_fabrique import TT_Fabrique, power_iteration, get_CovMatrix_from_TT_list

import torch

torch.set_default_dtype(torch.float64)

from math import exp, sqrt, pi

from modelclass import Custom_Polynom_Arithmetic

# from graphics import visualize_pdf
from copy import deepcopy

from tt_fabrique import TensorTrain


from colorama import Fore, Style
import time


def test_als():
    d = 10
    degrees = [3] * d
    ranks = [1] + [2] * (d - 1) + [1]

    domain = [[-1.0, 1.0] for _ in range(d)]

    op = orthpoly(degrees, domain)

    ETT = Extended_TensorTrain(op, ranks)
    # Check ALS FIT

    # ALS parameters
    reg_coeff = 1e-2
    iterations = 40
    tol = 5e-10

    # define data range for fitting and data samples
    sample_seed = 7771
    sample_params = [16000, 6, sample_seed]
    num_samples = max(
        sample_params[0], sample_params[1] * d * (max(degrees) + 1) * max(ranks) ** 2
    )  # number of samples proportional to TT parameters to prevent underfitting
    print("number of samples:", num_samples)
    num_val_samples = max(1000, d * (max(degrees) + 1) * max(ranks) ** 2)

    torch.manual_seed(sample_seed)

    # uniform sampling on the hypercube
    strip = domain[0]
    sample_points = (strip[1] - strip[0]) * torch.rand((num_samples, d)) + strip[0]
    validation_points = (strip[1] - strip[0]) * torch.rand(
        (num_val_samples, d)
    ) + strip[0]

    # get targets
    def poly(U):
        """evaluates the log density of the prior.

        Args:
            U (torch.tensor): Sample tensor of shape (N,D), where N is the number of samples and
                                D is the dimension.

        Returns:
            res (torch.tensor): tensor of density evaluations of shape (N,).
        """
        assert len(U.shape) == 2
        d = U.shape[1]
        mean_diff = U - torch.zeros_like(U)
        P0 = 0.1 * torch.eye(d)
        P0_inv = torch.linalg.inv(P0)
        P0_det = torch.linalg.det(P0)
        res = +torch.einsum("ni,ij,nj->n", mean_diff, P0_inv, mean_diff) / 2 + sqrt(
            ((P0_det) * (2 * pi) ** d)
        )  # corresponds to phi in exp(-phi) (but with normalization constant)
        return res.unsqueeze(1)

    targets = poly(sample_points)
    val_targets = poly(validation_points)

    print("============= Fitting V0 to terminal value ==============")
    rule = None
    # rule = tt.Dörfler_Adaptivity(delta = 1e-6,  maxranks = [32]*(n-1), dims = [feature_dim]*n, rankincr = 1)
    ETT.fit(
        sample_points,
        targets,
        iterations=iterations,
        rule=rule,
        tol=tol,
        verboselevel=1,
        reg_param=reg_coeff,
    )
    ETT.tt.set_core(d - 1)

    print("ETT values: ", ETT(sample_points))
    print("Density values: ", ETT.evaluate_density(sample_points))
    print("True value of polynomial in 0: ", poly(torch.zeros((1, d))).item())
    print("TT approximation in 0: ", ETT((torch.zeros(1, d))).item())

    train_error = (
        torch.norm(ETT(sample_points) - targets) ** 2 / torch.norm(targets) ** 2
    ).item()
    val_error = (
        torch.norm(ETT(validation_points) - val_targets) ** 2
        / torch.norm(val_targets) ** 2
    ).item()
    print("relative error on training set: ", train_error)
    print("relative error on validation set: ", val_error)
    print("========================================================")

    # Check that the TT captures the correct density
    x = (strip[1] - strip[0]) * torch.rand((100, 2)) + strip[0]

    from matplotlib import pyplot as plt

    visualize_pdf(ETT.marginal_density)
    plt.savefig("tt_density.pdf")


if __name__ == "__main__":
    test_als()
