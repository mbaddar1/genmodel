import numpy as np
import matplotlib.pyplot as plt
import tensap
from tictoc import TicToc
from scipy.integrate import solve_ivp


def posterior(N: int, d: int) -> np.ndarray:
    # D: float = 10.0
    # var: float = 0.3
    # n_comps = 2
    # weights = np.full((n_comps,), 1.0 / n_comps)
    # mixture_idx = np.random.choice(
    #     len(weights), size=N, replace=True, p=weights
    # )  # (N,)
    # t = 2 * np.pi * np.arange(n_comps) / n_comps
    # mu_target = D * np.stack([-np.sin(t), np.cos(t)], axis=1)  # (n_comps, d)
    # cov_target = var * np.stack([np.eye(d) for i in range(n_comps)])  # (n_comps, d, d)
    # y = np.stack(
    #     [
    #         np.random.multivariate_normal(mean=mu_target[i], cov=cov_target[i])
    #         for i in mixture_idx
    #     ]
    # )  # (N, d)
    # return y
    return np.random.multivariate_normal(
        mean=np.ones(d) * 2.0, cov=0.1 * np.eye(d), size=N
    )


n_samples: int = 30_000
d: int = 2
rv = tensap.RandomVector(
    [tensap.NormalRandomVariable()] * d + [tensap.UniformRandomVariable(0.0, 1.0)]
)
limits = (-5.0, 5.0)
samples = rv.random(n_samples)
samples_0, t = samples[:, :-1], samples[:, -1]
samples_1 = posterior(n_samples, d)
# plot the samples
plt.figure(figsize=(4, 4))
plt.title(r"Samples from $\pi_0$ and $\pi_1$")
plt.scatter(
    samples_0[:, 0],
    samples_0[:, 1],
    alpha=0.1,
    label=r"$\pi_0$",
)
plt.scatter(
    samples_1[:, 0],
    samples_1[:, 1],
    alpha=0.1,
    label=r"$\pi_1$",
)
plt.legend()
plt.xlim(*limits)
plt.ylim(*limits)

plt.tight_layout()
plt.show()

###################################### TT
degree: int = 15
bases = tensap.FunctionalBases(
    [
        tensap.PolynomialFunctionalBasis(x.orthonormal_polynomials(), range(degree + 1))
        for x in rv.random_variables
    ]
)

# t = np.random.uniform(0.0, 1.0, n_samples)
X_t = samples_0 * t[:, None] + samples_1 * (1.0 - t[:, None])  # (N, d)
X_train = np.column_stack((X_t, t[:, None]))  # (N, d+1)
Y_train = samples_1 - samples_0  # (N, d)

solver = tensap.TreeBasedTensorLearning.tensor_train(d + 1, tensap.SquareLossFunction())
# solver configuration
solver.bases = bases
solver.training_data = [X_train, Y_train]

with TicToc():
    f, output = solver.solve()
print("Stagnation criterion:", output["stagnation_criterion"])
print("Iter:", output["iter"])
train_error = solver.loss_function.test_error(f, [X_train, Y_train])
print("Train error:", train_error)


########## SAMPLING
def v(t: float, y: np.ndarray) -> np.ndarray:
    # y : (d,)
    # y = y.T
    # n = y.shape[0]
    # tmp = np.ones((n, 1)) * t
    # return f(np.c_([y, tmp])).T  # (d, n)
    return f(np.append(y, t)[None, :])[0]  # (d, )


def sample_ode(z0: np.ndarray, N: int) -> np.ndarray:
    # Euler
    dt = 1.0 / N
    traj = np.empty((N + 1,) + z0.shape)
    traj[0] = z0
    z = z0
    for i in range(N):
        t = i / N
        tmp = np.ones((z0.shape[0], 1)) * t
        pred = f(np.column_stack([z, tmp]))
        z = z + pred * dt
        traj[i + 1] = z
    return traj


N_test = n_samples
z0 = rv.random(N_test)[:, :-1]  # (N, d)
# outputs = np.stack([solve_ivp(v, (0.0, 1.0), z0[i]).y[:, -1] for i in range(N_test)])
outputs = sample_ode(z0, 100)[-1]
plt.figure(figsize=(4, 4))
plt.xlim(*limits)
plt.ylim(*limits)

plt.scatter(
    samples_1[:, 0],
    samples_1[:, 1],
    label=r"$\pi_1$",
    alpha=0.15,
)
plt.scatter(
    z0[:, 0],
    z0[:, 1],
    label=r"$\pi_0$",
    alpha=0.15,
)
plt.scatter(
    outputs[:, 0],
    outputs[:, 1],
    label="Generated",
    alpha=0.40,
)
plt.legend()
plt.title("Distribution")
plt.tight_layout()
plt.savefig("result.png")
plt.show()
