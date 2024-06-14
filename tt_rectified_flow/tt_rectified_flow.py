"""
TODO
    1. Make dist figure 3 cols :
        pi0 pi1 and pi1_gen             DONE
    2. Use GeomLoss Sinkhorn            DONE
    3. Use Swissroll 2D data            DONE
    4. Use Circles Data                 DONE
    5. Use Blob Data                    DONE
    6. Use Half Moon Data               In-Progress
"""
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_circles, make_blobs, make_moons
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily
from geomloss import SamplesLoss
from functional_tt_fabrique import Extended_TensorTrain, orthpoly

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


######################## Sample ODE #########################

def v(z, t):
    # z :(N, d)
    N = z.shape[0]
    zt = torch.cat(
        [
            z,
            torch.ones(N, 1) * t,
        ],
        dim=1,
    )
    result = torch.zeros(N, d)
    for i in range(d):
        result[:, i] = ETTs[i](zt).ravel()
    return result


def mask_nan(x: torch.Tensor):
    B, D = x.shape[0], x.shape[1]
    mask = torch.tensor([True] * B)
    for d in range(D):
        mask_d = torch.bitwise_not(torch.isnan(x[:, d]))
        mask = torch.bitwise_and(mask_d, mask)
    res = x[mask, :]
    return res


def remove_outliers(x: torch.Tensor):
    alpha = 0.05
    B, D = x.shape[0], x.shape[1]
    mask = torch.tensor([True] * B)

    q = torch.quantile(input=x, q=torch.tensor([alpha, 1 - alpha]), dim=0)
    for d in range(D):
        min_ = q[0, d].item()
        max_ = q[1, d].item()
        cond1 = torch.ge(x[:, d], min_)
        cond2 = torch.le(x[:, d], max_)
        mask_d = torch.bitwise_and(cond1, cond2)
        mask = torch.bitwise_and(mask, mask_d)
    x_clean = x[mask, :]
    return x_clean


def filter_tensor(x: torch.Tensor):
    x1 = mask_nan(x)
    x1 = remove_outliers(x1)
    return x1


def sample_ode(z0: torch.Tensor, N: int = None) -> torch.Tensor:
    # NOTE: Use Euler method to sample from the learned flow
    dt = 1.0 / N
    traj = torch.empty((N + 1,) + z0.shape)
    # batch_size = z0.shape[0]

    traj[0] = z0
    z = z0
    for i in range(N):
        t = i / N
        pred = v(z, t)
        z = z + pred * dt
        traj[i + 1] = z

    return traj


########################## Plot results #########################


def draw_plot(model, z0, z1, N, tt_rank):
    from scipy.integrate import odeint
    traj = sample_ode(z0, N)
    # traj = torch.cat([torch.from_numpy(odeint(v, z0[i], torch.linspace(0., 1., steps=N), tfirst=False)) for i in range(z0.shape[0])], axis=0)

    fig = plt.figure()

    # need to set limits as some times we have very large values that shrinks the plot
    # plt.xlim(*limits)
    # plt.ylim(*limits)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.set_xlim(*limits)
    ax1.set_ylim(*limits)
    ax1.set_title("Init")

    ax2.set_xlim(*limits)
    ax2.set_ylim(*limits)
    ax2.set_title("Target")

    ax3.set_xlim(*limits)
    ax3.set_ylim(*limits)
    ax3.set_title("Generated")

    fig.suptitle('Distribution Samples')

    ax2.scatter(
        z1[:, 0].cpu().numpy(),
        z1[:, 1].cpu().numpy(),
        label="pi_1",  # r"$\pi_1$",
        alpha=0.15,
        color="red"
    )

    ax1.scatter(
        traj[0][:, 0].cpu().numpy(),
        traj[0][:, 1].cpu().numpy(),
        label="pi_0",  # r"$\pi_0$",
        alpha=0.15,
        color="green"
    )
    z1_ = filter_tensor(traj[-1])  # filtered version of traj
    ax3.scatter(
        z1_[:, 0].cpu().numpy(),
        z1_[:, 1].cpu().numpy(),
        label="Generated",
        alpha=0.15,
        color="blue"
    )

    # plt.figure(figsize=(4, 4))

    # # plot init, target and generated
    # plt.scatter(
    #     z1[:, 0].cpu().numpy(),
    #     z1[:, 1].cpu().numpy(),
    #     label="pi_1",  # r"$\pi_1$",
    #     alpha=0.15,
    # )
    # plt.scatter(
    #     traj[0][:, 0].cpu().numpy(),
    #     traj[0][:, 1].cpu().numpy(),
    #     label="pi_0",  # r"$\pi_0$",
    #     alpha=0.15,
    # )
    # plt.scatter(
    #     traj[-1][:, 0].cpu().numpy(),
    #     traj[-1][:, 1].cpu().numpy(),
    #     label="Generated",
    #     alpha=0.15,
    # )
    # plt.legend()
    # plt.title("Distribution")
    # plt.tight_layout()
    # print(f"Saving gen vs actual samples")
    plt.savefig(f"generated_vs_actual_samples_rank_{tt_rank}_deg_{degree}.png")

    # traj_particles = torch.stack(traj)
    plt.figure(figsize=(4, 4))
    plt.axis("equal")
    for i in range(30):
        plt.plot(traj[:, i, 0], traj[:, i, 1])
    plt.title("Transport Trajectory")
    plt.tight_layout()
    print("Saving trajectory figure")
    plt.savefig("trajectory.png")


################  Preliminaries  ##############################
if __name__ == '__main__':

    D = 10.0  # 10.0
    limits = (-15, 15)
    var: float = 0.3
    n_comp: int = 2
    n_samples: int = 10_000
    d: int = 2

    # check if cuda is available
    print(f"Is cuda available => {torch.cuda.is_available()}")
    # we use a standard Gaussian as prior
    mu_prior = torch.zeros(d)
    cov_prior = torch.eye(d)
    initial_model = MultivariateNormal(mu_prior, cov_prior)

    # the target distribution is a Gaussian mixture
    target_mix = Categorical(torch.ones(n_comp))
    t = 2 * torch.pi * torch.arange(n_comp) / n_comp
    mu_target = D * torch.stack([-torch.sin(t), torch.cos(t)], axis=1)  # FIXME, stack doesn't get axis param
    cov_target = var * torch.stack([torch.eye(2) for i in range(n_comp)])
    target_comp = MultivariateNormal(mu_target, cov_target)
    target_model = MixtureSameFamily(target_mix, target_comp)
    # target_model = MultivariateNormal(D * torch.tensor([1.0, -1.0]), torch.eye(d))

    samples_0 = initial_model.sample(torch.Size((n_samples,)))
    # samples from Gaussian Mixture
    # samples_1 = target_model.sample(torch.Size((n_samples,)))
    # samples from 2D swissroll
    # Same code from
    # 1 ) https://github.com/Jmkernes/Diffusion/blob/main/diffusion/ddpm/main.py#L44
    # 2 ) https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Deep_Unsupervised_Learning_using_Nonequilibrium_Thermodynamics/diffusion_models.py#L12
    # 3 ) https://github.com/mbaddar1/Diffusion/blob/281e453d66d413976bc069c75d736c6df3c4a9de/diffusion/ddpm/main.py#L50
    # samples_1 = torch.tensor(make_swiss_roll(n_samples=n_samples, noise=1e-1)[0][:, [0, 2]] / 2.0)
    # samples_1 = torch.tensor(make_circles(n_samples=n_samples, shuffle=True, factor=0.9,noise=0.05)[0]*5.0)
    # samples_1 = torch.tensor(make_blobs(n_samples=n_samples, n_features=2)[0])
    samples_1 = torch.tensor(make_moons(n_samples=n_samples, shuffle=True)[0]*5.0)
    # plot the samples

    plt.figure(figsize=(4, 4))
    plt.title(r"Samples from $\pi_0$ and $\pi_1$")
    plt.scatter(
        samples_0[:, 0].cpu().numpy(),
        samples_0[:, 1].cpu().numpy(),
        alpha=0.1,
        label=r"$\pi_0$",
    )
    plt.scatter(
        samples_1[:, 0].cpu().numpy(),
        samples_1[:, 1].cpu().numpy(),
        alpha=0.1,
        label=r"$\pi_1$",
    )

    # plt.scatter(
    #     mu_target[:, 0].cpu(),
    #     mu_target[:, 1].cpu(),
    #     marker="*",
    #     color="black",
    #     label=r"$\mu$",
    # )
    plt.legend()
    plt.xlim(*limits)
    plt.ylim(*limits)

    plt.tight_layout()
    plt.savefig("init_targe_dist.png")
    plt.clf()
    # sys.exit(-1)
    ################### Samples and targets  #############################################

    # M: int = 10  # number of $X_t$ for each tuple (X_0^i, X_1^i)
    # t = torch.rand(M)

    # X_t = samples_0[..., None] * t + samples_1[..., None] * (1.0 - t)  # (N, d, M)
    # X_t = torch.permute(X_t, (1, 2, 0))  # (M, N, d)
    # X_t = X_t.reshape((-1, d))  # (M*N, d)
    # X_t = torch.concatenate(
    #     [X_t, torch.tile(t, (n_samples, 1)).T.ravel()[:, None]], axis=1
    # )  # (M*N, d+1)
    # # X_t = torch.concatenate((X_t, t*torch.ones((M,N,1)).reshape(-1,1)), axis=1)

    # sample_points = X_t
    # targets = samples_1 - samples_0  # (N, d)
    # # targets = targets.repeat(0)
    # targets = torch.tile(targets, (M, 1, 1))  # (M, N, d)
    # targets = targets.reshape((-1, d))  # (M*N, d)
    t = torch.rand(n_samples, 1)
    X_t = samples_0 * (
            1. - t) + samples_1 * t  # (N, d) # this is the hotfix chalres made to original code on 30 Apr 2024
    X_t = torch.cat([X_t, t], axis=1)
    targets = samples_1 - samples_0  # (N, d)

    print(samples_0)
    print(samples_1)
    print(samples_0.shape)
    print(X_t)
    print("shape of targets:", targets.shape)
    print("samples shape:", X_t.shape)

    ##############  TT fitting #################################

    print(f"Starting tt fitting")
    # TT parameters
    tt_rank = 8
    degree = 20
    degrees = [degree] * (d + 1)  # hotfix by charles that made the GMM work
    ranks = [1] + [tt_rank] * d + [1]

    domain = [list(limits) for _ in range(d)] + [[0, 1]]

    op = orthpoly(degrees, domain)

    ETTs = [Extended_TensorTrain(op, ranks) for i in range(d)]
    # ETT = Extended_TensorTrain(op, ranks)

    # ALS parameters
    reg_coeff = 1e-20
    iterations = 40
    tol = 5e-10

    # define data range for fitting and data samples
    # sample_seed = 7771
    # sample_params = [16000, 6, sample_seed]
    # num_samples = max(
    #     sample_params[0], sample_params[1] * d * (max(degrees) + 1) * max(ranks) ** 2
    # )  # number of samples proportional to TT parameters to prevent underfitting
    # print("number of samples:", num_samples)
    # num_val_samples = max(1000, d * (max(degrees) + 1) * max(ranks) ** 2)

    print("============= Fitting TT ==============")
    rule = None
    # rule = tt.Dörfler_Adaptivity(delta = 1e-6,  maxranks = [32]*(n-1), dims = [feature_dim]*n, rankincr = 1)
    for i, ETT in enumerate(ETTs):
        ETT.fit(
            X_t,
            targets[:, i: i + 1],
            iterations=iterations,
            rule=rule,
            tol=tol,
            verboselevel=1,
            reg_param=reg_coeff,
        )
        ETT.tt.set_core(d)
    N = 1000
    draw_plot(ETTs, initial_model.sample(torch.Size((n_samples,))), samples_1, N, tt_rank=tt_rank)

    # Calculate Sinkhorn losses

    # sample_ref_1 = target_model.sample(torch.Size((n_samples,)))  # .cuda()
    # sample_ref_2 = target_model.sample(torch.Size((n_samples,)))  # .cuda()
    sample_ref_1 = torch.tensor(make_swiss_roll(n_samples=n_samples, noise=1e-1)[0][:, [0, 2]] / 2.0)
    sample_ref_2 = torch.tensor(make_swiss_roll(n_samples=n_samples, noise=1e-1)[0][:, [0, 2]] / 2.0)

    # check statistics for close samples
    mean_0 = torch.mean(samples_1, dim=0)
    mean_1 = torch.mean(sample_ref_1, dim=0)
    mean_2 = torch.mean(sample_ref_2, dim=0)

    loss = SamplesLoss(loss="sinkhorn")
    L_ref = loss(sample_ref_1, sample_ref_2)

    z0 = initial_model.sample(torch.Size((n_samples,)))
    traj = sample_ode(z0, N)

    # tmp = traj[-1]
    # nan_count = torch.sum(torch.isnan(tmp))
    # min_ = tmp.min(dim=0)[0]
    # max_ = tmp.max(dim=0)[0]
    # k_min = torch.kthvalue(input=tmp, k=1, dim=0)
    z1_ = filter_tensor(traj[-1])  # filtered endpoint in traj
    L_gen_1 = loss(z1_, sample_ref_1)
    L_gen_2 = loss(z1_, sample_ref_2)

    print(f"rank = {tt_rank}")
    print(f"deg = {degree}")
    print(f"Ref sinkhorn = {L_ref}")
    print(f"Gen sinkhorn loss = {(L_gen_1 + L_gen_2) / 2.0}")

"""
Quick Experiments results
---
i) 
Model : TT-Recflow with legendre poly 
Dataset : Swissroll 2D 
Opt : Fixed Rank TT-ALS
Results: 
Numerically
Ref sinkhorn divergence = 0.003
Best results so far
Rank        Deg         Sinkhorn
6           14          0.08
6           15          0.09
6           16          0.06 *
------------------------------------------
ii)
Model : TT-Recflow with legendre poly 
Dataset : Circles
Opt : Fixed Rank TT-ALS
Numerically
Ref sinkhorn divergence = 0.01
Rank        Deg         Sinkhorn
6           10          0.76
6           15          0.63
7           10          0.74 *
-----------------------------------------
iii)
Model : TT-Recflow with legendre poly 
Dataset : Blobs (no. centres = 3)
Opt : Fixed Rank TT-ALS
Numerically
Ref sinkhorn divergence = 0.01
Rank        Deg         Sinkhorn
8           20          11.2
8           25          10.11
8           30          9.4 *
8           35          10.42
8           40          13.77
iv) 
Model : TT-Recflow with legendre poly 
Dataset : Moons 
Opt : Fixed Rank TT-ALS
Numerically
Ref sinkhorn divergence = 0.005
Rank        Deg         Sinkhorn
6           20          3.99 *
7           10          4.12
7           20          4.12
8           20          4.17
"""
