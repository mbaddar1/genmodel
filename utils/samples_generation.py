import torch
from sklearn.datasets import make_swiss_roll, make_circles, make_blobs, make_moons
from torch.distributions import Categorical, MultivariateNormal, MixtureSameFamily


def get_target_samples(dataset_name: str, n_samples: int, **kwargs) -> torch.Tensor:
    # Gaussian Mixture and multivariate normal
    if dataset_name == "gm":
        n_comp = kwargs["n_comp"]
        D = kwargs["D"]
        var = kwargs["var"]
        target_mix = Categorical(torch.ones(n_comp))
        t = 2 * torch.pi * torch.arange(n_comp) / n_comp
        mu_target = D * torch.stack([-torch.sin(t), torch.cos(t)], axis=1)  # FIXME, stack doesn't get axis param
        cov_target = var * torch.stack([torch.eye(2) for i in range(n_comp)])
        target_comp = MultivariateNormal(mu_target, cov_target)
        target_model = MixtureSameFamily(target_mix, target_comp)
        target_samples = target_model.sample(torch.Size((n_samples,)))
    elif dataset_name == "mvn":
        D = kwargs["D"]
        target_model = MultivariateNormal(D * torch.tensor([1.0, -1.0]),
                                          torch.tensor([[1.0, 0.9], [0.9, 1.0]]))
        target_samples = target_model.sample(torch.Size((n_samples,)))
    elif dataset_name == "swissroll":
        # Swissroll 2d
        target_samples = torch.tensor(make_swiss_roll(n_samples=n_samples, noise=1e-1)[0][:, [0, 2]] / 2.0)
    elif dataset_name == "circles":
        # circles
        target_samples = torch.tensor(make_circles(n_samples=n_samples, shuffle=True, factor=0.9, noise=0.05)[0] * 5.0)
    elif dataset_name == "blobs":
        # blobs
        target_samples = torch.tensor(make_blobs(n_samples=n_samples, n_features=2)[0])
    elif dataset_name == "moons":
        # moons
        target_samples = torch.tensor(make_moons(n_samples=n_samples, shuffle=True)[0] * 5.0)
    else:
        raise ValueError(f"Unsupported dataset : {dataset_name}")
    return target_samples
