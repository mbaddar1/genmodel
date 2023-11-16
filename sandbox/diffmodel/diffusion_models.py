"""
Copied From
https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Deep_Unsupervised_Learning_using_Nonequilibrium_Thermodynamics/diffusion_models.py
as a part of this tutorial
https://papers-100-lines.medium.com/diffusion-models-from-scratch-tutorial-in-100-lines-of-pytorch-code-5dac9f472f1c

"""
import datetime
import logging
from typing import List

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_circles
from datetime import datetime


def mvn_sample_batch(size):
    A = torch.tensor([[0.2, 5], [0.5, 4.0]])
    cov = torch.matmul(A, A.T)
    mean = torch.tensor([-1.5, 2.0])
    dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
    sample = dist.sample(torch.Size([size]))
    return sample.detach().numpy()


def circles_sample_batch(size):
    x, _ = make_circles(n_samples=size, shuffle=True, noise=0.05, random_state=0, factor=0.3)
    return x


def swiss_roll_sample_batch(size):
    x, _ = make_swiss_roll(size)
    return x[:, [2, 0]] / 10.0 * np.array([1, -1])


class MLP(nn.Module):

    def __init__(self, N=40, data_dim=2, hidden_dim=64):
        super(MLP, self).__init__()

        self.network_head = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.network_tail = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                         nn.ReLU(), nn.Linear(hidden_dim, data_dim * 2)
                                                         ) for _ in range(N)])

    def forward(self, x, t: int):
        h = self.network_head(x)
        return self.network_tail[t](h)


class DiffusionModel(nn.Module):

    def __init__(self, model: nn.Module, n_steps=40, device='cuda'):
        super().__init__()

        self.model = model
        self.device = device

        betas = torch.linspace(-18, 10, n_steps)
        self.beta = torch.sigmoid(betas) * (3e-1 - 1e-5) + 1e-5

        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta

    def forward_process(self, x0, t):

        t = t - 1  # Start indexing at 0
        beta_forward = self.beta[t]
        alpha_forward = self.alpha[t]
        alpha_cum_forward = self.alpha_bar[t]
        xt = x0 * torch.sqrt(alpha_cum_forward) + torch.randn_like(x0) * torch.sqrt(1. - alpha_cum_forward)
        # Retrieved from https://github.com/Sohl-Dickstein/Diffusion-Probabilistic-Models/blob/master/model.py#L203
        mu1_scl = torch.sqrt(alpha_cum_forward / alpha_forward)
        mu2_scl = 1. / torch.sqrt(alpha_forward)
        cov1 = 1. - alpha_cum_forward / alpha_forward
        cov2 = beta_forward / alpha_forward
        lam = 1. / cov1 + 1. / cov2
        mu = (x0 * mu1_scl / cov1 + xt * mu2_scl / cov2) / lam
        sigma = torch.sqrt(1. / lam)
        return mu, sigma, xt

    def reverse(self, xt, t):

        t = t - 1  # Start indexing at 0
        if t == 0: return None, None, xt
        mu, h = self.model(xt, t).chunk(2, dim=1)
        sigma = torch.sqrt(torch.exp(h))
        samples = mu + torch.randn_like(xt) * sigma
        return mu, sigma, samples

    def sample(self, size, device):
        noise = torch.randn((size, 2)).to(device)
        samples = [noise]
        for t in range(self.n_steps):
            _, _, x = self.reverse(samples[-1], self.n_steps - t - 1 + 1)
            samples.append(x)
        return samples


def plot(model: torch.nn.Module, dataset_name: str, n_epochs: int, training_losses: List[float], window: int):
    plt.figure(figsize=(10, 6))
    N = 5000
    if dataset_name == "swissroll":
        x0 = swiss_roll_sample_batch(N)
    elif dataset_name == "circles":
        x0 = circles_sample_batch(N)
    elif dataset_name == "mvn":
        x0 = mvn_sample_batch(N)
    else:
        raise ValueError(f'invalid dataset-name : {dataset_name}')
    x20 = model.forward_process(torch.from_numpy(x0).to(device), 20)[-1].data.cpu().numpy()
    x40 = model.forward_process(torch.from_numpy(x0).to(device), 40)[-1].data.cpu().numpy()
    data = [x0, x20, x40]
    # original data
    for i, t in enumerate([0, 20, 39]):
        plt.subplot(2, 3, 1 + i)
        plt.scatter(data[i][:, 0], data[i][:, 1], alpha=.1, s=1)
        # plt.xlim([-2, 2])
        # plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$q(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
        if i == 0: plt.title(r'$t=0$', fontsize=17)
        if i == 1: plt.title(r'$t=\frac{T}{2}$', fontsize=17)
        if i == 2: plt.title(r'$t=T$', fontsize=17)
    # sampled data
    samples = model.sample(5000, device)
    for i, t in enumerate([0, 20, 40]):
        plt.subplot(2, 3, 4 + i)
        plt.scatter(samples[40 - t][:, 0].data.cpu().numpy(), samples[40 - t][:, 1].data.cpu().numpy(),
                    alpha=.1, s=1, c='r')
        # plt.xlim([-2, 2])
        # plt.ylim([-2, 2])
        plt.gca().set_aspect('equal')
        if t == 0: plt.ylabel(r'$p(\mathbf{x}^{(0...T)})$', fontsize=17, rotation=0, labelpad=60)
    plt.savefig(f"Imgs/diffusion_model_{dataset_name}_nepochs_{n_epochs}.png", bbox_inches='tight')
    plt.clf()
    plt.title('loss curve')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.plot(list(np.arange(1, len(training_losses) + 1)), training_losses)
    for i in range(len(training_losses)):
        start = max(i - window, 0)
        end = i
        training_losses[i] = np.mean(training_losses[start:end])
    plt.savefig(f'loss_curve/loss_curve_{dataset_name}_nepochs_{n_epochs}')
    plt.close()


def train(model, optimizer, nb_epochs, batch_size, dataset_name):
    training_losses = []
    for _ in tqdm(range(nb_epochs)):
        if dataset_name == "swissroll":
            x0 = torch.from_numpy(swiss_roll_sample_batch(batch_size)).float().to(device)
        elif dataset_name == "circles":
            x0 = torch.from_numpy(circles_sample_batch(batch_size)).float().to(device)
        elif dataset_name == "mvn":
            x0 = torch.from_numpy(mvn_sample_batch(batch_size)).float().to(device)
        else:
            raise ValueError(f'invalid dataset = {dataset_name}')
        t = np.random.randint(2, 40 + 1)
        mu_posterior, sigma_posterior, xt = model.forward_process(x0, t)
        mu, sigma, _ = model.reverse(xt, t)

        KL = (torch.log(sigma) - torch.log(sigma_posterior) + (sigma_posterior ** 2 + (mu_posterior - mu) ** 2) / (
                2 * sigma ** 2) - 0.5)
        loss = KL.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_losses.append(loss.item())

    return training_losses


formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('diff-model')
logger.setLevel(logging.INFO)

# create file handler which logs even debug messages
fh = logging.FileHandler(f'logs/diff_model_{datetime.now().isoformat()}.log')
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
logger.addHandler(fh)

if __name__ == "__main__":
    dataset_name = "swissroll"
    n_epochs = int(10)
    loss_window = 3000
    batch_size = 64_000
    assert dataset_name in ["swissroll", "circles", "blobs", "mvn"]

    device = torch.device('cpu')

    start_time = datetime.now()

    model_mlp = MLP(hidden_dim=128).to(device)
    model = DiffusionModel(model_mlp)
    logger.info(
        f'Starting training at {start_time} \n'
        f'params: dataset = {dataset_name},n_epochs= {n_epochs} batch_size = {batch_size}\n'
        f'Model = {str(model)}')
    optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-4)

    training_losses = train(model=model, optimizer=optimizer, nb_epochs=n_epochs, batch_size=64_000,
                            dataset_name=dataset_name)
    plot(model=model, dataset_name=dataset_name, n_epochs=n_epochs, training_losses=training_losses, window=loss_window)
    end_time = datetime.now()
    logger.info(f'Training finished in {(end_time - start_time).seconds} seconds')

    """
    Run dump
    dataset : swissroll
    /home/mbaddar/mbaddar/phd/lrgen/venv/bin/python /home/mbaddar/mbaddar/phd/lrgen/sandbox/
    diffmodel/diffusion_models.py 
    100%|██████████| 150000/150000 [1:00:39<00:00, 41.22it/s]
    Device = cuda, n-epochs = 150000,batch_size = 64000 => training 3641 sec
    ---
    /home/mbaddar/mbaddar/phd/lrgen/venv/bin/python /home/mbaddar/mbaddar/phd/lrgen/sandbox/diffmodel/diffusion_models.py 
    Training params: dataset = mvn,n_epochs= 5000 , batch_size = 64000
    100%|██████████| 5000/5000 [01:26<00:00, 57.90it/s]
    Dataset-name = mvn,device = cuda, n-epochs = 5000,batch_size = 64000 => training 89 sec
    """
