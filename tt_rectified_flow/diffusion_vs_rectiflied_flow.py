# -*- coding: utf-8 -*-
"""Diffusion vs Rectiflied flow

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g9Bw09S9f5NkMASDFUFzdOM5DcoSkH5l

Sources:
- Rectified flow:
  - https://arxiv.org/abs/2209.03003
  - https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html
  - https://github.com/gnobitab/RectifiedFlow
- Denoising diffusion model:
  - https://arxiv.org/abs/2006.11239
  - https://github.com/dataflowr/notebooks/blob/master/Module18/ddpm_micro_sol.ipynb

More :
    https://huggingface.co/blog/Isamu136/insta-rectified-flow

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from tqdm.autonotebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    """Multilayer perceptron

    Parameters
    ----------
    sizes : List[int]
        Size of each layer (input + hidden + output)
    activation : nn.Module, optional
        Activation function, by default nn.ReLU()
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: nn.Module = nn.ReLU(),
    ):
        # construct the layers
        super().__init__()
        sizes = [input_dim+1] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self._model = nn.Sequential(*layers)

        # apply weight initialization
        #self.apply(self.init_weights)

    def init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      inputs = torch.cat([x, t], dim=1)
      return self._model(inputs)

"""# Toy example

- $\pi_0$ is a standard Gaussian
- $\pi_1$ is a Gaussian mixture
"""

from torch.distributions import Normal, Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.mixture_same_family import MixtureSameFamily

D = 10.
limits = (-D-5, D+5)
var: float = 0.3
n_comp: int = 5
n_samples: int = 10_000

"""
initial_mix = Categorical(torch.ones(n_comp))
initial_comp = MultivariateNormal(torch.FloatTensor([[D * np.sqrt(3) / 2., D / 2.],
                                                     [-D * np.sqrt(3) / 2., D / 2.],
                                                     [0.0, - D * np.sqrt(3) / 2.]]),
                                  var * torch.stack([torch.eye(2) for i in range(n_comp)]))
initial_model = MixtureSameFamily(initial_mix, initial_comp)
"""
# we use a standard Gaussian as prior
initial_model = MultivariateNormal(torch.tensor([0., 10.]), torch.eye(2))
samples_0 = initial_model.sample((n_samples,))

# the target distribution is a Gaussian mixture
target_mix = Categorical(torch.ones(n_comp))
t = 2*torch.pi*torch.arange(n_comp)/n_comp
loc = D * torch.stack([-torch.sin(t), torch.cos(t)], axis=1)
target_comp = MultivariateNormal(loc,
                                 var*torch.stack([torch.eye(2) for i in range(n_comp)]))
target_model = MixtureSameFamily(target_mix, target_comp)
samples_1 = target_model.sample((n_samples,))
print('Shape of the samples:', samples_0.shape, samples_1.shape)

plt.figure(figsize=(4,4))
plt.title(r'Samples from $\pi_0$ and $\pi_1$')
plt.scatter(samples_0[:, 0].cpu().numpy(), samples_0[:, 1].cpu().numpy(),
            alpha=0.1, label=r'$\pi_0$')
plt.scatter(samples_1[:, 0].cpu().numpy(), samples_1[:, 1].cpu().numpy(),
            alpha=0.1, label=r'$\pi_1$')
plt.scatter(loc[:, 0].cpu(), loc[:, 1].cpu(),
            marker="*", color="black", label=r'$\mu$')
plt.legend()
plt.xlim(*limits)
plt.ylim(*limits)

plt.tight_layout()

"""## Flow model

Given empirical observations of $X_0\sim \pi_0$ and $X_1\sim \pi_1$,
the rectified flow induced from $(X_0,X_1)$
is an ordinary differentiable model (ODE)
on time $t\in[0,1]$,
$$
d Z_t = v^X(Z_t, t) d t,
$$
which $v$ is set in a way that ensures that $Z_1$ follows $\pi_1$ when $Z_0 \sim \pi_0$. Let $X_t = t X_1 + (1-t)X_0$ be the linear interpolation of $X_0$ and $X_1$. Then $v$ is given by
$$
v^X(z,t) = \mathbb{E}[X_1 - X_0 ~|~ X_t = z  ] =  \arg\min_{v} \int_0^1 \mathbb{E}[|| X_1-X_0 - v(X_t,t) ||^2] \mathrm{d} t,
$$
where the (conditional) expectation is w.r.t. the joint distribution of $(X_0,X_1)$.

We parameterize $v^X(z,t)$ with a 3-layer neural network.
"""

class RectifiedFlow(nn.Module):
  def __init__(self, model: nn.Module, num_steps: int = 1000):
    super().__init__()
    self.model = model
    self.N = num_steps

  def get_train_tuple(self, z0: torch.Tensor,
                      z1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.rand((z1.shape[0], 1))
    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0

    return z_t, t, target

  def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return self.model(x, t)

  @torch.no_grad()
  def sample_ode(self, z0: torch.Tensor, N: int = None) -> torch.Tensor:
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
      N = self.N
    dt = 1./N
    traj = torch.empty((N+1,) + z0.shape)
    batch_size = z0.shape[0]

    traj[0] = z0
    z = z0
    for i in range(N):
      t = torch.ones((batch_size,1)) * i / N
      pred = self.forward(z.to(device), t.to(device)).cpu()
      z = z + pred * dt
      traj[i+1] = z

    return traj

"""We define the training method here. The loss function is:
$$
\min_{\theta}
\int_0^1 E_{X_0 \sim \pi_0, X_1 \sim \pi_1} \left [ {||( X_1 - X_0) - v_\theta\big (X_t,~ t\big)||}^2
\right ] \text{d}t,
~~~~~\text{with}~~~~
X_t = t X_1 + (1-t) X_0.
$$
"""

def train_rectified_flow(rectified_flow: RectifiedFlow,
                         optimizer: torch.optim.Optimizer,
                         pairs: torch.Tensor,
                         batch_size: int,
                         n_iters: int) -> tuple[RectifiedFlow, torch.Tensor]:
  loss_curve = torch.empty((n_iters+1,))

  pbar = tqdm(range(n_iters+1))
  for i in pbar:
    optimizer.zero_grad()
    indices = torch.randperm(len(pairs))[:batch_size]
    batch = pairs[indices]
    z0 = batch[:, 0]
    z1 = batch[:, 1]
    z_t, t, target = rectified_flow.get_train_tuple(z0=z0, z1=z1)
    target = target.to(device)

    pred = rectified_flow.model(z_t.to(device), t.to(device))
    loss = (target - pred).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
    loss = loss.mean()
    loss.backward()

    optimizer.step()
    loss_curve[i] = loss.item()
    if i % 50 == 0:
      pbar.set_postfix(loss=loss.item())

  return rectified_flow, loss_curve

@torch.no_grad()
def draw_plot(rectified_flow: RectifiedFlow,
              z0: torch.Tensor,
              z1: torch.Tensor,
              N: int = None) -> None:
  traj = rectified_flow.sample_ode(z0=z0, N=N)

  plt.figure(figsize=(4,4))
  plt.xlim(*limits)
  plt.ylim(*limits)

  plt.scatter(z1[:, 0].cpu().numpy(), z1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
  plt.scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
  plt.scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
  plt.legend()
  plt.title('Distribution')
  plt.tight_layout()

  #traj_particles = torch.stack(traj)
  plt.figure(figsize=(4,4))
  plt.axis('equal')
  for i in range(30):
    plt.plot(traj[:, i, 0], traj[:, i, 1])
  plt.title('Transport Trajectory')
  plt.tight_layout()

"""### 1-Rectified flow"""

x_0 = samples_0[torch.randperm(len(samples_0))]
x_1 = samples_1[torch.randperm(len(samples_1))]
x_pairs = torch.stack([x_0, x_1], dim=1)
print(x_pairs.shape)

"""We create ```rectified_flow_1``` and its corresponding ```optimizer``` and traing 1-Rectified Flow with ```train_rectified_flow``` using $(X_0, X_1)$ above."""

iterations = 10000
batch_size = 2048
input_dim = 2

model = MLP(input_dim, [100]*2, input_dim, activation=nn.Tanh())
rectified_flow_1 = RectifiedFlow(model=model, num_steps=100).to(device)
optimizer = torch.optim.Adam(rectified_flow_1.model.parameters(), lr=5e-3)

rectified_flow_1, loss_curve = train_rectified_flow(rectified_flow_1, optimizer,
                                                    x_pairs, batch_size, iterations)
plt.plot(np.linspace(0, iterations, iterations+1), loss_curve[:(iterations+1)])
plt.yscale("log")
plt.grid()
plt.title('Training Loss Curve')

"""We run the Euler method to solve the ODE with $N=1000$ steps to generate samples from 1-Rectified Flow.

Orange dots = samples from $\pi_0$

Blue dots   = samples from $\pi_1$

Green dots  = samples from 1-Rectified Flow

1-Rectified Flow successfully learned a velocity field $v(Z_t, t)$ that can transport $\pi_0$ to $\pi_1$.
"""

draw_plot(rectified_flow_1, z0=initial_model.sample([2000]), z1=samples_1, N=100)

"""We can see that the trajectories above fit well on the trajectories of linear intepolation of data, but are "rewired" in the center when the trajectories are intersect. Hence, the resulting trajectories are either (almost)straight, or has a ">" shape.

Due to the non-straight ">"-shape paths, if we simulate the ODE with a small step $N$, we would obtain poor performance, as shown below ($N=1$).
"""

draw_plot(rectified_flow_1, z0=initial_model.sample([2000]), z1=samples_1, N=1)

"""### Reflow for 2-Rectified Flow
Now let's use the *reflow* procedure to get a straightened rectified flow,
denoted as 2-Rectified Flow, by repeating the same procedure on with $(X_0,X_1)$ replaced by  $(Z_0^1, Z_1^1)$, where   $(Z_0^1, Z_1^1)$ is the coupling simulated from 1-Rectified Flow.
Specifically, we randomly sample 10000 $Z_0^1$  and generate their corresponding  $Z_1^1$ by simulating 1-Rectified Flow.
"""

z10 = samples_0.detach().clone()
traj = rectified_flow_1.sample_ode(z0=z10.detach().clone(), N=100)
z11 = traj[-1].detach().clone()
z_pairs = torch.stack([z10, z11], dim=1)
print(z_pairs.shape)

"""The coupling $(Z_0^1, Z_1^1)$ is now deterministic. The loss function is:
$$
\min_{\theta}
\int_0^1 E_{(Z_0, Z_1) \sim (Z_0^1, Z_1^1)} \left [ {||( Z_1 - Z_0) - v_\theta\big (Z_t,~ t\big)||}^2
\right ] \text{d}t,
~~~~~\text{with}~~~~
Z_t = t Z_1 + (1-t) Z_0.
$$

We create ```rectified_flow_2``` and its corresponding ```optimizer``` and training 2-Rectified Flow with ```train_rectified_flow```.
"""

reflow_iterations = 50_000

model = MLP(input_dim, [100]*2, input_dim)
rectified_flow_2 = RectifiedFlow(model=model, num_steps=100).to(device)
optimizer = torch.optim.Adam(rectified_flow_2.model.parameters(), lr=5e-3)

rectified_flow_2, loss_curve = train_rectified_flow(rectified_flow_2, optimizer,
                                                    z_pairs, batch_size,
                                                    reflow_iterations)
plt.plot(np.linspace(0, reflow_iterations, reflow_iterations+1), loss_curve[:(reflow_iterations+1)])
plt.yscale("log")

"""We run the Euler method to solve the ODE with $N=100$ steps to generate samples from 2-Rectified Flow.

2-Rectified Flow can also successfully learn a velocity field $v(Z_t, t)$ that can transport $\pi_0$ to $\pi_1$.

The key point, however, is that the transport trajectory is now **straightened** and hence we would not lose much accuracy even if we solve the ODE with one Euler step ($N=1$).
"""

draw_plot(rectified_flow_2, z0=initial_model.sample([1000]), z1=samples_1.detach().clone())

"""So below is the result when we solve the ODE with $N=1$ Euler step to generate samples from 2-Rectified Flow.

With the **straightened** trajectory, we get almost perfect results with only  one-step generation! Thanks to the power of **Reflow** !

More croncretely, the output of the model is now $Z_1 = Z_0 + v(Z_0, 0)$.

Effectively, we have trained a one-step model, by using ODE as an intermediate step.
"""

draw_plot(rectified_flow_2, z0=initial_model.sample([1000]), z1=samples_1.detach().clone(), N=1)

"""## [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
(J. Ho, A. Jain, P. Abbeel 2020)

![](https://raw.githubusercontent.com/dataflowr/website/master/modules/extras/diffusions/ddpm.png)


Given a schedule $\beta_1<\beta_2<\dots <\beta_T$, the **forward diffusion process** is defined by:
$q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1},\beta_t I)$ and $q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$.

With $\alpha_t = 1-\beta_t$ and $\overline{\alpha_t} = \prod_{i=1}^t\alpha_i$, we see that, with $\epsilon\sim\mathcal{N}(0,I)$:
\begin{align*}
x_t = \sqrt{\overline{\alpha}_t}x_0 + \sqrt{1-\overline{\alpha}_t}\epsilon.
\end{align*}
The law $q(x_{t-1}|x_t,\epsilon)$ is explicit: $q(x_{t-1}|x_t,\epsilon) = \mathcal{N}(x_{t-1};\mu(x_t,\epsilon,t), \gamma_t I)$ with,
\begin{align*}
\mu(x_t,\epsilon, t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon\right)\text{ and, }
\gamma_t = \frac{1-\overline{\alpha}_{t-1}}{1-\overline{\alpha}_{t}}\beta_t
\end{align*}


**Training**: to approximate **the reversed diffusion** $q(x_{t-1}|x_t)$ by a neural network given by $p_{\theta}(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_{\theta}(x_t,t), \beta_t I)$ and $p(x_T) \sim \mathcal{N}(0,I)$, we maximize the usual Variational bound:
\begin{align*}
\mathbb{E}_{q(x_0)} \ln p_{\theta}(x_0) &\geq L_T +\sum_{t=2}^T L_{t-1}+L_0 \text{ with, }L_{t-1} = \mathbb{E}_q\left[ \frac{1}{2\sigma_t^2}\|\mu_\theta(x_t,t) -\mu(x_t,\epsilon,t)\|^2\right].
\end{align*}
With the change of variable:
\begin{align*}
\mu_\theta(x_t,t) = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\right),
\end{align*}
ignoring the prefactor and sampling $\tau$ instead of summing over all $t$, the loss is finally:
\begin{align*}
\ell(\theta) = \mathbb{E}_\tau\mathbb{E}_\epsilon \left[ \|\epsilon - \epsilon_\theta(\sqrt{\overline{\alpha}_\tau}x_0 + \sqrt{1-\overline{\alpha}_\tau}\epsilon, \tau)\|^2\right]
\end{align*}



**Sampling**: to simulate the reversed diffusion with the learned $\epsilon_\theta(x_t,t)$ starting from $x_T\sim \mathcal{N}(0,I)$, iterate for $t=T,\dots, 1$:
\begin{align*}
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left( x_t-\frac{1-\alpha_t}{\sqrt{1-\overline{\alpha}_t}}\epsilon_\theta(x_t,t)\right)+\sqrt{\beta_t}\epsilon,\text{ with } \epsilon\sim\mathcal{N}(0,I).
\end{align*}
"""

class DDPM(nn.Module):
    def __init__(self, network: nn.Module,
                 num_timesteps: int,
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02) -> None:
        super(DDPM, self).__init__()
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.network = network
        self.device = device
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5 # used in add_noise
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5 # used in add_noise and step

    def add_noise(self,
                  x_start: torch.Tensor,
                  x_noise: torch.Tensor,
                  timesteps: torch.Tensor) -> torch.Tensor:
        # The forward process
        # x_start and x_noise (bs, d)
        # timesteps (bs)
        s1 = self.sqrt_alphas_cumprod[timesteps] # bs
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps] # bs
        s1 = s1.reshape(-1,1) # (bs, 1) for broadcasting
        s2 = s2.reshape(-1,1) # (bs, 1)
        return s1 * x_start + s2 * x_noise

    def reverse(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # The network return the estimation of the noise we added
        return self.network(x, t.view(-1, 1))

    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor) -> torch.Tensor:
        # one step of sampling
        # timestep (1)
        t = timestep
        coef_epsilon = (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod
        coef_eps_t = coef_epsilon[t].reshape(-1,1)
        coef_first = 1/self.alphas ** 0.5
        coef_first_t = coef_first[t].reshape(-1,1)
        pred_prev_sample = coef_first_t*(sample-coef_eps_t*model_output)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output).to(self.device)
            variance = ((self.betas[t] ** 0.5) * noise)

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

def training_loop(model: DDPM,
                  optimizer: torch.optim.Optimizer,
                  targets: torch.Tensor,
                  batch_size: int,
                  num_epochs: int) -> tuple[DDPM, torch.Tensor]:
    """Training loop for DDPM"""
    loss_curve = torch.empty((num_epochs,))

    pbar = tqdm(range(num_epochs))
    model.train()
    for epoch in pbar:
        indices = torch.randperm(len(targets))[:batch_size]
        batch = targets[indices].to(device) # (B, d)

        noise = torch.randn(batch.shape, device=device)
        timesteps = torch.randint(0, model.num_timesteps, (batch.shape[0],),
                                  dtype=torch.long,
                                  device=device)

        noisy = model.add_noise(batch, noise, timesteps)
        noise_pred = model.reverse(noisy, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.detach().item())
        loss_curve[epoch] = loss.detach().cpu().item()

    return model, loss_curve

x_1 = samples_1[torch.randperm(len(samples_1))]
print(x_1.shape)

num_timesteps = 1000
iterations = 20_000
network = MLP(input_dim, [100]*2, input_dim)
network = network.to(device)
model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02)
optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
ddpm_1, loss_curve = training_loop(model, optimizer, x_1, batch_size, iterations)

plt.plot(np.linspace(0, iterations, iterations), loss_curve[:iterations])
plt.yscale("log")
plt.grid()
plt.title('Training Loss Curve')

def generate_points(ddpm: DDPM,
                    dimension: int,
                    sample_size: int,
                    steps: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate the points from the Gaussian noise"""
    points = torch.empty((sample_size, dimension))
    ddpm.eval()
    if steps is None:
      steps = ddpm.num_timesteps
    traj = torch.empty((steps+1, sample_size, dimension))

    with torch.no_grad():
        sample = torch.randn(sample_size, dimension, device=device)
        traj[-1] = sample.detach().cpu()
        for i, t in enumerate(tqdm(range(steps-1, -1, -1))):
            time_tensor = torch.ones(sample_size,1, dtype=torch.long,
                                     device=device) * t
            residual = ddpm.reverse(sample, time_tensor)
            sample = ddpm.step(residual, time_tensor[0], sample)
            traj[t] = sample.detach().cpu()

        points = sample.detach().cpu()
    return traj, points

traj, points = generate_points(model, 2, 1000, steps=500)

fig, axes = plt.subplots(ncols=2, figsize=(8, 4))

axes[0].scatter(x_1[:, 0].cpu().numpy(), x_1[:, 1].cpu().numpy(), label=r'$\pi_1$', alpha=0.15)
axes[0].scatter(traj[-1][:, 0].cpu().numpy(), traj[-1][:, 1].cpu().numpy(), label=r'$\pi_0$', alpha=0.15)
axes[0].scatter(traj[0][:, 0].cpu().numpy(), traj[0][:, 1].cpu().numpy(), label='Generated', alpha=0.15)
axes[0].set_title('Distribution')
fig.tight_layout()
axes[0].legend()
axes[0].set_xlim(*limits)
axes[0].set_ylim(*limits)

#traj_particles = torch.stack(traj)
axes[1].axis('equal')
idx = torch.randint(0, traj.shape[1], size=(30,))
for i in idx:
  axes[1].plot(traj[:,i, 0], traj[:,i,1], linewidth=1)
axes[1].set_title('Transport Trajectory')
axes[1].set_xlim(*limits)
axes[1].set_ylim(*limits)
fig.tight_layout()

