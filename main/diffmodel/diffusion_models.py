"""
Copied From
https://github.com/MaximeVandegar/Papers-in-100-Lines-of-Code/blob/main/Deep_Unsupervised_Learning_using_Nonequilibrium_Thermodynamics/diffusion_models.py
as a part of this tutorial
https://papers-100-lines.medium.com/diffusion-models-from-scratch-tutorial-in-100-lines-of-pytorch-code-5dac9f472f1c
https://kstathou.medium.com/how-to-set-up-a-gpu-instance-for-machine-learning-on-aws-b4fb8ba51a7c
"""
import datetime
import logging
import os.path
from typing import List, Union, Iterable
from argparse import ArgumentParser
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.utils.data
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, make_circles
from datetime import datetime

# Logging
run_timestamp = datetime.now().isoformat()  # run version
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('diffusion-model')


def plot_train_losses(training_losses: Union[List[float] | np.array], dataset_name: str, epoch: int, window: int,
                      run_timestamp: str):
    plt.title('loss curve')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.plot(list(np.arange(1, len(training_losses) + 1)), training_losses)
    for i in range(len(training_losses)):
        start = max(i - window, 0)
        end = i
        training_losses[i] = np.mean(training_losses[start:end])
    plt.savefig(f'loss_curve/loss_curve_{dataset_name}_epoch_{epoch}_{run_timestamp}.png')
    plt.close()


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in tqdm(files, desc="delete old checkpoints"):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")


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
        self.name = "nn_head_tail"
        self.network_head = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(),
                                          nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), )
        self.network_tail = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                                         nn.ReLU(), nn.Linear(hidden_dim, data_dim * 2)
                                                         ) for _ in range(N)])

    def forward(self, x, t: int):
        h = self.network_head(x)
        return self.network_tail[t](h)


class DiffusionModel(nn.Module):

    def __init__(self, core_model: nn.Module, n_steps, device: str):
        super().__init__()

        self.core_model = core_model
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
        mu, h = self.core_model(xt, t).chunk(2, dim=1)
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


def plot_data(model: torch.nn.Module, dataset_name: str, epoch: int, run_timestamp: str):
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
    plt.savefig(f"Imgs/diffusion_model_{dataset_name}_epoch_{epoch}_{run_timestamp}.png", bbox_inches='tight')
    plt.clf()


def save_checkpoint(dataset_name: str, core_model: torch.nn.Module, optimizer, epoch: int, train_time_sec: int,
                    loss_formula_str: str,
                    loss_window: int,
                    loss_avg: float, checkpoint_out_path_prefix: str, device: str):
    out_path = os.path.join(checkpoint_out_path_prefix, f"epoch_{epoch}.pt")
    checkpoint_dict = {'epoch': epoch, 'optimizer': str(optimizer), 'optimizer_state_dict': optimizer.state_dict(),
                       'model_state_dict': core_model.state_dict(), 'loss_avg': loss_avg,
                       'train_time_sec': train_time_sec,
                       'loss_formula_str': loss_formula_str, 'loss_window': loss_window,
                       'timestamp': datetime.now().isoformat(), 'device': device, 'dataset_name': dataset_name}
    torch.save(obj=checkpoint_dict, f=out_path)
    logger.info(f"Successfully saved checkpoint \n: {checkpoint_dict}\n to {out_path}")


def train(model, optimizer, last_epoch, last_train_time_point, n_epochs, batch_size, dataset_name, window,
          checkpoint_epoch_count, checkpoint_out_path_prefix, device: str):
    training_losses = []
    loss_formula_str = """KL = (torch.log(sigma) - torch.log(sigma_posterior) + (sigma_posterior ** 2 + 
                                (mu_posterior - mu) ** 2) / (2 * sigma ** 2) - 0.5)"""
    start_time = datetime.now()
    start_epoch = last_epoch + 1
    end_epoch = start_epoch + n_epochs  # exclusive not inclusive
    for i in tqdm(range(n_epochs)):
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
        # TODO Notes
        #   1. When setting momentum to 0.9 losses diverges to nan , try to understand why later
        #   2.
        optimizer.zero_grad()
        loss.backward()
        training_losses.append(loss.item())
        # checkpoint
        # save checkpoint either at the very start point or later on with the counter policy
        curr_epoch = last_epoch + 1 + i
        if curr_epoch == 1 or curr_epoch % checkpoint_epoch_count == 0:
            start = max(0, i - window)
            loss_avg = np.average(training_losses[start:(i + 1)])
            save_checkpoint(core_model=diffusion_model.core_model, optimizer=optimizer, epoch=curr_epoch,
                            train_time_sec=last_train_time_point + (datetime.now() - start_time).seconds,
                            loss_window=window, loss_formula_str=loss_formula_str, loss_avg=loss_avg,
                            checkpoint_out_path_prefix=checkpoint_out_path_prefix, device=device,
                            dataset_name=dataset_name)
        optimizer.step()
    per_run_training_time = (datetime.now() - start_time).seconds
    # remember end-epoch is exclusive not inclusive
    logger.info(
        f"Training finished from epoch {start_epoch} to {end_epoch - 1} (inclusive) in {per_run_training_time} seconds")
    return training_losses


def save_model(model: torch.nn.Module, dataset_name: str, nepochs: int, run_timestamp: str):
    assert hasattr(model, "name"), "model must have name for saving"
    torch.save(model.state_dict(), f"./models/{model.name}_{dataset_name}_nepochs_{nepochs}_{run_timestamp}.model")


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset-name', type=str, required=True,
                        choices=["swissroll", "circles", "blobs", "mvn"])
    parser.add_argument('--model', type=str, required=True, choices=["nn_head_tail"])
    parser.add_argument('--n-epochs', type=int, required=True)
    parser.add_argument('--start-checkpoint-file', type=str, required=False)
    parser.add_argument('--checkpoint-out-dir', type=str, required=False, default="checkpoints")
    parser.add_argument('--device', type=str, required=False, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument('--batch-size', type=int, required=False, default=64000)
    parser.add_argument('--loss-window', type=int, required=False, default=10000)
    parser.add_argument('--checkpoint-count', type=int, required=False, default=10000)
    parser.add_argument('--lr', type=float, required=False, default=1e-4)
    parser.add_argument('--momentum', type=float, required=False, default=0.9)
    parser.add_argument('--n-diffusion-steps', type=int, required=False, default=40)
    parser.add_argument('--hidden-dim', required=False, type=int, default=128)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logger.info(f"Parsing Args")
    args = get_args()
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "gpu":
        device = torch.device("cuda")
    else:
        raise ValueError(f"Device {args.device} is not supported")
    checkpoint_out_dir = os.path.join(args.checkpoint_out_dir, f"{args.model}_{args.dataset_name}")
    if not os.path.exists(checkpoint_out_dir):
        os.makedirs(checkpoint_out_dir)
    if args.start_checkpoint_file is None:
        logger.info("No checkpoint provided, training from scratch")
        last_epoch = 0
        last_train_time_point = 0
        if args.model == "nn_head_tail":
            core_model = MLP(hidden_dim=args.hidden_dim).to(device)
            optimizer = torch.optim.SGD(core_model.parameters(), lr=args.lr)
        else:
            raise ValueError(f"Model : {args.model}")
    else:
        logger.info(f"Loading from checkpoint {args.start_checkpoint_file}")
        checkpoint_dict = torch.load(args.start_checkpoint_file)
        logger.info(f"Dump of the loaded checkpoint \n:{checkpoint_dict}")
        last_epoch = checkpoint_dict['epoch']
        last_train_time_point = checkpoint_dict['train_time_sec']
        logger.info(f"Last epoch (loaded from checkpoint) ={last_epoch} , "
                    f"starting from epoch = {last_epoch + 1}")
        if args.model == "nn_head_tail":
            core_model = MLP(hidden_dim=args.hidden_dim).to(device)
            optimizer = torch.optim.SGD(core_model.parameters(), lr=args.lr, momentum=args.lr)
            core_model.load_state_dict(checkpoint_dict['model_state_dict'])
            optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
        else:
            raise ValueError(f"Model Arch : {args.model_arch}")

    diffusion_model = DiffusionModel(core_model=core_model, n_steps=args.n_diffusion_steps, device=args.device)

    # Train Model and save checkpoint inside
    training_losses = train(model=diffusion_model, optimizer=optimizer, n_epochs=args.n_epochs,
                            batch_size=args.batch_size,
                            dataset_name=args.dataset_name, window=args.loss_window,
                            checkpoint_epoch_count=args.checkpoint_count,
                            checkpoint_out_path_prefix=checkpoint_out_dir,
                            last_epoch=last_epoch,
                            last_train_time_point=last_train_time_point, device=args.device)

    # Test a Model
    # TODO separate testing into another script
    logger.info(f"Plotting generated and actual data")
    plot_data(model=diffusion_model, dataset_name=args.dataset_name, epoch=last_epoch + args.n_epochs,
              run_timestamp=run_timestamp)
    logger.info(f"Plotting train losses")
    plot_train_losses(training_losses=training_losses, dataset_name=args.dataset_name,
                      epoch=last_epoch + args.n_epochs, window=args.loss_window, run_timestamp=run_timestamp)
