import logging
import gc
import os

import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
if __name__ == '__main__':
    time_steps = 1000
    device = torch.device('cuda')
    image_size = 64
    num_images = 8
    num_channels = 3
    """
    Note
    =====
    In the original code , I have tried to use cuda and had the following error
    "RuntimeError: CUDA error: out of memory"?
    I have tried
        1. using gc collect and torch empty cache
        2. os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    But none of them worked
    What worked is to reduce the image size passed for data and mode from 128 to 64
        ...
        device = torch.device('cuda')
        image_size=64
        diffusion = GaussianDiffusion( model,
            image_size=image_size,
            timesteps=time_steps  # number of steps
            ).to(device)
        training_images = torch.rand(num_images, num_channels, image_size, image_size).to(device)  
        # images are normalized from 0 to 1
    This is the snippet that works.
    """
    #
    # Test if cuda is available
    logger.info(f'Is cuda available ? : {torch.cuda.is_available()}')
    logger.info(f'Cuda device count = {torch.cuda.device_count()}')
    # https://github.com/mbaddar1/denoising-diffusion-pytorch?tab=readme-ov-file#usage

    # Trying to handle the memory issue
    #   https://discuss.pytorch.org/t/memory-management-using-pytorch-cuda-alloc-conf/157850/4
    # FIXME None worked yet
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f'Creating Unet model:')

    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    ).to(device)

    # Double-checking if models are actually on cuda
    #   https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/2
    is_core_model_on_cuda = next(model.parameters()).is_cuda
    logger.info(f'If core model is on cuda ? : {is_core_model_on_cuda}')

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,
        timesteps=time_steps  # number of steps
    ).to(device)

    is_diffusion_model_on_cuda = next(diffusion.parameters()).is_cuda
    logger.info(f'Is diffusion model on cuda? : {is_diffusion_model_on_cuda}')

    training_images = torch.rand(num_images, num_channels, image_size, image_size).to(device)
    # images are normalized from 0 to 1
    loss = diffusion(training_images)
    loss.backward()
    # after a lot of training

    sampled_images = diffusion.sample(batch_size=4)
    print(sampled_images.shape)  # (4, 3, 128, 128)
    print("finished")