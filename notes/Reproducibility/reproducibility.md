
## Reproducibility

ML experiments may be very hard to reproduce. You have a lot of hyperparameters, different dataset splits, different ways to preprocess your data, bugs, etc.
Ideally, you should log data split (already preprocessed), all hyperparameters (including learning rate scheduling), the initial state of your model and optimizer, random seeds used for initialization, dataset shuffling and all of your code. Your GPU is also should be in deterministic mode (which is not the default mode). **For every single model run**. This is a very hard task. Different random seed can significantly [change your metrics](https://arxiv.org/abs/2002.06305) and even GPU-induced randomness can be important. We're not solving all of these problems, but we need to address at least what we can handle.

For every result you report in the paper you need (at least) to:
1. Track your model and optimizer hyperparameters (including learning rate schedule)
1. Save final model parameters
1. Report all of the parameters in the paper (make a table in the appendix) and release the code
1. [Set random seeds](https://github.com/catalyst-team/catalyst/blob/f909e5b44eb4c2c26039e201bcbe67001529a515/catalyst/utils/seed.py)
(it is not as easy as `torch.manual_seed(42)`, follow the link).
1. Store everything in the cloud

To save your hyperparameters you can use the TensorBoard HParams plugin, but we recommend using a specialized service like [wandb.ai](https://app.wandb.ai). These services not only store all of your logs but provide an easy interface to store hyperparameters, code and model files.

Ideally, also:
1. Save the exact code you used (create a tag in your repository for each run)
1. Save your preprocessed data,
especially if you are working on a dataset paper
([Data Version Control](https://dvc.org/) helps)
1. Save your model and optimizer initialization (the state at step 0)
1. Use GPU in deterministic mode (this will slightly affect the performance)
1. Store everything in the cloud **and** locally


An easy way to do this:
```python
# Before the training:
import random
import wandb
import torch
import numpy as np

random.seed(args.seed)     # python random generator
np.random.seed(args.seed)  # numpy random generator

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

wandb.init(..., config=args)  # keep all your hyperparameters in args
# wandb also saves your code files and git repository automatically

checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'settings': args,
    'epoch': epoch,
    'step': global_step
}

torch.save(checkpoint, save_path_for_init)
wandb.save(save_path_for_init)  # upload your initialization to wandb

# Your training loop is here

# After the training:
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'settings': args,
    'epoch': epoch,
    'step': global_step
}

torch.save(checkpoint, save_path)
wandb.save(save_path)  # upload your trained model to wandb
```

### TL;DR

At least keep all your hyperparameters for every run.
Use specialized tools like
[wandb.ai](https://app.wandb.ai) or
[tensorboard.dev](https://tensorboard.dev/) + TensorBoard HParams for this.
Store them in the cloud, not on your machine.


### Additional readings
  * [PyTorch docs on reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
  * [Fully reproducible research paper example](https://github.com/ibab/fully-reproducible)
  * [Independently Reproducible Machine Learning](https://thegradient.pub/independently-reproducible-machine-learning)
