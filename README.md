# GenModel
PhD Code and Experiments File
____

# Experimentation Logging Steps
___
1) Add the following files for each experiment 
    1.1) Python code files (*.py)

    1.2) Model files (.model)
    
    1.3) Log file (.log)
    
    1.4) Output Image file (.png)
    
    1.5) Loss Curve File (.png)

Each expeirment has an id which is the run timestamp 

Example for a set of experiment files ( git status output)
 	
 	new file:   sandbox/diffmodel/Imgs/diffusion_model_swissroll_nepochs_500000_2023-12-07T13:26:00.750259.png
    
    modified:   sandbox/diffmodel/diffusion_models.py
    
    new file:   sandbox/diffmodel/logs/diff_model_2023-12-07T13:26:00.750259.log
    
    new file:   sandbox/diffmodel/loss_curve/loss_curve_swissroll_nepochs_500000_2023-12-07T13:26:00.750259.png
    
    new file:   sandbox/diffmodel/models/nn_head_tail_swissroll_nepochs_500000_2023-12-07T13:26:00.750259.model

Another experiment 'git add' command

	git add 

	sandbox/diffmodel/diffusion_models.py 

	sandbox/diffmodel/Imgs/diffusion_model_mvn_nepochs_10000_2023-12-07T20\:07\:02.643081.png 

	sandbox/diffmodel/logs/diff_model_2023-12-07T20\:07\:02.643081.log 

	sandbox/diffmodel/models/nn_head_tail_mvn_nepochs_10000_2023-12-07T20\:07\:02.643081.model 

	sandbox/diffmodel/loss_curve/loss_curve_mvn_nepochs_10000_2023-12-07T20\:07\:02.643081.png 


======
# Configuring the score_sde_pytorch repo

1. This config and setup is tested using python3.9

2. Install using requirements.txt , not requirements_old.txt. The old one is the one provided with original repo, but didnot work

3. If faced the problem of "fatal error: Python.h: No such file or directory compilation terminated." :

```
sudo apt-get install python3.9-dev
```

In general , it should be ``` sudo apt-get install python3.x-dev```. Watchout out for the minor version part of python 3.x

ref : https://stackoverflow.com/a/21530768 

4. Run main.pý with no args for testing

The expected output should be 


```
2023-12-12 11:52:03.802868: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
2023-12-12 11:52:03.802921: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
2023-12-12 11:52:03.804851: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2023-12-12 11:52:03.817102: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-12-12 11:52:05.478074: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
FATAL Flags parsing error:
  flag --workdir=None: Flag --workdir must have a value other than None.
  flag --config=None: Flag --config must have a value other than None.
  flag --mode=None: Flag --mode must have a value other than None.
Pass --helpshort or --helpfull to see help on flags.
```