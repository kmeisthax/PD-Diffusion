# PDDiffusion

## Setup

After cloning this repository, `cd` into it and run the following commands.

```
mkdir output
mkdir sets
python3 -m venv env
echo "../../../../" > env/lib/python3.10/site-packages/PDDiffusion.pth
```

`python3` may just be `python` on some systems; the `python3.10` path component will differ if your system has a newer or older version of Python when it made the virtual environment.

## Downloading datasets

### Smithsonian Open Access

TODO

### Wikimedia Commons

```
source env/bin/activate
python -m PDDiffusion.datasets.WikimediaCommons
```

This currently uses the public API and is limited to 200 image downloads at one time. Support for working with a local Mediawiki install or Wikimedia Commons data dump is in progress.

TODO: Parameters for adding your e-mail address to public Mediawiki requests

## Training

```
source env/bin/activate
python -m PDDiffusion.train
```

This pulls from any datasets you downloaded. Trained models will be stored in the output directory.

The model is set to save every training epoch; and reload from disk if the training process is restarted.

# Known Issues

## No GPU usage

PyTorch does not ship with GPU support by default on Windows. The `requirements.txt` file does not install the CUDA- or ROCm-aware versions of PyTorch.

Note: Since all our examples recommend the use of a virtual environment, you still need to do this even if you've already installed GPU-aware PyTorch for Jupyter notebooks.

To get a GPU-aware PyTorch, first uninstall PyTorch and then install one of the GPU-aware versions instead:

```
source env/bin/activate
pip uninstall torch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117 ;for CUDA on Windows
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm5.2/ ;for ROCm on Linux
```

For up-to-date index URLs see https://pytorch.org/get-started/locally/

Once a GPU-aware PyTorch has been installed, you should be able to configure it with `accelerate config` in your environment.

## Module 'signal' has no attribute 'SIGKILL'

You're on Windows. PyTorch ships with code that assumes Unix signals are available in defaults.

Manually corehack the `file_based_local_timer.py` file to use SIGTERM by default instead.