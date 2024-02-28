# CascadeTuner
Implements a minimalistic version of Stable Cascade training

# Installation Notes:

Install Anaconda or Miniconda.

At the terminal:
`conda env create -f environment.yaml`
Wait a bit, as it takes time to install.
`conda activate CTuner`
This will activate the virtual environment for CascadeTuner.

If bitsandbytes doesn't start under Windows 10/11:

pip uninstall bitsandbytes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl

# Launching your script:
`accelerate launch --mixed_precision="bf16" train_stage_c.py --yaml "configs/your_yaml_here.yaml"`

If it doesn't launch:
`conda install -c defaults intel-openmp -f` usually fixes PIL issues under Windows 10/11.