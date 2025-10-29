# Source code for AISTATS-2026 submission "Can Training Dynamics of Scale-Invariant Neural Networks Be Explained by the Thermodynamics of an Ideal Gas?"

## Requirements

Our code relies on the following libraries:
`torch`, `torchvision`, `numpy`, `scipy`, `sklearn`,
`tqdm`, `matplotlib`, `seaborn`

## Run training 

To run the training of the neural networks
```bash
    python run.py
```
A detailed description of the training configuration is provided in `run.py`

## Draw plots
Jupyter notebooks for drawing the plots from the paper are provided in the folder
`notebooks`.

## Isotropic noise model
The training and plots of the isotropic noise model
is provided in `notebooks/VMF.ipynb`
