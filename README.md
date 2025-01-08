# FNO

## data
The data should be put in the dataset subfolder. We use data provided by the pdebench, you can download data from https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986.

The data that we use to train FNO are:
1D_Burgers_Sols_Nu0.1.hdf5
2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5
2D_DarcyFlow_beta1.0_Train.hdf5

since the data files are too large(50GB), we only provide Darcyflow data


## Config file

The `config` directory contains many `yaml` config files with naming format `config_{1/2/3}D_{PDE name}.yaml` where all args for training and testing are saved. The explanations of some FNO specific args are as follows:

* training args:
    * `training_type`: string, set 'autoregressive' for autoregressive trainging using autoregressive loss or 'single' for single step training using single step loss.
    * `initial_step`: int, the number of input time steps. (default: 10)

* model args:
    * `num_channels`: int, the number of input channels and output channels that equals to the number of variables to be solved. For example, there are 3 variables to be solved for 1D Compressible NS equation: density, pressure and velocity.
    * `modes`: int, number of Fourier modes to multiply.
    * `width`: int, number of channels for the Fourier layer.

## Train

1. Check the following args in the config file:
    1. The path of `file_name` and `saved_folder` are correct;
    2. `if_training` is `True`;
2. Set hyperparameters for training, such as `lr`, `batch size`, etc. 
3. Run command:
```bash
python train.py ./config/train/${config file name}
# Example: python train.py ./config/train/config_1D_Advection.yaml
```
4. Setting: to Train Burgers Equation, Please use BurgersDataLoader, set spatial_dim=1 in get_model function 
to Train DarcyFlow or Navier-Stokes equation, Please use FNODataLoader, set spatial_dim=1 in get_model function


## Resume training

1. Modify config file:
    1. Make sure `if_training` is `True`;
    2. Set `continue_training` to `True`;
    3. Set `model_path` to the checkpoint path where traing restart;
2. Run command:
```bash
python train.py ./config/${config file name}
```

## Test
To test the model, find the corredponding yaml file and 
    1. Set `if_training` to `False`;
    2. Set `model_path` to the checkpoint path where the model to be evaluated is saved.

then run the same command as train