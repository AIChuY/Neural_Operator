import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import math 
import random

class FNODataset(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1
                 ):
        # Define path to files
        root_path = os.path.abspath(saved_folder + filename)
        assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'
        
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            if 'tensor' not in keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                
                self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                        idx_cfd[2]//reduced_resolution,
                                        idx_cfd[3]//reduced_resolution,
                                        math.ceil(idx_cfd[1]/reduced_resolution_t),
                                        4],
                                        dtype=np.float32)
                # density
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,0] = _data   # batch, x, t, ch
                # pressure
                _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,1] = _data   # batch, x, t, ch
                # Vx
                _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,2] = _data   # batch, x, t, ch
                # Vy
                _data = np.array(f['Vy'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data, (0, 2, 3, 1))
                self.data[...,3] = _data   # batch, x, t, ch

                x = np.array(f["x-coordinate"], dtype=np.float32)
                y = np.array(f["y-coordinate"], dtype=np.float32)
                x = torch.tensor(x, dtype=torch.float)
                y = torch.tensor(y, dtype=torch.float)
                X, Y = torch.meshgrid(x, y)
                self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
        
                                                                
            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                elif len(_data.shape) == 4:  
                    if "nu" in f.keys(): # 2D Darcy flow
                    # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch
                    else:
                        # label: (num_sample, t, x1, x2)
                        _data = _data[::reduced_batch, :, ::reduced_resolution, ::reduced_resolution]
                        ## convert to (num_sample, x1, ..., xd, t)
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = _data[:, :, :, :, None]

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y)
                    self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                else:
                    pass
        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        return self.data[idx,...,:self.initial_step,:], self.data[idx], self.grid

class BurgersDataset(Dataset):
    def __init__(self, filename,
                 initial_step=10,
                 saved_folder='../data/',
                 reduced_resolution=1,
                 reduced_resolution_t=1,
                 reduced_batch=1,
                 if_test=False,
                 test_ratio=0.1,
                 num_samples_max = -1
                 ):
        
        # Define path to files
        root_path = os.path.abspath(saved_folder + filename)
        assert filename[-2:] != 'h5', 'HDF5 data is assumed!!'
        
        with h5py.File(root_path, 'r') as f:
            keys = list(f.keys())
            keys.sort()
            if 'tensor' not in keys:
                _data = np.array(f['density'], dtype=np.float32)  # batch, time, x,...
                idx_cfd = _data.shape
                
                self.data = np.zeros([idx_cfd[0]//reduced_batch,
                                        idx_cfd[2]//reduced_resolution,
                                        math.ceil(idx_cfd[1]/reduced_resolution_t),
                                        3],
                                    dtype=np.float32)
                #density
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data[:, :, :], (0, 2, 1))
                self.data[...,0] = _data   # batch, x, t, ch
                # pressure
                _data = np.array(f['pressure'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data[:, :, :], (0, 2, 1))
                self.data[...,1] = _data   # batch, x, t, ch
                # Vx
                _data = np.array(f['Vx'], dtype=np.float32)  # batch, time, x,...
                _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                ## convert to [x1, ..., xd, t, v]
                _data = np.transpose(_data[:, :, :], (0, 2, 1))
                self.data[...,2] = _data   # batch, x, t, ch

                self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                print(self.data.shape)
            else:  # scalar equations
                ## data dim = [t, x1, ..., xd, v]
                _data = np.array(f['tensor'], dtype=np.float32)  # batch, time, x,...
                if len(_data.shape) == 3:  # 1D
                    _data = _data[::reduced_batch,::reduced_resolution_t,::reduced_resolution]
                    ## convert to [x1, ..., xd, t, v]
                    _data = np.transpose(_data[:, :, :], (0, 2, 1))
                    self.data = _data[:, :, :, None]  # batch, x, t, ch

                    self.grid = np.array(f["x-coordinate"], dtype=np.float32)
                    self.grid = torch.tensor(self.grid[::reduced_resolution], dtype=torch.float).unsqueeze(-1)
                elif len(_data.shape) == 4:  
                    if "nu" in f.keys(): # 2D Darcy flow
                    # u: label
                        _data = _data[::reduced_batch,:,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        #if _data.shape[-1]==1:  # if nt==1
                        #    _data = np.tile(_data, (1, 1, 1, 2))
                        self.data = _data
                        # nu: input
                        _data = np.array(f['nu'], dtype=np.float32)  # batch, time, x,...
                        _data = _data[::reduced_batch, None,::reduced_resolution,::reduced_resolution]
                        ## convert to [x1, ..., xd, t, v]
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = np.concatenate([_data, self.data], axis=-1)
                        self.data = self.data[:, :, :, :, None]  # batch, x, y, t, ch
                    else:
                        # label: (num_sample, t, x1, x2)
                        _data = _data[::reduced_batch, :, ::reduced_resolution, ::reduced_resolution]
                        ## convert to (num_sample, x1, ..., xd, t)
                        _data = np.transpose(_data[:, :, :, :], (0, 2, 3, 1))
                        self.data = _data[:, :, :, :, None]

                    x = np.array(f["x-coordinate"], dtype=np.float32)
                    y = np.array(f["y-coordinate"], dtype=np.float32)
                    x = torch.tensor(x, dtype=torch.float)
                    y = torch.tensor(y, dtype=torch.float)
                    X, Y = torch.meshgrid(x, y)
                    self.grid = torch.stack((X, Y), axis=-1)[::reduced_resolution, ::reduced_resolution]
                else:
                    pass

        if num_samples_max>0:
            num_samples_max  = min(num_samples_max,self.data.shape[0])
        else:
            num_samples_max = self.data.shape[0]

        test_idx = int(num_samples_max * test_ratio)
        if if_test:
            self.data = self.data[:test_idx]
        else:
            self.data = self.data[test_idx:num_samples_max]

        # Time steps used as initial conditions
        self.initial_step = initial_step

        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        return self.data[idx,...,:self.initial_step,:], self.data[idx,...,-1,:], self.grid



def setup_seed(seed):
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed_all(seed)  # GPU
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn