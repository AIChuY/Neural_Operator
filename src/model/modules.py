"""The components of neural operator models."""
import os
from typing import List

import torch
from torch import nn


class FNN(nn.Module):
    """The fully connected neural network."""

    def __init__(
        self,
        hidden_layer: List[int] = [64, 64],
        dim_in: int = -1,
        dim_out: int = -1,
        activation: torch.nn.Module = None,
    ):
        """Initialize the fully connected neural network.

        Args:
        ----
            hidden_layer (list, optional): The hidden layer sizes. Defaults to [64, 64].
            dim_in (int, optional): The input dimension. Defaults to -1.
            dim_out (int, optional): The output dimension. Defaults to -1.
            activation (torch.nn.Module, optional): The activation function. Defaults to None.
        """
        super().__init__()
        self.sigma = activation
        dim = [dim_in] + hidden_layer + [dim_out]
        self.layers = nn.ModuleList([nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass.

        Args:
        ----
            x (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.
        """
        for i in range(len(self.layers) - 1):
            x = self.sigma(self.layers[i](x))
        # linear activation at the last layer
        return self.layers[-1](x)


class NeuralBasis(nn.Module):
    """The neural basis."""

    def __init__(
        self, dim_in: int = 1, hidden: List[int] = [4, 4, 4], n_base: int = 4, activation: torch.nn.Module = None
    ):
        """Initialize the neural basis.

        Args:
        ----
            dim_in (int, optional): The input dimension. Defaults to 1.
            hidden (list, optional): The hidden layer sizes. Defaults to [4, 4, 4].
            n_base (int, optional): The number of basis. Defaults to 4.
            activation (torch.nn.Module, optional): The activation function. Defaults to None.
        """
        super().__init__()
        self.sigma = activation
        dim = [dim_in] + hidden + [n_base]
        self.layers = nn.ModuleList([nn.Linear(dim[i - 1], dim[i]) for i in range(1, len(dim))])

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Define the forward pass.

        Args:
        ----
            t (torch.Tensor): The input tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.
        """
        for i in range(len(self.layers) - 1):
            t = self.sigma(self.layers[i](t))
        # linear activation at the last layer
        return self.layers[-1](t)


class Basic_Model(nn.Module):
    """The basic model."""

    def __init__(self):
        """Initialize the basic model."""
        super().__init__()
        pass

    def forward(self):
        """Define the forward pass."""
        pass

    def load_params_from_file(self, filename: str, optimizer: torch.optim.Optimizer = None, to_cpu: bool = False):
        """Load parameters from file.

        Args:
        ----
            filename (str): The file name.
            optimizer (torch.optim.Optimizer, optional): The optimizer. Defaults to None.
            to_cpu (bool, optional): Whether to load the parameters to CPU. Defaults to False.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError

        print("==> Loading parameters from checkpoint %s to %s" % (filename, "CPU" if to_cpu else "GPU"))
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint["model_state"]
        if optimizer:
            optimizer_state_disk = checkpoint["optimizer_state"]
            optimizer.load_state_dict(optimizer_state_disk)
            print("loaded optimizer")
        else:
            print("optimizer is not loaded")

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                print("Update weight %s: %s" % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                print("Not updated weight %s: %s" % (key, str(state_dict[key].shape)))

        print("==> Done (loaded %d/%d)" % (len(update_model_state), len(self.state_dict())))

    def freeze_basis(self, lr=1e-4, weight_decay=0):
        """Freeze the basis.

        Args:
        ----
            lr (float, optional): The learning rate. Defaults to 1e-4.
            weight_decay (int, optional): The weight decay. Defaults to 0.

        Returns:
        -------
            torch.optim.Optimizer: The optimizer.
        """
        total = 0
        freezed = 0
        for name, param in self.named_parameters():
            if name[:5] == "BL_in":
                param.requires_grad = False
                freezed += 1
                print("Freeze " + name)
            total += 1
        print("==> Freezed (loaded %d/%d)" % (freezed, total))
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay
        )
        return optimizer


def _inner_product(f1, f2, h):
    """Compute the inner product of f1 and f2.

    f1 - (B, J) : B functions, observed at J time points,
    f2 - (B, J) : same as f1
    h  - (J-1,1): weights used in the trapezoidal rule
    pay attention to dimension
    <f1, f2> = sum (h/2) (f1(t{j}) + f2(t{j+1}))
    """
    prod = f1 * f2  # (B, J = len(h) + 1)
    return torch.matmul((prod[:, :-1] + prod[:, 1:]), h.unsqueeze(dim=-1)) / 2


def _parralleled_inner_product(f1, f2, h):
    prod = f1 * f2
    return torch.matmul((prod[:, :, :-1] + prod[:, :, 1:]), h) / 2


def trapezoidal_2d_parralleled(f, h):
    """Compute the 2d trapezoidal rule for parralleled data.

    Args:
    ----
        f (torch.Tensor): The input tensor.
        h (list): The weights.

    Returns:
    -------
        torch.Tensor: The output tensor.
    """
    assert isinstance(h, list) and len(h) == 2  # noqa: PLR2004 # TODO: fix this after figuring out the meaning
    _, _, l1, l2 = f.size()
    # l1, l2 = f.shape[2], f.shape[3]
    c = torch.ones((l1, l2), device=f.device)
    c[[0, -1], :] = 1 / 2
    c[:, [0, -1]] = 1 / 2
    c[[0, 0, -1, -1], [0, -1, 0, -1]] = 1 / 4
    return h[0] * h[1] * torch.sum(torch.mul(c, f), dim=(-2, -1))


def _parralleled_inner_product_2d(f1, f2, h):
    prod = f1 * f2
    return trapezoidal_2d_parralleled(prod, h)


def trapezoidal_2d(f, h):
    """Compute the 2d trapezoidal rule.

    Args:
    ----
        f (torch.Tensor): The input tensor.
        h (list): The weights.

    Returns:
    -------
        torch.Tensor: The output tensor.
    """
    assert isinstance(h, list) and len(h) == 2  # noqa: PLR2004 # TODO: fix this after figuring out the meaning
    _, l1, l2 = f.size()
    # l1, l2 = f.shape[1], f.shape[2]
    c = torch.ones((l1, l2), device=f.device)
    c[[0, -1], :] = 1 / 2
    c[:, [0, -1]] = 1 / 2
    c[[0, 0, -1, -1], [0, -1, 0, -1]] = 1 / 4
    return h[0] * h[1] * torch.sum(torch.mul(c, f), dim=(-2, -1))


def _inner_product_2d(f1, f2, h):
    prod = f1 * f2
    return trapezoidal_2d(prod, h)
