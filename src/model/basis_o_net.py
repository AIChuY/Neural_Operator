"""Model for basis operator."""
import numpy as np
import torch
from sklearn.neighbors import KernelDensity  # type: ignore

from .modules import (
    FNN,
    Basic_Model,
    NeuralBasis,
    _inner_product,
    _inner_product_2d,
    _parralleled_inner_product,
    _parralleled_inner_product_2d,
)


class BasisONet:
    """Factory for basis operator network."""

    def __init__(self, config):
        """Initialize the basis operator network.

        Args:
        ----
            config (dict): The configuration dictionary.

        """
        if config["model_name"] == "NavierStokes":
            self.model = NavierStokes(config)
        elif config["model_name"] == "KDVBurgers":
            self.model = KDVBurgers(config)
        elif config["model_name"] == "Poisson":
            self.model = Poisson(config)
        elif config["model_name"] == "DarcyFlow":
            self.model = DarcyFlow(config)
        else:
            raise ValueError(f"Model {config['model_name']} not found.")


class NavierStokes(Basic_Model):
    """Basis operator network for Navier Stokes problem."""

    def __init__(self, config):
        """Initialize the basis operator network.

        Args:
        ----
            config (dict): The configuration dictionary.

        """
        super().__init__()
        self.n_base_in = config["nbasis_in"]
        self.n_base_out = config["nbasis_out"]
        self.device = config["device"]
        self.t_in = torch.tensor(config["grid_in"]).to(self.device).float().reshape(-1, 3)
        self.t_out = torch.tensor(config["grid_out"]).to(self.device).float().reshape(-1, 3)
        self.h_in = torch.tensor(config["grid_in"][1:] - config["grid_out"][:-1]).to(self.device).float()
        self.BL_in = NeuralBasis(
            dim_in=3, hidden=config["base_in_hidden"], n_base=self.n_base_in, activation=config["activation"]
        )
        self.Middle = FNN(
            hidden_layer=config["middle_hidden"],
            dim_in=self.n_base_in,
            dim_out=self.n_base_out,
            activation=config["activation"],
        )
        self.BL_out = NeuralBasis(
            dim_in=3, hidden=config["base_out_hidden"], n_base=self.n_base_out, activation=config["activation"]
        )

    def forward(self, x, y):
        """Define the forward pass.

        Args:
        ----
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The output tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        B_in, J1_in, J2_in, J3_in = x.size()
        x = x.reshape(B_in, -1)
        B_out, J1_out, J2_out, J3_out = y.size()
        y = y.reshape(B_out, -1)
        T_in, T_out = self.t_in, self.t_out
        self.bases_in = self.BL_in(T_in)  # (J1_in*J2_in*J3_in, n_base_in)
        self.bases_out = self.BL_out(T_out)  # (J1_out*J2_out*J3_out, n_base_out)
        score_in = torch.einsum("bs,sn->bn", x, self.bases_in) / self.bases_in.shape[0]  # (B, n_base_in)
        score_out = self.Middle(score_in)  # (B, n_basis_out)
        out = torch.einsum("bn,sn->bs", score_out, self.bases_out)  # (B, J1_out*J2_out*J3_out)
        autoencoder_in = torch.einsum("bn,sn->bs", score_in, self.bases_in)
        score_out_temp = torch.einsum("bs,sn->bn", y, self.bases_out) / self.bases_out.shape[0]
        autoencoder_out = torch.einsum("bn,sn->bs", score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out


class KDVBurgers(Basic_Model):
    """Basis operator network for KDV Burgers problem."""

    def __init__(self, config):
        """Initialize the basis operator network.

        Args:
        ----
            config (dict): The configuration dictionary.

        """
        super().__init__()
        self.n_base_in = config["nbasis_in"]
        self.n_base_out = config["nbasis_out"]
        self.device = config["device"]
        self.t_in = torch.tensor(config["grid_in"]).to(self.device).float()
        self.t_out = torch.tensor(config["grid_out"]).to(self.device).float().reshape(-1, 2)
        self.h_in = torch.tensor(config["grid_in"][1:] - config["grid_in"][:-1]).to(self.device).float()
        self.h_out = [
            config["grid_out"][0, 1, 0] - config["grid_out"][0, 0, 0],
            config["grid_out"][1, 0, 1] - config["grid_out"][0, 0, 1],
        ]
        self.BL_in = NeuralBasis(
            dim_in=1, hidden=config["base_in_hidden"], n_base=self.n_base_in, activation=config["activation"]
        )
        self.Middle = FNN(
            hidden_layer=config["middle_hidden"],
            dim_in=self.n_base_in,
            dim_out=self.n_base_out,
            activation=config["activation"],
        )
        self.BL_out = NeuralBasis(
            dim_in=2, hidden=config["base_out_hidden"], n_base=self.n_base_out, activation=config["activation"]
        )

    def forward(self, x, y):
        """Define the forward pass.

        Args:
        ----
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The output tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        B_in, J_in = x.size()
        B_out, J1_out, J2_out = y.size()
        T_in, T_out = self.t_in.unsqueeze(dim=-1), self.t_out
        self.bases_in = self.BL_in(T_in)  # (J_in, n_base_in)
        self.bases_out = self.BL_out(T_out)  # (J1_out*J2_out, n_base_out)
        self.bases_in = self.bases_in.transpose(-1, -2)  # (n_base_in, J_in)
        self.bases_out = self.bases_out.transpose(-1, -2)  # (n_base_out, J1_out*J2_out)
        score_in = _parralleled_inner_product(
            x.unsqueeze(1).repeat((1, self.n_base_in, 1)),
            self.bases_in.unsqueeze(0).repeat((B_in, 1, 1)),
            self.h_in,
        )  # (B_in, n_base_in)
        score_out = self.Middle(score_in)
        out = torch.mm(score_out, self.bases_out)
        autoencoder_in = torch.mm(score_in, self.bases_in)
        score_out_temp = _parralleled_inner_product_2d(
            y.unsqueeze(1).repeat((1, self.n_base_out, 1, 1)),
            self.bases_out.unsqueeze(0).repeat((B_out, 1, 1)).reshape(B_out, self.n_base_out, J1_out, J2_out),
            self.h_out,
        )  # (B_out, n_base_out)
        autoencoder_out = torch.mm(score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out


class Poisson(Basic_Model):
    """Basis operator network for Poisson problem."""

    def __init__(self, config):
        """Initialize the basis operator network.

        Args:
        ----
            config (dict): The configuration dictionary.

        """
        super().__init__()
        self.n_base_in = config["nbasis_in"]
        self.n_base_out = config["nbasis_out"]
        self.device = config["device"]
        self.t_in = torch.tensor(config["grid_in"]).to(self.device).float().reshape(-1, 2)
        self.t_out = torch.tensor(config["grid_out"]).to(self.device).float().reshape(-1, 2)
        self.h_in = torch.tensor(config["grid_in"][1:] - config["grid_out"][:-1]).to(self.device).float()
        self.BL_in = NeuralBasis(
            dim_in=2, hidden=config["base_in_hidden"], n_base=self.n_base_in, activation=config["activation"]
        )
        self.Middle = FNN(
            hidden_layer=config["middle_hidden"],
            dim_in=self.n_base_in,
            dim_out=self.n_base_out,
            activation=config["activation"],
        )
        self.BL_out = NeuralBasis(
            dim_in=2, hidden=config["base_out_hidden"], n_base=self.n_base_out, activation=config["activation"]
        )
        self.kde_in = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(config["grid_in"])
        self.density_in = np.exp(self.kde_in.score_samples(config["grid_in"]))
        self.density_in = torch.tensor(self.density_in).to(config["device"]).float().reshape(-1, 1)
        self.kde_out = KernelDensity(kernel="gaussian", bandwidth=0.02).fit(config["grid_out"])
        self.density_out = np.exp(self.kde_out.score_samples(config["grid_out"]))
        self.density_out = torch.tensor(self.density_out).to(config["device"]).float().reshape(-1, 1)

    def forward(self, x, y):
        """Define the forward pass.

        Args:
        ----
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The output tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        B_in, J1_in = x.size()
        x = x.reshape(B_in, -1)
        B_out, J1_out = y.size()
        y = y.reshape(B_out, -1)
        T_in, T_out = self.t_in, self.t_out
        density_in = self.density_in
        density_in = density_in.squeeze(1)
        density_out = self.density_out
        density_out = density_out.squeeze(1)
        self.bases_in = self.BL_in(T_in)  # (J1_in*J2_out, n_base_in)
        self.bases_out = self.BL_out(T_out)  # (J1_out*J2_out, n_base_out)
        score_in = torch.einsum("bs,sn->bn", x / density_in, self.bases_in) / self.bases_in.shape[0]  # (B, n_base_in)
        score_out = self.Middle(score_in)  # (B, n_basis_out)
        out = torch.einsum("bn,sn->bs", score_out, self.bases_out)  # (B, J1_out*J2_out*J3_out)
        autoencoder_in = torch.einsum("bn,sn->bs", score_in, self.bases_in)
        # density_out = self.Density_net_out(self.density_out)
        # y_tmp = y / self.density_out.squeeze(1)
        score_out_temp = torch.einsum("bs,sn->bn", y / density_out, self.bases_out) / self.bases_out.shape[0]
        autoencoder_out = torch.einsum("bn,sn->bs", score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out


class DarcyFlow(Basic_Model):
    """Basis operator network for Navier Stokes problem."""

    def __init__(self, config):
        """Initialize the basis operator network.

        Args:
        ----
            config (dict): The configuration dictionary.

        """
        super().__init__()
        self.n_base_in = config["nbasis_in"]
        self.n_base_out = config["nbasis_out"]
        self.device = config["device"]
        self.t_in = torch.tensor(config["grid_in"]).to(self.device).float().reshape(-1, 2)
        self.t_out = torch.tensor(config["grid_out"]).to(self.device).float().reshape(-1, 2)
        self.h_in = torch.tensor(config["grid_in"][1:] - config["grid_out"][:-1]).to(self.device).float()
        self.h_in = [
            config["grid_in"][0, 1, 0] - config["grid_in"][0, 0, 0],
            config["grid_in"][1, 0, 1] - config["grid_in"][0, 0, 1],
        ]
        self.h_out = [
            config["grid_out"][0, 1, 0] - config["grid_out"][0, 0, 0],
            config["grid_out"][1, 0, 1] - config["grid_out"][0, 0, 1],
        ]
        assert self.h_in[0] > 0 and self.h_in[1] > 0 and self.h_out[0] > 0 and self.h_out[1] > 0
        self.BL_in = NeuralBasis(
            dim_in=2, hidden=config["base_in_hidden"], n_base=self.n_base_in, activation=config["activation"]
        )
        self.Middle = FNN(
            hidden_layer=config["middle_hidden"],
            dim_in=self.n_base_in,
            dim_out=self.n_base_out,
            activation=config["activation"],
        )
        self.BL_out = NeuralBasis(
            dim_in=2, hidden=config["base_out_hidden"], n_base=self.n_base_out, activation=config["activation"]
        )

    def forward(self, x, y):
        """Define the forward pass.

        Args:
        ----
            x (torch.Tensor): The input tensor.
            y (torch.Tensor): The output tensor.

        Returns:
        -------
            torch.Tensor: The output tensor.

        """
        B_in, J1_in, J2_in = x.size()
        B_out, J1_out, J2_out = y.size()
        T_in, T_out = self.t_in, self.t_out
        self.bases_in = self.BL_in(T_in)  # (J_in, n_base_in)
        self.bases_out = self.BL_out(T_out)  # (J1_out*J2_out, n_base_out)
        self.bases_in = self.bases_in.transpose(-1, -2)  # (n_base_in, J_in)
        self.bases_out = self.bases_out.transpose(-1, -2)  # (n_base_out, J1_out*J2_out)
        score_in = _parralleled_inner_product_2d(
            x.unsqueeze(1).repeat((1, self.n_base_in, 1, 1)),
            self.bases_in.unsqueeze(0).repeat((B_in, 1, 1)).reshape(B_in, self.n_base_in, J1_in, J2_in),
            self.h_in,
        )  # (B_in, n_base_in)
        score_out = self.Middle(score_in)
        out = torch.mm(score_out, self.bases_out)
        autoencoder_in = torch.mm(score_in, self.bases_in)
        score_out_temp = _parralleled_inner_product_2d(
            y.unsqueeze(1).repeat((1, self.n_base_out, 1, 1)),
            self.bases_out.unsqueeze(0).repeat((B_out, 1, 1)).reshape(B_out, self.n_base_out, J1_out, J2_out),
            self.h_out,
        )  # (B_out, n_base_out)
        autoencoder_out = torch.mm(score_out_temp, self.bases_out)
        return out, autoencoder_in, autoencoder_out
