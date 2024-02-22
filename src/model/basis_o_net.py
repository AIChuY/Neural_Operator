"""Model for basis operator."""
import numpy as np
import torch

from .modules import FNN, Basic_Model, NeuralBasis, _inner_product_2d, _parralleled_inner_product_2d


class BasisONet(Basic_Model):
    """Basis operator network."""

    # Just a placeholder for testing commit
    def __init__(  # noqa: PLR0913
        self,
        n_base_in=9,
        base_in_hidden=[64, 64, 64],
        middle_hidden=[64, 64, 64],
        n_base_out=9,
        base_out_hidden=[64, 64, 64],
        grid_in=None,
        grid_out=None,
        device=None,
        activation=None,
    ):
        """Initialize the basis operator network.

        Args:
        ----
            n_base_in (int, optional): The number of basis nodes in the input space. Defaults to 9.
            base_in_hidden (list, optional): The hidden layer sizes for the input basis. Defaults to [64, 64, 64].
            middle_hidden (list, optional): The hidden layer sizes for the middle layer. Defaults to [64, 64, 64].
            n_base_out (int, optional): The number of basis nodes in the output space. Defaults to 9.
            base_out_hidden (list, optional): The hidden layer sizes for the output basis. Defaults to [64, 64, 64].
            grid_in (np.ndarray, optional): The input grid. Defaults to None.
            grid_out (np.ndarray, optional): The output grid. Defaults to None.
            device (torch.device, optional): The device to use. Defaults to None.
            activation (torch.nn.Module, optional): The activation function. Defaults to None.

        """
        super().__init__()
        self.n_base_in = n_base_in
        self.n_base_out = n_base_out
        self.device = device
        self.t_in = torch.tensor(grid_in).to(device).float().reshape(-1, 3)
        self.t_out = torch.tensor(grid_out).to(device).float().reshape(-1, 3)
        self.h_in = torch.tensor(grid_in[1:] - grid_in[:-1]).to(device).float()
        self.BL_in = NeuralBasis(dim_in=3, hidden=base_in_hidden, n_base=n_base_in, activation=activation)
        self.Middle = FNN(
            hidden_layer=middle_hidden,
            dim_in=n_base_in,
            dim_out=n_base_out,
            activation=activation,
        )
        self.BL_out = NeuralBasis(dim_in=3, hidden=base_out_hidden, n_base=n_base_out, activation=activation)

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

    def check_orthogonality_in(self, J1, J2, path=None):
        """Check the orthogonality of the input basis.

        Args:
        ----
            J1 (int): The number of grids in the first dimension.
            J2 (int): The number of grids in the second dimension.
            path (str, optional): The path to save the orthogonality matrix. Defaults to None.

        Returns:
        -------
            np.ndarray: The orthogonality matrix.

        """
        T = self.t_in
        # evaluate the current basis nodes at time grid
        self.bases_in = self.BL_in(T)  # (J, n_base)
        orth_matrix = torch.ones((self.bases_in.shape[1], self.bases_in.shape[1])).to(self.device)
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = (
                    torch.einsum("s,s->", self.bases_in[:, i], self.bases_in[:, j]) / self.bases_in.shape[0]
                )
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix

    def check_orthogonality_out(self, J1, J2, path=None):
        """Check the orthogonality of the output basis.

        Args:
        ----
            J1 (int): The number of grids in the first dimension.
            J2 (int): The number of grids in the second dimension.
            path (str, optional): The path to save the orthogonality matrix. Defaults to None.

        Returns:
        -------
            np.ndarray: The orthogonality matrix.

        """
        T = self.t_out
        # evaluate the current basis nodes at time grid
        self.bases_out = self.BL_out(T)  # (J, n_base)
        orth_matrix = torch.ones((self.bases_out.shape[1], self.bases_out.shape[1])).to(self.device)
        for i in range(orth_matrix.shape[0]):
            for j in range(orth_matrix.shape[1]):
                orth_matrix[i, j] = (
                    torch.einsum("s,s->", self.bases_out[:, i], self.bases_out[:, j]) / self.bases_out.shape[0]
                )
        orth_matrix = orth_matrix.detach().cpu().numpy()
        if path:
            np.savetxt(path, orth_matrix)
        else:
            return orth_matrix
