"""The loss functions for training the model."""
import torch


class LpLoss(object):
    """The Lp loss function."""

    def __init__(self, p: int = 2, size_average: bool = True, reduction: bool = True):
        """Initialize the Lp loss function.

        Args:
        ----
            p (int, optional): The p in Lp loss. Defaults to 2.
            size_average (bool, optional): Whether to average the loss. Defaults to True.
            reduction (bool, optional): Whether to reduce the loss. Defaults to True.

        """
        super(LpLoss, self).__init__()
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the relative Lp loss.

        Args:
        ----
            x (torch.Tensor): The predicted value.
            y (torch.Tensor): The ground truth value.

        Returns:
        -------
            torch.Tensor: The relative Lp loss.

        """
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Define the call method."""
        return self.rel(x, y)
