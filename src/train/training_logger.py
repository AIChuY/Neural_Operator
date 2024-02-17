"""Logger for training process."""
import logging
from typing import List


class TrainingLogger:
    """Logger for training process."""

    def __init__(self, log_file: str = "training.log"):
        """Initialize the logger.

        Args:
        ----
            log_file (string, optional): The file to log to.

        """
        super().__init__()

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Create a file handler
        self.file_handler = logging.FileHandler(log_file)

        # Create a stream handler (prints to console)
        self.stream_handler = logging.StreamHandler()

        # Create a formatter
        self.formatter = logging.Formatter("%(asctime)s - %(message)s")

        # Set the formatter for both handlers
        self.file_handler.setFormatter(self.formatter)
        self.stream_handler.setFormatter(self.formatter)

        # Add the handlers to the logger
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.stream_handler)

    def log_train(self, epoch: int, losses: List[float]) -> None:
        """Log the training loss.

        Args:
        ----
            epoch (int): The current epoch.
            losses (List[float]): The training L2 losses in the order of total loss, operator loss, encoder loss
                and decoder loss.

        """
        self.logger.debug(
            f"ep:{epoch} | "
            f"train_l2_total:{losses[0]:.6f} | "
            f"train_l2_op:{losses[1]:.6f} | "
            f"train_l2_aec_in:{losses[2]:.6f} | "
            f"train_l2_aec_out:{losses[3]:.6f}"
        )

    def log_val(self, epoch: int, losses: List[float]) -> None:
        """Log the validation loss.

        Args:
        ----
            epoch (int): The current epoch.
            losses (List[float]): The validation L2 losses in the order of total loss, operator loss, encoder loss
                and decoder loss.

        """
        self.logger.debug(
            f"ep:{epoch} | "
            f"valid_l2_total:{losses[0]:.6f} | "
            f"valid_l2_op:{losses[1]:.6f} | "
            f"valid_l2_aec_in:{losses[2]:.6f} | "
            f"valid_l2_aec_out:{losses[3]:.6f}"
        )

    def log_test(self, losses: List[float]) -> None:
        """Log the test loss.

        Args:
        ----
            losses (List[float]): The test L2 losses in the order of total loss, operator loss, encoder loss
                and decoder loss.

        """
        self.logger.debug(
            f"test_l2_total:{losses[0]:.6f} | "
            f"test_l2_op:{losses[1]:.6f} | "
            f"test_l2_aec_in:{losses[2]:.6f} | "
            f"test_l2_aec_out:{losses[3]:.6f}"
        )
