"""Data configuration for Darcy flow problem."""
CONFIG = {
    "dataset": {
        "file_dir": "../datafiles/darcy_flow/",
        "sub": 1,
        "ntrain": 1000,
        "nvalid": 500,
        "ntest": 500,
        "batch_size": 32,
    }
}
