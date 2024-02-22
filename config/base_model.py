"""Basic setting for model structure."""
CONFIG = {
    "nbasis_in": 9,
    "nbasis_out": 9,
    "base_in_hidden": [512, 512, 512, 512, 512],
    "base_out_hidden": [512, 512, 512, 512, 512],
    "middle_hidden": [512, 512, 512],
    "model_name": "BasisONet",
    "activation": None,
}
