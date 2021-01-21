import torch.nn as nn

DEFAULTS_ADULT = {
    "z_dim": 2,
    "max_epochs": 1,
    "batch_size": 64,
    "lambda_od": 0.037,
    "gamma_od": 0.8,
    "encoder_hidden_dims": [64],
    "lambda_entropy": 0.55,
    "gamma_entropy": 1.66,
    "input_dim": 108,
    "target_output_dim": 1,
    "sens_output_dim": 1,
    "target_disc_hidden_dims": [64, 64],
    "sens_disc_hidden_dims": [64, 64],
    "target_disc_batch_norm": False,
    "predictor_epochs": 10,
    # adam parameters
    "encoder_lr": 1e-3,
    "encoder_weight_decay": 5e-4,
    "discs_lr": 1e-3,
    "discs_weight_decay": 5e-4,
}
DEFAULTS_GERMAN = {
    "z_dim": 2,
    "max_epochs": 12,
    "batch_size": 64,
    "lambda_od": 0.01,
    "gamma_od": 1.4,
    "encoder_hidden_dims": [64],
    "lambda_entropy": 1,
    "gamma_entropy": 2,
    "input_dim": 61,
    "target_output_dim": 1,
    "sens_output_dim": 1,
    "target_disc_hidden_dims": [64, 64],
    "sens_disc_hidden_dims": [64, 64],
    "target_disc_batch_norm": False,
    "predictor_epochs": 10,
    # adam parameters
    "encoder_lr": 1e-3,
    "encoder_weight_decay": 5e-4,
    "discs_lr": 1e-3,
    "discs_weight_decay": 5e-4,
}
DEFAULTS_YALEB = {
    "z_dim": None,
    "max_epochs": None,
    "batch_size": None,
    "lambda_od": None,
    "gamma_od": None,
    "lambda_entropy": None,
    "gamma_entropy": None,
    "input_dim": None,
    "target_output_dim": 38,
    "target_disc_hidden_dims": [100, 100],
    "target_disc_batch_norm": True,
    "sens_output_dim": None,
    "sens_disc_hidden_dims": None,
    "predictor_epochs": None,
}  # {
#     "z_dim": 100,
#     "max_epochs": 20,
#     "target_output_dim": 65,
#     "target_disc_hidden_dims": [100, 100],
# }
DEFAULTS_CIFAR10 = {
    "z_dim": None,
    "max_epochs": None,
    "batch_size": None,
    "lambda_od": None,
    "gamma_od": None,
    "lambda_entropy": None,
    "gamma_entropy": None,
    "input_dim": None,
    "target_output_dim": None,
    "sens_output_dim": None,
    "target_disc_hidden_dims": None,
    "sens_disc_hidden_dims": None,
    "predictor_epochs": None,
    "target_disc_batch_norm": False,
}  # {
#     "z_dim": 128,
#     "max_epochs": 30,
#     "target_predictor_hidden_dims": [256, 128],
#     "target_predictor_output_dim": 1,
# }
DEFAULTS_CIFAR100 = {
    "z_dim": 128,
    "max_epochs": 30,
    "target_predictor_hidden_dims": [256, 128],
    "target_predictor_output_dim": 1,
}  # {"z_dim": 128, "max_epochs": 30}

DATASET2DEFAULTS = {
    "adult": DEFAULTS_ADULT,
    "german": DEFAULTS_GERMAN,
    "yaleb": DEFAULTS_YALEB,
    "cifar10": DEFAULTS_CIFAR10,
    "cifar100": DEFAULTS_CIFAR100,
}
