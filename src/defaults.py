DEFAULTS_ADULT = {
    "z_dim": 2,
    "max_epochs": 2,
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
    "loss_components": "entropy,kl,orth",
    "step_size": 30,
}
DEFAULTS_GERMAN = {
    "z_dim": 2,
    "max_epochs": 15,
    "batch_size": 64,
    "lambda_od": 0.01,
    "gamma_od": 1.4,
    "lambda_entropy": 1.0,
    "gamma_entropy": 2.0,
    "encoder_hidden_dims": [64],
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
    "loss_components": "entropy,kl,orth",
    "step_size": 30,
}
DEFAULTS_YALEB = {
    "z_dim": 100,
    "max_epochs": 30,
    "batch_size": 64,
    "lambda_od": 0.037,
    "gamma_od": 1.1,
    "encoder_hidden_dims": [],
    "lambda_entropy": 1,
    "gamma_entropy": 2,
    "input_dim": 32256,
    "target_output_dim": 38,
    "sens_output_dim": 5,
    "target_disc_hidden_dims": [100, 100],
    "sens_disc_hidden_dims": [100, 100],
    "target_disc_batch_norm": False,
    "predictor_epochs": 10,
    # adam parameters
    "encoder_lr": 1e-4,
    "encoder_weight_decay": 5e-2,
    "discs_lr": 1e-4,
    "discs_weight_decay": 5e-2,
    "loss_components": "entropy,kl,orth",
    "step_size": 30,
}
# {
#     "z_dim": 100,
#     "max_epochs": 20,
#     "target_output_dim": 65,
#     "target_disc_hidden_dims": [100, 100],
# }
DEFAULTS_CIFAR10 = {
    "z_dim": 128,
    "max_epochs": 30,
    "batch_size": 128,
    "lambda_od": 0.063,
    "gamma_od": 1.7,
    "lambda_entropy": 1.0,
    "gamma_entropy": 1.0,
    "target_output_dim": 1,
    "sens_output_dim": 10,
    "target_disc_hidden_dims": [256, 128],
    "sens_disc_hidden_dims": [256, 128],
    "predictor_epochs": 10,
    "target_disc_batch_norm": False,
    "encoder_lr": 1e-4,
    "encoder_weight_decay": 1e-2,
    "discs_lr": 1e-2,
    "discs_weight_decay": 1e-3,
    "target_predictor_hidden_dims": [256, 128],
    "target_predictor_output_dim": 1,
    "loss_components": "entropy,kl,orth",
    "step_size": 30,
}

DEFAULTS_CIFAR100 = {
    "z_dim": 128,
    "max_epochs": 80,
    "batch_size": 128,
    "lambda_od": 0.0325,
    "lambda_entropy": 0.1,
    "gamma_od": 1.2,
    "gamma_entropy": 1.67,
    "target_output_dim": 20,
    "sens_output_dim": 100,
    "target_disc_hidden_dims": [256, 128],
    "sens_disc_hidden_dims": [256, 128],
    "target_disc_batch_norm": False,
    "predictor_epochs": 10,
    # adam parameters
    "encoder_lr": 1e-4,
    "encoder_weight_decay": 1e-2,
    "discs_lr": 1e-2,
    "discs_weight_decay": 1e-3,
    "target_predictor_hidden_dims": [256, 128],
    "target_predictor_output_dim": 20,
    "loss_components": "entropy,kl,orth",
    "step_size": 30,
}

DATASET2DEFAULTS = {
    "adult": DEFAULTS_ADULT,
    "german": DEFAULTS_GERMAN,
    "yaleb": DEFAULTS_YALEB,
    "cifar10": DEFAULTS_CIFAR10,
    "cifar100": DEFAULTS_CIFAR100,
}
