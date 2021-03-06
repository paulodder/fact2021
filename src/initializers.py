import utils
from fodvae import FODVAE
from torch import optim, nn
from models import MLP, MLPEncoder, ResNetEncoder
from predictors import MLPPredictor, LRPredictorTrainer, MLPPredictorTrainer
from dataloaders import load_representation_dataloaders
from evaluation import EvaluationManager
from defaults import DATASET2DEFAULTS, DEFAULTS_CIFAR10, DEFAULTS_CIFAR100


def optim_init_fn(model):
    return optim.Adam(model.parameters())


def _cifar10_target_predictor(config):
    """Gets target predictor for the cifar10 dataset"""
    output_dim = 1
    return MLPPredictor.init_without_model(
        input_dim=config.z_dim,
        output_dim=DEFAULTS_CIFAR10["target_predictor_output_dim"],
        hidden_dims=DEFAULTS_CIFAR10["target_predictor_hidden_dims"],
        optim_init_fn=optim_init_fn,
    )


def _cifar100_target_predictor(config):
    """Gets target predictor for the cifar100 dataset"""
    # z_dim = config.z_dim
    # output_dim = 20
    return MLPPredictor.init_without_model(
        input_dim=config.z_dim,
        output_dim=DEFAULTS_CIFAR100["target_predictor_output_dim"],
        hidden_dims=DEFAULTS_CIFAR100["target_predictor_hidden_dims"],
        optim_init_fn=optim_init_fn,
    )


def get_sensitive_discriminator(args):
    model = MLP(
        input_dim=args.z_dim,
        hidden_dims=args.sens_disc_hidden_dims,
        output_dim=args.sens_output_dim,
        nonlinearity=nn.Sigmoid if args.sens_output_dim == 1 else nn.Softmax,
    )
    return model


def get_target_discriminator(config):
    model = MLP(
        input_dim=config.z_dim,
        hidden_dims=config.target_disc_hidden_dims,
        output_dim=config.target_output_dim,
        batch_norm=config.target_disc_batch_norm,
        nonlinearity=nn.Sigmoid
        if config.target_output_dim == 1
        else nn.Softmax,
    )
    return model


def get_encoder(config):
    if config.dataset in {"adult", "german", "yaleb"}:
        model = MLPEncoder(
            input_dim=config.input_dim,
            hidden_dims=config.encoder_hidden_dims,
            z_dim=config.z_dim,
        )
    elif config.dataset in {"cifar10", "cifar100"}:
        model = ResNetEncoder(z_dim=config.z_dim, continue_training=True)
    return model


def get_fodvae(config):
    "gets FODVAE according to config"
    loss_components = config.loss_components.split(",")
    disc_sensitive = get_sensitive_discriminator(config)
    disc_target = get_target_discriminator(config)
    encoder = get_encoder(config)
    fvae = FODVAE(
        encoder,
        disc_target,
        disc_sensitive,
        lambda_od=config.lambda_od,
        lambda_entropy=config.lambda_entropy,
        gamma_od=config.gamma_od,
        gamma_entropy=config.gamma_entropy,
        step_size=config.step_size,
        encoder_lr=config.encoder_lr,
        encoder_weight_decay=config.encoder_weight_decay,
        discs_lr=config.discs_lr,
        discs_weight_decay=config.discs_weight_decay,
        z_dim=config.z_dim,
        dataset=config.dataset,
        loss_components=loss_components,
    )
    return fvae


def get_target_predictor_trainer(config):
    """Gets target predictor trainer depending on the config.dataset value and
    potentially relevant parameters defined in config
    returned object has a fit and predict method"""
    if config.dataset in {"adult", "german"}:
        return LRPredictorTrainer()
    if config.dataset == "yaleb":
        return LRPredictorTrainer(lambda ds: ds.y.argmax(1))
    if config.dataset == "cifar10":
        return MLPPredictorTrainer(
            _cifar10_target_predictor(config),
            epochs=config.predictor_epochs,
            gpus=utils.get_n_gpus(),
        )
    if config.dataset == "cifar100":
        return MLPPredictorTrainer(
            _cifar100_target_predictor(config),
            epochs=config.predictor_epochs,
            gpus=utils.get_n_gpus(),
        )
    raise ValueError(f"dataset {dataset} is not recognized.")


def get_sensitive_predictor_trainer(config):
    model = get_sensitive_discriminator(config)
    return MLPPredictorTrainer(
        MLPPredictor(
            model,
            optim_init_fn,
        ),
        gpus=utils.get_n_gpus(),
        epochs=config.predictor_epochs,
    )


def get_evaluation_managers(config, get_embs):
    (
        train_dl_target,
        test_dl_target,
        train_dl_sens,
        test_dl_sens,
    ) = load_representation_dataloaders(
        config.dataset, config.batch_size, get_embs
    )

    predictor_target_trainer = get_target_predictor_trainer(config)
    predictor_sens_trainer = get_sensitive_predictor_trainer(config)

    eval_manager_target = EvaluationManager(
        predictor_target_trainer,
        train_dl_target,
        test_dl_target,
        eval_on_test=config.eval_on_test,
    )
    eval_manager_sens = EvaluationManager(
        predictor_sens_trainer,
        train_dl_sens,
        test_dl_sens,
        eval_on_test=config.eval_on_test,
    )
    return eval_manager_target, eval_manager_sens
