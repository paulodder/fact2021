from fodvae import FODVAE
from torch import optim, nn
from models import MLP, MLPEncoder, ResNetEncoder
from predictors import MLPPredictor, LRPredictorTrainer, MLPPredictorTrainer
from dataloaders import load_representation_dataloader
from evaluation import EvaluationManager
from defaults import DATASET2DEFAULTS, DEFAULTS_CIFAR10


def _cifar10_target_predictor(config):
    """Gets target predictor for the cifar10 dataset"""
    output_dim = 1
    optim_init_fn = lambda model: optim.Adam(model.parameters())
    return MLPPredictor.init_without_model(
        input_dim=config.z_dim,
        output_dim=CIFAR10_DEFAULTS["target_predictor_output_dim"],
        hidden_dims=CIFAR10_DEFAULTS["target_predictor_hidden_dims"],
        optim_init_fn=optim_init_fn,
    )


def _cifar100_target_predictor(config):
    """Gets target predictor for the cifar100 dataset"""
    # z_dim = config.z_dim
    # output_dim = 20
    optim_init_fn = lambda model: optim.Adam(model.parameters())
    return MLPPredictor.init_with_model(
        input_dim=config.z_dim,
        output_dim=CIFAR100_DEFAULTS["target_predictor_output_dim"],
        hidden_dims=CIFAR100_DEFAULTS["target_predictor_hidden_dims"],
    )


def get_sensitive_discriminator(args):
    model = MLP(
        input_dim=args.z_dim,
        hidden_dims=args.sens_disc_hidden_dims,
        output_dim=args.sens_output_dim,
        nonlinearity=nn.Sigmoid if args.sens_output_dim == 1 else nn.Softmax,
    )
    return model
    # elif args.dataset == "yaleb":
    #     model = MLP(
    #         input_dim=args.z_dim,
    #         hidden_dims=[100, 100],
    #         output_dim=5,
    #         nonlinearity=nn.Softmax,
    #     )
    # elif args.dataset == "cifar10":
    #     model = MLP(
    #         input_dim=args.z_dim,
    #         hidden_dims=[256, 128],
    #         output_dim=10,
    #         nonlinearity=nn.Sigmoid,
    #     )
    # else:
    #     model = MLP(
    #         input_dim=args.z_dim,
    #         hidden_dims=[256, 128],
    #         output_dim=100,
    #         nonlinearity=nn.Sigmoid,
    #     )
    # return model


def get_target_discriminator(config):
    model = MLP(
        input_dim=config.z_dim,
        hidden_dims=config.target_disc_hidden_dims,
        output_dim=config.target_output_dim,
        batch_norm=config.batch_norm,
        nonlinearity=nn.Sigmoid
        if config.target_output_dim == 1
        else nn.Softmax,
    )
    return model
    # elif args.dataset == "yaleb":
    #     model = MLP(
    #         input_dim=args.z_dim,
    #         hidden_dims=args.target_output_dim,
    #         output_dim=args.target_disc_hidden_dims,
    #         nonlinearity=nn.Softmax,
    #     )
    # elif args.dataset == "cifar10":
    #     model = MLP(
    #         input_dim=args.z_dim,
    #         hidden_dims=[256, 128],
    #         output_dim=1,
    #         nonlinearity=nn.Sigmoid,
    #     )
    # else:
    #     model = MLP(
    #         input_dim=args.z_dim,
    #         hidden_dims=[256, 128],
    #         output_dim=20,
    #         nonlinearity=nn.Sigmoid,
    #     )
    # return model


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
    # if config.dataset == "adult":
    #     # lambda_od = config.get("lambda_od", 0.036)
    #     # lambda_entropy = config.get("lambda_entropy", 0.55)
    #     # gamma_od = config.get("gamma_od", 0.8)
    #     # gamma_entropy = config.get("gamma_entropy", 0.133)
    #     # step_size = config.get("step_size", 30)
    #     # input_dim = 108
    #     encoder = get_encoder(config)

    #     fvae = FODVAE(
    #         encoder,
    #         disc_target,
    #         disc_sensitive,
    #         lambda_od=config.lambda_od,
    #         lambda_entropy=config.lambda_entropy,
    #         gamma_od=config.gamma_od,
    #         gamma_entropy=config.gamma_entropy,
    #         step_size=config.step_size,
    #         z_dim=config.z_dim,
    #         dataset=config.dataset,
    #         loss_components=config.loss_components,
    #     )
    #     return fvae
    # elif config.dataset == "german":
    #     lambda_od = config.get("lambda_od", 0.036)
    #     lambda_entropy = config.get("lambda_entropy", 0.55)
    #     gamma_od = config.get("gamma_od", 0.8)
    #     gamma_entropy = config.get("gamma_entropy", 0.133)
    #     step_size = config.get("step_size", 30)
    #     input_dim = 61
    #     encoder = MLPEncoder(input_dim=input_dim, z_dim=config.z_dim)
    #     return fvae
    # elif config.dataset == "yaleb":
    #     lambda_od = config.get("lambda_od", 0.037)
    #     lambda_entropy = config.get("lambda_entropy", 1)
    #     gamma_od = config.get("gamma_od", 1.1)
    #     gamma_entropy = config.get("gamma_entropy", 2)
    #     step_size = config.get("step_size", 100)
    #     input_dim = 32256
    #     encoder = MLPEncoder(
    #         input_dim=input_dim, hidden_dims=[], z_dim=config.z_dim
    #     )
    #     # MLP(
    #     #     input_dim=config.z_dim,
    #     #     hidden_dims=[100, 100],
    #     #     output_dim=38,
    #     #     batch_norm=True,
    #     #     nonlinearity=nn.Softmax,
    #     # )
    #     # fvae = FODVAE(
    #     #     encoder,
    #     #     disc_target,
    #     #     disc_sensitive,
    #     #     lambda_od=0.036,
    #     #     lambda_entropy=0.5,
    #     #     gamma_od=0.8,
    #     #     gamma_entropy=1.33,
    #     #     step_size=30,
    #     #     z_dim=config.z_dim,
    #     # )
    #     fvae = FODVAE(
    #         encoder,
    #         disc_target,
    #         disc_sensitive,
    #         lambda_od=lambda_od,
    #         lambda_entropy=lambda_entropy,
    #         gamma_od=gamma_od,
    #         gamma_entropy=gamma_entropy,
    #         step_size=step_size,
    #         z_dim=config.z_dim,
    #         dataset=config.dataset,
    #         loss_components=loss_components,
    #     )
    #     return fvae
    # elif config.dataset in {"cifar10", "cifar100"}:
    #     lambda_od = config.get("lambda_od", 0.036)
    #     lambda_entropy = config.get("lambda_entropy", 0.55)
    #     gamma_od = config.get("gamma_od", 0.8)
    #     gamma_entropy = config.get("gamma_entropy", 0.133)
    #     step_size = config.get("step_size", 30)
    #     encoder = ResNetEncoder(z_dim=config.z_dim, continue_training=True)
    #     disc_target = get_target_discriminator(config)
    #     fvae = FODVAE(
    #         encoder,
    #         disc_target,
    #         disc_sensitive,
    #         lambda_od=lambda_od,
    #         lambda_entropy=lambda_entropy,
    #         gamma_od=gamma_od,
    #         gamma_entropy=gamma_entropy,
    #         step_size=step_size,
    #         z_dim=config.z_dim,
    #         dataset=config.dataset,
    #         loss_components=loss_components,
    #     )
    #     return fvae


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
            max_epochs=config.predictor_epochs,
        )
    if config.dataset == "cifar100":
        return MLPPredictorTrainer(
            _cifar100_target_predictor(config), config.max_epochs
        )
    raise ValueError(f"dataset {dataset} is not recognized.")


def get_sensitive_predictor_trainer(config):
    model = get_sensitive_discriminator(config)
    optim_init_fn = lambda model: optim.Adam(model.parameters())
    return MLPPredictorTrainer(
        MLPPredictor(
            model,
            optim_init_fn,
        ),
        epochs=config.predictor_epochs,
    )


def get_evaluation_managers(config, get_embs):
    train_dl_target, test_dl_target = load_representation_dataloader(
        config.dataset, config.batch_size, get_embs, y_is_target=True
    )
    train_dl_sens, test_dl_sens = load_representation_dataloader(
        config.dataset, config.batch_size, get_embs, y_is_target=False
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
