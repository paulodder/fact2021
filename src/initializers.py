from fodvae import FODVAE
from torch import optim, nn
from models import MLP, MLPEncoder, ResNetEncoder
from predictors import MLPPredictor, LRPredictorTrainer, MLPPredictorTrainer
from dataloaders import load_representation_dataloaders
from evaluation import EvaluationManager


def optim_init_fn(model):
    return optim.Adam(model.parameters())


def _cifar10_target_predictor(args):
    """Gets target predictor for the cifar10 dataset"""
    z_dim = args.z_dim
    output_dim = 1
    return MLPPredictor.init_without_model(
        input_dim=z_dim,
        output_dim=output_dim,
        hidden_dims=[256, 128],
        optim_init_fn=optim_init_fn,
    )


def _cifar100_target_predictor(args):
    """Gets target predictor for the cifar100 dataset"""
    z_dim = args.z_dim
    output_dim = 20
    return MLPPredictor.init_without_model(
        input_dim=z_dim,
        output_dim=output_dim,
        hidden_dims=[256, 128],
        optim_init_fn=optim_init_fn,
    )


def get_sensitive_discriminator(args):
    if args.dataset in {"adult", "german"}:
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[64, 64],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
    elif args.dataset == "yaleb":
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[100, 100],
            output_dim=5,
            nonlinearity=nn.Softmax,
        )
    elif args.dataset == "cifar10":
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[256, 128],
            output_dim=10,
            nonlinearity=nn.Sigmoid,
        )
    else:
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[256, 128],
            output_dim=100,
            nonlinearity=nn.Sigmoid,
        )
    return model


def get_target_discriminator(args):
    if args.dataset in {"adult", "german"}:
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[64, 64],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
    elif args.dataset == "yaleb":
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[100, 100],
            output_dim=65,
            nonlinearity=nn.Softmax,
        )
    elif args.dataset == "cifar10":
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[256, 128],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
    else:
        model = MLP(
            input_dim=args.z_dim,
            hidden_dims=[256, 128],
            output_dim=20,
            nonlinearity=nn.Sigmoid,
        )
    return model


def get_fodvae(args):
    "gets FODVAE according to args"
    loss_components = args.loss_components.split(",")
    if args.dataset == "adult":
        lambda_od = args.get("lambda_od", 0.036)
        lambda_entropy = args.get("lambda_entropy", 0.55)
        gamma_od = args.get("gamma_od", 0.8)
        gamma_entropy = args.get("gamma_entropy", 0.133)
        step_size = args.get("step_size", 30)
        input_dim = 108
        encoder = MLPEncoder(input_dim=input_dim, z_dim=args.z_dim)
        disc_target = MLP(
            input_dim=args.z_dim,
            hidden_dims=[64, 64],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
        disc_sensitive = get_sensitive_discriminator(args)
        fvae = FODVAE(
            encoder,
            disc_target,
            disc_sensitive,
            lambda_od=lambda_od,
            lambda_entropy=lambda_entropy,
            gamma_od=gamma_od,
            gamma_entropy=gamma_entropy,
            step_size=step_size,
            z_dim=args.z_dim,
            dataset=args.dataset,
            loss_components=loss_components,
        )
        return fvae
    elif args.dataset == "german":
        lambda_od = args.get("lambda_od", 0.036)
        lambda_entropy = args.get("lambda_entropy", 0.55)
        gamma_od = args.get("gamma_od", 0.8)
        gamma_entropy = args.get("gamma_entropy", 0.133)
        step_size = args.get("step_size", 30)
        input_dim = 61
        encoder = MLPEncoder(input_dim=input_dim, z_dim=args.z_dim)
        disc_target = MLP(
            input_dim=args.z_dim,
            hidden_dims=[64, 64],
            output_dim=1,
            nonlinearity=nn.Sigmoid,
        )
        disc_sensitive = get_sensitive_discriminator(args)
        fvae = FODVAE(
            encoder,
            disc_target,
            disc_sensitive,
            lambda_od=lambda_od,
            lambda_entropy=lambda_entropy,
            gamma_od=gamma_od,
            gamma_entropy=gamma_entropy,
            step_size=step_size,
            z_dim=args.z_dim,
            dataset=args.dataset,
            loss_components=loss_components,
        )
        return fvae
    elif args.dataset == "yaleb":
        lambda_od = args.get("lambda_od", 0.037)
        lambda_entropy = args.get("lambda_entropy", 1)
        gamma_od = args.get("gamma_od", 1.1)
        gamma_entropy = args.get("gamma_entropy", 2)
        step_size = args.get("step_size", 100)
        input_dim = 32256
        encoder = MLPEncoder(
            input_dim=input_dim, hidden_dims=[], z_dim=args.z_dim
        )
        disc_target = MLP(
            input_dim=args.z_dim,
            hidden_dims=[100, 100],
            output_dim=38,
            batch_norm=True,
            nonlinearity=nn.Softmax,
        )
        disc_sensitive = get_sensitive_discriminator(args)
        # fvae = FODVAE(
        #     encoder,
        #     disc_target,
        #     disc_sensitive,
        #     lambda_od=0.036,
        #     lambda_entropy=0.5,
        #     gamma_od=0.8,
        #     gamma_entropy=1.33,
        #     step_size=30,
        #     z_dim=args.z_dim,
        # )
        fvae = FODVAE(
            encoder,
            disc_target,
            disc_sensitive,
            lambda_od=lambda_od,
            lambda_entropy=lambda_entropy,
            gamma_od=gamma_od,
            gamma_entropy=gamma_entropy,
            step_size=step_size,
            z_dim=args.z_dim,
            dataset=args.dataset,
            loss_components=loss_components,
        )
        return fvae
    elif args.dataset in {"cifar10", "cifar100"}:
        lambda_od = args.get("lambda_od", 0.036)
        lambda_entropy = args.get("lambda_entropy", 0.55)
        gamma_od = args.get("gamma_od", 0.8)
        gamma_entropy = args.get("gamma_entropy", 0.133)
        step_size = args.get("step_size", 30)
        encoder = ResNetEncoder(z_dim=args.z_dim, continue_training=True)
        disc_target = get_target_discriminator(args)
        disc_sensitive = get_sensitive_discriminator(args)
        fvae = FODVAE(
            encoder,
            disc_target,
            disc_sensitive,
            lambda_od=lambda_od,
            lambda_entropy=lambda_entropy,
            gamma_od=gamma_od,
            gamma_entropy=gamma_entropy,
            step_size=step_size,
            z_dim=args.z_dim,
            dataset=args.dataset,
            loss_components=loss_components,
        )
        return fvae


def get_target_predictor_trainer(args):
    """Gets target predictor trainer depending on the args.dataset value and
    potentially relevant parameters defined in args"""
    if args.dataset in ["adult", "german"]:
        return LRPredictorTrainer()
    if args.dataset == "yaleb":
        return LRPredictorTrainer(lambda ds: ds.y.argmax(1))
    if args.dataset == "cifar10":
        return MLPPredictorTrainer(
            _cifar10_target_predictor(args),
            epochs=args.predictor_epochs,
        )
    if args.dataset == "cifar100":
        return MLPPredictorTrainer(
            _cifar100_target_predictor(args), epochs=args.predictor_epochs
        )
    raise ValueError(f"dataset {dataset} is not recognized.")


def get_sensitive_predictor_trainer(args):
    model = get_sensitive_discriminator(args)
    return MLPPredictorTrainer(
        MLPPredictor(
            model,
            optim_init_fn,
        ),
        epochs=args.predictor_epochs,
    )


def get_evaluation_managers(args, get_embs):
    (
        train_dl_target,
        test_dl_target,
        train_dl_sens,
        test_dl_sens,
    ) = load_representation_dataloaders(
        args.dataset, args.batch_size, get_embs
    )

    predictor_target_trainer = get_target_predictor_trainer(args)
    predictor_sens_trainer = get_sensitive_predictor_trainer(args)

    eval_on_test = args.get("eval_on_test", True)
    eval_manager_target = EvaluationManager(
        predictor_target_trainer,
        train_dl_target,
        test_dl_target,
        eval_on_test=eval_on_test,
    )
    eval_manager_sens = EvaluationManager(
        predictor_sens_trainer,
        train_dl_sens,
        test_dl_sens,
        eval_on_test=eval_on_test,
    )
    return eval_manager_target, eval_manager_sens
