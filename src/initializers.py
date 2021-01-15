from fodvae import FODVAE
from torch import optim, nn
from models import MLP, MLPEncoder, ResNetEncoder
from predictors import MLPPredictor, LRPredictor, MLPPredictorTrainer


def _cifar10_target_predictor(args):
    """Gets target predictor for the cifar10 dataset"""
    z_dim = args.z_dim
    output_dim = 1
    optim_init_fn = lambda model: optim.Adam(model.parameters())
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
    optim_init_fn = lambda model: optim.Adam(model.parameters())
    return MLPPredictor.init_with_model(
        input_dim=z_dim,
        output_dim=output_dim,
        optim_init_fn=optim_init_fn,
        hidden_dims=[256, 128],
    )


def get_target_predictor(args):
    """Gets target predictor depending on the args.dataset value and
    potentially relevant parameters defined in args"""
    if args.dataset in ["adult", "german"]:
        return LRPredictor.predict_targets()
    if args.dataset == "yaleb":
        # optim_init_fn = lambda model: optim.Adam(model.parameters())
        # return MLPPredictorTrainer(
        #     MLPPredictor(
        #         MLP(
        #             input_dim=args.z_dim,
        #             hidden_dims=[],
        #             output_dim=38,
        #             batch_norm=False,
        #         ),
        #         optim_init_fn,
        #     ),
        #     30,
        # )
        return LRPredictor(lambda ds: ds.targets.argmax(1))
    if args.dataset == "cifar10":
        return MLPPredictorTrainer(
            _cifar10_target_predictor(args), max_epochs=args.max_epochs,
        )
    if args.dataset == "cifar100":
        return MLPPredictorTrainer(
            _cifar100_target_predictor(args), args.max_epochs
        )
    raise ValueError(f"dataset {dataset} is not recognized.")


def get_sensitive_predictor(model, args):
    optim_init_fn = lambda model: optim.Adam(model.parameters())
    return MLPPredictorTrainer(
        MLPPredictor(model, optim_init_fn, train_for_sensitive=True,),
        max_epochs=args.max_epochs,
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
    if args.dataset == "adult":
        lambda_od = args.get("lambda_od", 0.036)
        lambda_entropy = args.get("lambda_entropy", 0.55)
        gamma_od = args.get("gamma_od", 0.8)
        gamma_entropy = args.get("gamma_entropy", 0.133)
        step_size = args.get("step_size", 1000)
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
        )
        return fvae
    elif args.dataset == "german":
        lambda_od = args.get("lambda_od", 0.036)
        lambda_entropy = args.get("lambda_entropy", 0.55)
        gamma_od = args.get("gamma_od", 0.8)
        gamma_entropy = args.get("gamma_entropy", 0.133)
        step_size = args.get("step_size", 1000)
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
        #     step_size=1000,
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
        )
        return fvae
    else:
        lambda_od = args.get("lambda_od", 0.036)
        lambda_entropy = args.get("lambda_entropy", 0.55)
        gamma_od = args.get("gamma_od", 0.8)
        gamma_entropy = args.get("gamma_entropy", 0.133)
        step_size = args.get("step_size", 1000)
        encoder = ResNetEncoder(z_dim=args.z_dim)
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
        )
        return fvae
