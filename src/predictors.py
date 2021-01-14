import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from torch import optim, nn
from models import MLP
import utils


class LRPredictor:
    def __init__(self, dataset2y):
        self.model = LogisticRegression()
        self.dataset2y = dataset2y

    def fit(self, dataloader):
        x = dataloader.dataset.targets_latent
        y = self.dataset2y(dataloader.dataset)
        self.model.fit(x, y)

    def predict(self, dataloader):
        x = dataloader.dataset.targets_latent
        return self.model.predict(x)

    @classmethod
    def predict_targets(cls):
        return cls(lambda ds: ds.targets)

    @classmethod
    def predict_sens(cls):
        return cls(lambda ds: ds.s)


class MLPPredictor(pl.LightningModule):
    def __init__(self, model, optim_init_fn, train_for_sensitive=False):
        super(MLPPredictor, self).__init__()
        self.model = model
        self.optim_init_fn = optim_init_fn
        self.train_for_sensitive = train_for_sensitive
        self.loss_fn = utils.bce_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = self.optim_init_fn(self.model)
        return optim

    def training_step(self, batch, batch_idx):
        x, y, s = batch
        output = self.forward(x)
        goal = s if self.train_for_sensitive else y
        loss = self.loss_fn(output, goal)
        return loss

    @classmethod
    def init_without_model(
        cls, input_dim, output_dim, optim_init_fn, hidden_dims=[256, 128]
    ):
        return cls(
            MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
            ),
            optim_init_fn=optim_init_fn,
        )


class MLPPredictorTrainer:
    def __init__(
        self, predictor, max_epochs,
    ):
        self.predictor = predictor
        self.trainer = pl.Trainer(max_epochs=max_epochs)

    def fit(self, dataloader):
        self.trainer.fit(self.predictor, dataloader)

    def predict(self, dataloader):
        x = dataloader.dataset.targets_latent
        return self.predictor.forward(x)
