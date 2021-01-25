import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from torch import optim, nn
from models import MLP
import utils
import trainers


class LRPredictor:
    def __init__(self, dataset2y=None):
        if dataset2y is None:
            dataset2y = lambda ds: ds.y
        self.dataset2y = dataset2y
        self.model = LogisticRegression()

    def fit(self, dataloader):
        x = dataloader.dataset.x
        y = self.dataset2y(dataloader.dataset)
        self.model.fit(x, y)

    def predict(self, dataloader):
        x = dataloader.dataset.x
        return self.model.predict(x)


class LRPredictorTrainer(LRPredictor):
    def get_saved_models(self):
        return [self.model]

    def forward_model(self, model, x):
        return model.predict(x)


class MLPPredictor(pl.LightningModule):
    def __init__(self, model, optim_init_fn):
        super(MLPPredictor, self).__init__()
        self.model = model
        self.optim_init_fn = optim_init_fn
        self.loss_fn = utils.bce_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optim = self.optim_init_fn(self.model)
        return optim

    automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimizers().step()

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
        self,
        predictor,
        epochs,
    ):
        self.predictor = predictor
        self.trainer = trainers.TrainerWithSavedModels(
            min_epochs=epochs, max_epochs=epochs
        )

    def fit(self, dataloader):
        self.trainer.fit(self.predictor, dataloader)

    def predict(self, dataloader):
        x = dataloader.dataset.x
        return self.predictor.forward(x)

    def get_saved_models(self):
        return self.trainer.get_saved_models()

    def forward_model(self, model, x):
        return model.forward(x)
