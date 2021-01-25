import pytorch_lightning as pl
import torch
from sklearn.metrics import classification_report
import utils


class EvaluationManager:
    def __init__(
        self,
        trainer,
        train_dl,
        test_dl,
        eval_on_test=True,
    ):
        self.trainer = trainer
        self.train_dl = train_dl
        self.test_dl = test_dl if eval_on_test else train_dl

    def fit(self):
        self.trainer.fit(self.train_dl, gpus=get_n_gpus())

    def generate_report(self, model, x, y):
        # We do not have control over the interface of the model
        # itself, so we leave this up to the trainer.
        y_pred = self.trainer.forward_model(model, x)
        y_pred = utils.prepare_tensor_for_evaluation(y_pred)

        return classification_report(y, y_pred, output_dict=True), y_pred

    def evaluate(self, output_dict=True):
        models = self.trainer.get_saved_models()
        best_model_and_report = None
        best_performance = None
        best_i = 0

        x = self.test_dl.dataset.x
        y = utils.prepare_tensor_for_evaluation(self.test_dl.dataset.y)

        for i, model in enumerate(models):
            report, y_pred = self.generate_report(model, x, y)
            performance = report["accuracy"]

            if best_model_and_report is None or best_performance < performance:
                if not output_dict:
                    report = classification_report(
                        y, y_pred, output_dict=False
                    )
                best_model_and_report = (model, report)
                best_performance = performance
                best_i = i

        print(f"[eval] best model version: {best_i+1}/{len(models)}")
        return (*best_model_and_report, best_performance)
