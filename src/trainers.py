import pytorch_lightning as pl
import copy


class TrainerWithSavedModels(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__models = []

    def get_saved_models(self):
        return self.__models

    def save_model_on_train_epoch_end(self):
        model = copy.deepcopy(self.get_model())
        self.__models.append(model)

    def call_hook(self, hook_name, *args, **kwargs):
        # Hack to run our code at the end of a training epoch
        if hook_name == "on_train_epoch_end":
            self.save_model_on_train_epoch_end()
        # Resume normal behaviour
        super().call_hook(hook_name, *args, **kwargs)
