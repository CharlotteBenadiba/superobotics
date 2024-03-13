import time

import torch

from torch import nn

import config

from data_manager import TrainingDataManager

class Trainer:

    def __init__(self, data_manager: TrainingDataManager):

        """

        Initializes the trainer with the given data manager

        :param data_manager: the data manager to use for training, containing the training and validation data

        :type data_manager: TrainingDataManager

        """

        self._device = config.DEVICE

        self._model = self._prepare_pretrained_model_for_finetune()

        self._data_manager = data_manager

        self._init_training_params()

    def _init_training_params(self):

        """

        Initializes the training parameters based on the config file

        """

        self._criterion = config.CRITERION

        self._optimizer = config.OPTIMIZER(self._model.fc.parameters(), lr=config.LEARNING_RATE,

                                           momentum=config.MOMENTUM)

        self._scheduler = config.SCHEDULER(self._optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)

        self._num_epochs = config.NUM_EPOCHS

    def train_model(self):

        """

        Trains the model using the training and validation data from the data manager

        """

        since = time.perf_counter()

        dataloaders = self._data_manager.dataloaders

        dataset_sizes = self._data_manager.dataset_sizes

        best_model_params_path = config.BEST_MODEL_PARAMS_PATH

        torch.save(self._model.state_dict(), best_model_params_path)

        best_acc = 0.0

        for epoch in range(self._num_epochs):

            print(f'Epoch {epoch}/{self._num_epochs - 1}')

            print('-' * 10)

            # Each epoch has a training and validation phase

            for phase in [config.TRAIN, config.VAL]:

                if phase == config.TRAIN:

                    self._model.train()  # Set model to training mode

                else:

                    self._model.eval()   # Set model to evaluate mode

                running_loss = 0.0

                running_corrects = 0

                for inputs, labels in dataloaders[phase]:

                    inputs = inputs.to(self._device)

                    labels = labels.to(self._device)

                    # zero the parameter gradients

                    self._optimizer.zero_grad()

                    # forward

                    # track history if only in train

                    with torch.set_grad_enabled(phase == config.TRAIN):

                        outputs = self._model(inputs)

                        _, predictions = torch.max(outputs, 1)

                        loss = self._criterion(outputs, labels)

                        # backward + optimize only if in training phase

                        if phase == config.TRAIN:

                            loss.backward()

                            self._optimizer.step()

                    # statistics

                    running_loss += loss.item() * inputs.size(0)

                    running_corrects += torch.sum(predictions == labels.data)      # type: ignore

                if phase == config.TRAIN:

                    self._scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]

                epoch_acc = running_corrects / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model

                if phase == config.VAL and epoch_acc > best_acc:

                    best_acc = epoch_acc

                    torch.save(self._model.state_dict(), best_model_params_path)

        time_elapsed = time.perf_counter() - since

        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights

        self._model.load_state_dict(torch.load(best_model_params_path))

        torch.save(self._model, config.BEST_MODEL_PARAMS_PATH)

    def _prepare_pretrained_model_for_finetune(self) -> nn.Module:

        """

        Prepares the pretrained model for fine-tuning by replacing the last layer with a new layer with the number of

        classes in the dataset. It expects the model to be a torchvision model with a 'fc' attribute that is the last

        layer of the model

        :return: nn.Module - the model ready for fine-tuning

        """

        model_conv = config.BASE_PRETRAINED_MODEL

        for param in model_conv.parameters():

            param.requires_grad = False

        num_features = model_conv.fc.in_features

        model_conv.fc = nn.Linear(num_features, config.INITIAL_MODEL_NUM_CLASSES)

        model_conv = model_conv.to(self._device)

        return model_conv
