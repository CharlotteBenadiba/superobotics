from typing import List

import torch

from PIL import Image

from torch import nn

from reference_dict import name_reference_dict

import config

class Classifier:

    def __init__(self, model: nn.Module = None):

        """

        Initializes the classifier with model, class names, and device

        :param model: nn.Module - optional model to use. If not provided, the classifier will load the model using the

        _load_model method

        """

        self.model = model if model is not None else self._load_model()

        assert self.model is not None, "Model not found"

        self._class_names = self._get_all_class_names()

        self._device = config.DEVICE

    def __call__(self, image_path: str) -> int:

        """

        Calls the predict method with the given image path. This method is for convenience of use: instead of calling

        classifier.predict(image_path), you can call classifier(image_path)

        :param image_path: str - path to the image

        :return: int - the product reference number

        """

        return self.predict(image_path)

    def predict(self, image_path: str) -> int:

        """

        Predicts the class of the image in the given path and returns the product reference number

        :param image_path: path to the image

        :type image_path: str

        :return: int - the product reference number

        """

        input_batch = self._preprocess_image(image_path).to(self._device)

        self.model.eval()

        with torch.no_grad():

            prediction = self.model(input_batch)

        _, index = torch.max(prediction, 1)

        return self._idx_to_reference(index)

    @staticmethod

    def _preprocess_image(image_path: str) -> torch.Tensor:

        """

        Preprocesses the image in the given path to a tensor that can be used as input to the model

        :param image_path: str - path to the image

        :return: torch.Tensor - the preprocessed image as a tensor, ready to be used as input to the model

        """

        input_image = Image.open(image_path)

        preprocess = config.PREPROCESSING_DEFAULT_TRANSFORMS

        input_tensor = preprocess(input_image)

        return input_tensor.unsqueeze(0)

    def _idx_to_reference(self, index: int) -> int:

        """

        Converts the index of the class, as predicted by the model, to the product reference number

        :param index: int - the index of the class

        :return: int - the product reference number

        """

        return name_reference_dict[self._get_class_name(index)]

    @staticmethod

    def _get_all_class_names() -> List[str]:

        """

        Read the list of all class names from the file CLASS_NAMES_PATH and return it as a list

        This file is created by the DataManager class that gets initialized in the training process

        :return: List[str] - the list of all class names

        """

        with open(config.CLASS_NAMES_PATH, 'r') as file:

            return file.read().splitlines()

    def _get_class_name(self, index: int) -> str:

        """

        Returns the class name for the given index

        :param index: int - the index of the class, as predicted by the model

        :return: str - the class name

        """

        return self._class_names[index]

    def get_class_index(self, class_name: str) -> int:

        """

        Returns the index of the class with the given name

        :param class_name: str - the name of the class

        :return: int - the index of the class

        """

        return self._class_names.index(class_name)

    @staticmethod

    def _load_model() -> nn.Module:

        """

        Load the model from the file BEST_MODEL_PARAMS_PATH and return it

        :return: nn.Module - the model

        """

        return torch.load(config.BEST_MODEL_PARAMS_PATH)
