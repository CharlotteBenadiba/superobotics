import time

from typing import Type

import torch

from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, \

    MulticlassRecall

import config

from classifier import Classifier

from data_manager import TestingDataManager

class Evaluator:

    def __init__(self, classifier: Classifier, data_manager: TestingDataManager):

        """

        Initializes the evaluator with the given classifier and data manager

        :param classifier: Classifier - the classifier to evaluate

        :param data_manager: DataManager - the data manager containing the test data

        """

        self._classifier = classifier

        self._data_manager = data_manager

        self._device = config.DEVICE

        self._labels = torch.zeros(self._data_manager.dataset_sizes[config.TEST])

        self._predictions = torch.zeros(self._data_manager.dataset_sizes[config.TEST])

        self._average_accuracy = None

        self._average_precision = None

        self._average_recall = None

        self._accuracies = None

        self._precisions = None

        self._recalls = None

    def evaluate(self):

        """

        Evaluates the model on the test data. The results are stored in the class private variables and used by the

        properties to return the evaluation metrics

        """

        start = time.perf_counter()

        print("Evaluating the model on the test data...")

        self._classifier.model.eval()

        with torch.no_grad():

            inputs, labels = next(iter(self._data_manager.dataloaders[config.TEST]))

            inputs, labels = inputs.to(self._device), labels.to(self._device)

            outputs = self._classifier.model(inputs)

            _, predicted = torch.max(outputs, 1)

            self._labels = labels

            self._predictions = predicted

        print(f"Evaluation took {time.perf_counter() - start:.2f} seconds")

    @property

    def average_accuracy(self) -> float:

        """

        Returns the average accuracy of the model on the test data

        :return: float - the average accuracy of the model on the test data

        """

        if self._average_accuracy is None:

            self._average_accuracy = self._get_average_metric(MulticlassAccuracy)

        return self._average_accuracy

    @property

    def average_precision(self) -> float:

        """

        Returns the average precision of the model on the test data

        :return: float - the average precision of the model on the test data

        """

        if self._average_precision is None:

            self._average_precision = self._get_average_metric(MulticlassPrecision)

        return self._average_precision

    @property

    def average_recall(self) -> float:

        """

        Returns the average recall of the model on the test data

        :return: float - the average recall of the model on the test data

        """

        if self._average_recall is None:

            self._average_recall = self._get_average_metric(MulticlassRecall)

        return self._average_recall

    def class_accuracy(self, class_name: str) -> float:

        """

        Returns the accuracy of the given class

        :param class_name: str - the name of the class (as it appears in the class names file)

        :return: float - the accuracy of the given class

        """

        if self._accuracies is None:

            self._accuracies = self._get_class_metric(MulticlassAccuracy)

        return float(self._accuracies[self._classifier.get_class_index(class_name)])

    def class_precision(self, class_name: str) -> float:

        """

        Returns the precision of the given class

        :param class_name: str - the name of the class (as it appears in the class names file)

        :return: float - the precision of the given class

        """

        if self._precisions is None:

            self._precisions = self._get_class_metric(MulticlassPrecision)

        return float(self._precisions[self._classifier.get_class_index(class_name)])

    def class_recall(self, class_name: str) -> float:

        """

        Returns the recall of the given class

        :param class_name: str - the name of the class (as it appears in the class names file)

        :return: float - the recall of the given class

        """

        if self._recalls is None:

            self._recalls = self._get_class_metric(MulticlassRecall)

        return float(self._recalls[self._classifier.get_class_index(class_name)])

    def _get_average_metric(self, metric: Type) -> float:

        """

        Returns the average of the given metric

        :param metric: Type - the metric to calculate. Must be a class that creates a

        torchmetrics.classification.MulticlassStatScores object

        :return: float - the average of the given metric

        """

        return float(metric(self._data_manager.num_classes).to(self._device)(self._predictions, self._labels))

    def _get_class_metric(self, metric: Type) -> torch.Tensor:

        """

        Returns the metric for each class

        :param metric: Type - the metric to calculate. Must be a class that creates a

        torchmetrics.classification.MulticlassStatScores object

        :return: torch.Tensor - the metric for each class

        """

        return metric(self._data_manager.num_classes, average=None).to(self._device)(self._predictions, self._labels)
