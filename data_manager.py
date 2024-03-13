import os

from abc import abstractmethod

from typing import Dict

from torch.utils.data import DataLoader

from torchvision import datasets

import config

class DataManagerBase:

    def __init__(self, data_dir: str):

        """

        Initializes the data manager with the given data directory. The data manager will create the datasets and

        dataloaders for the training process or the testing process according to the inherited class

        :param data_dir: str - path to the data directory containing the data in a specific structure according to the

        inherited class

        """

        self._data_dir = data_dir

        self._init_data_dir()

        self._datasets = self._create_datasets()

        self._dataset_sizes = {x: len(self._datasets[x]) for x in self._datasets.keys()}

        self._dataloaders = self._create_dataloaders()

        self._num_classes = config.INITIAL_MODEL_NUM_CLASSES

    @property

    def dataloaders(self) -> Dict[str, DataLoader]:

        return self._dataloaders

    @property

    def dataset_sizes(self) -> Dict[str, int]:

        return self._dataset_sizes

    @property

    def num_classes(self) -> int:

        return self._num_classes

    @abstractmethod

    def _create_datasets(self) -> Dict[str, datasets.ImageFolder]:

        """

        Creates the datasets. The method must be implemented by the inherited class.

        :return: Dict[str, datasets.ImageFolder] - the datasets

        """

        pass

    def _create_dataloaders(self) -> Dict[str, DataLoader]:

        """

        Creates the dataloaders for the datasets. Default implementation uses a batch size of BATCH_SIZE and shuffles

        the data

        :return: Dict[str, DataLoader] - the dataloaders

        """

        return {x: DataLoader(self._datasets[x], batch_size=config.BATCH_SIZE, shuffle=True,

                              num_workers=config.NUM_WORKERS) for x in self._datasets.keys()}

    def _init_data_dir(self):

        """

        Initializes the data directory. The method should be implemented by the inherited class only if needed

        """

        pass

class TrainingDataManager(DataManagerBase):

    def __init__(self, data_dir: str):

        """

        Initializes the data manager for the training process

        :param data_dir: str - path to the data directory containing the data in the following structure:

            data_dir/<phase>/class_name/image.jpg for each phase (train, val, test)

            or data_dir/class_name/image.jpg (the data manager will divide the data into train, val and test folders)

        """

        super().__init__(data_dir)

        self._save_class_names()

    def _save_class_names(self):

        """

        Saves the class names to a file. The file will be used by the classifier to map the model's output to the

        corresponding class name

        """

        with open(config.CLASS_NAMES_PATH, 'w') as file:

            for class_name in self._datasets[config.TRAIN].classes:

                file.write(class_name + '\n')

    def _create_datasets(self) -> Dict[str, datasets.ImageFolder]:

        """

        Creates the datasets for the training process

        :return: Dict[str, datasets.ImageFolder] - the datasets for the training process

        """

        data_transforms = {

            config.TRAIN: config.PREPROCESSING_TRAIN_TRANSFORMS,

            config.VAL: config.PREPROCESSING_DEFAULT_TRANSFORMS

        }

        return {x: datasets.ImageFolder(os.path.join(self._data_dir, x), data_transforms[x])

                for x in data_transforms.keys()}

    def _init_data_dir(self):

        """

        Initializes the data directory. If the data directory does not contain train, val and test folders, the method

        will divide the data into train, val and test folders.

        """

        dir_list = os.listdir(self._data_dir)

        if config.TRAIN not in dir_list or config.VAL not in dir_list or config.TEST not in dir_list:

            self._divide_data()

    def _divide_data(self):

        """

        Divides the data into train and val folders

        The method assumes that the data is in the following structure:

            data_dir/class_name/image.jpg

        The method will create train, val and test folders such that the resulting structure will be:

            data_dir/<phase>/class_name/image.jpg for each phase (train, val, test)

        """

        train_path = os.path.join(self._data_dir, config.TRAIN)

        val_path = os.path.join(self._data_dir, config.VAL)

        test_path = os.path.join(self._data_dir, config.TEST)

        os.mkdir(train_path)

        os.mkdir(val_path)

        os.mkdir(test_path)

        for root, dirs, files in os.walk(self._data_dir):

            for dir_name in dirs:

                if dir_name in config.PHASES:

                    continue

                files = os.listdir(os.path.join(root, dir_name))

                num_files = len(files)

                first_split = int(num_files * 0.6)

                second_split = int(num_files * 0.8)

                for i, file in enumerate(files):

                    original_file_path = os.path.join(root, dir_name, file)

                    if i < first_split:

                        self._move_single_file(original_file_path, dir_name, file, train_path)

                    elif i < second_split:

                        self._move_single_file(original_file_path, dir_name, file, val_path)

                    else:

                        self._move_single_file(original_file_path, dir_name, file, test_path)

                parent_dir = os.path.join(root, dir_name)

                if not os.listdir(parent_dir):

                    os.rmdir(parent_dir)

    @staticmethod

    def _move_single_file(original_file_path: str, dir_name: str, file: str, new_parent_dir: str):

        """

        Moves a single file from the original_file_path to a new directory with the given dir_name and file name, such

        that the new directory is a subdirectory of the new_parent_dir. For example, if the original_file_path is

        'data_dir/class_name/image.jpg' and the new_parent_dir is 'train', the file will be moved to

        'data_dir/train/class_name/image.jpg'

        :param original_file_path: str - the original file path

        :param dir_name: str - the name of the directory to create

        :param file: str - the name of the file

        :param new_parent_dir: str - the new parent directory

        """

        new_dir_name = os.path.join(new_parent_dir, dir_name)

        if not os.path.exists(new_dir_name):

            os.mkdir(new_dir_name)

        os.rename(original_file_path, os.path.join(new_dir_name, str(file)))

class TestingDataManager(DataManagerBase):

    def _create_datasets(self) -> Dict[str, datasets.ImageFolder]:

        """

        Creates the datasets for the testing process

        :return: Dict[str, datasets.ImageFolder] - the datasets for the testing process

        """

        data_transforms = {

            config.TEST: config.PREPROCESSING_DEFAULT_TRANSFORMS

        }

        return {x: datasets.ImageFolder(os.path.join(self._data_dir, x), data_transforms[x])

                for x in data_transforms.keys()}

    def _create_dataloaders(self):

        """

        Creates the dataloaders for the datasets. Testing dataloaders do not shuffle the data and use the entire test

        dataset in a single batch

        """

        return {x: DataLoader(self._datasets[x], batch_size=self.dataset_sizes[x], shuffle=False,

                              num_workers=config.NUM_WORKERS) for x in self._datasets.keys()}
