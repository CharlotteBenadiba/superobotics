import torch

import torchvision

from torch import nn, optim

from torchvision import transforms

# Common parameters

CLASS_NAMES_PATH = r"single_classifier_class_names.txt"

BEST_MODEL_PARAMS_PATH = r"single_classifier.pth"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Phases

TRAIN = 'train'

VAL = 'val'

TEST = 'test'

PHASES = [TRAIN, VAL, TEST]

# Preprocessing parameters

PREPROCESSING_DEFAULT_TRANSFORMS = transforms.Compose([

            transforms.Resize(256),

            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ])

PREPROCESSING_TRAIN_TRANSFORMS = transforms.Compose([

                transforms.RandomResizedCrop(224),

                transforms.RandomHorizontalFlip(),

                transforms.ToTensor(),

                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            ])

# Trainer parameters

INITIAL_MODEL_NUM_CLASSES = 100

BASE_PRETRAINED_MODEL = torchvision.models.resnet18(weights='IMAGENET1K_V1')

DATA_DIR_PATH = r"zz_classification_dataset_v8_uploaded_again"

# Training parameters

CRITERION = nn.CrossEntropyLoss()

OPTIMIZER = optim.SGD

SCHEDULER = optim.lr_scheduler.StepLR

NUM_EPOCHS = 3

BATCH_SIZE = 4

NUM_WORKERS = 4

# Optimizer parameters

MOMENTUM = 0.9

LEARNING_RATE = 0.001

# Scheduler parameters

STEP_SIZE = 7

GAMMA = 0.1
