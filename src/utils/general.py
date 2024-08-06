import os
import torch
import torchvision.models as models
import torch.nn as nn
from models import SSWQ

def increment_path(p):
    new_p = p
    i = 0
    while True:
        if not os.path.exists(new_p):
            return new_p

        new_p = p+f"_{i}"
        i+=1


def load_model(model_name, num_classes):
    # ResNet18
    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    # AlexNet 
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    # VGG16 
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Linear(in_features, num_classes)
    # ResNet50
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    # EfficientNet 
    elif model_name == "efficientnet_b3":
        model = efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
        model.classifier.fc = nn.Linear(model.classifier.fc.in_features, num_classes)
    # ResNet34
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    # SSWQ
    elif model_name == "SSWQ":
        model = SSWQ(num_classes)

    return model

