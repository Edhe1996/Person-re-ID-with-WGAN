# This is the implementation of baseline CNN model for person re-id

import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable

from ResNet import resnet50
# from torchvision.models.resnet import resnet50


# Define the initialization methods for new fc and classification layer
def init_fc(m):
    classname = m.__class__.__name__

    # Fill the conv and linear layers with values according to the method
    # described in “Delving deep into rectifiers: Surpassing human-level
    # performance on ImageNet classification” - He, K. et al. (2015)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
    # Fill the batch normalization layer with random values (mean = 1 and std = 0.02)
    # and set the bias to 0
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_classifier(m):
    classname = m.__class__.__name__

    # Fill the linear layer with random values (mean = 1 and std = 0.02)
    # and set the bias to 0
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


# Define the new fully-connected and classifier layers
class NewBlock(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=True, relu=True, num_bottlenecks=512):
        super(NewBlock, self).__init__()
        new_fc = []
        new_fc += [nn.Linear(input_dim, num_bottlenecks)]
        new_fc += [nn.BatchNorm1d(num_bottlenecks)]
        if relu:
            new_fc += [nn.LeakyReLU(0.1)]
        if dropout:
            new_fc += [nn.Dropout(p=0.5)]

        new_fc = nn.Sequential(*new_fc)
        new_fc.apply(init_fc)

        new_classifier = []
        new_classifier += [nn.Linear(num_bottlenecks, num_classes)]
        new_classifier = nn.Sequential(*new_classifier)
        new_classifier.apply(init_classifier)

        self.new_fc = new_fc
        self.new_classifier = new_classifier

    def forward(self, x):
        x = self.new_fc(x)
        x = self.new_classifier(x)

        return x


# Define the base ResNet50 model
class ResNet50Baseline(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Baseline, self).__init__()
        # Use pretrained ResNet50 as baseline
        baseline = resnet50(pretrained=True)
        # Modify the average pooling layer
        baseline.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.model = baseline
        self.classifier = NewBlock(2048, num_classes)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.classifier(x)

        return x

