# This is the PyTorch implementation for ResNet

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

# pretrained models' urls
model_urls = {
    'ResNet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'ResNet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'ResNet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'ResNet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ResNet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


# Basic residual block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)  # save a little memory usage
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.downsample = downsample   # a method for downsampling
        self.stride = stride

    def forward(self, x):
        # (identity) shortcut connection
        residual = x

        # Here dim represents height x width, size means dim x depth (number of feature maps)
        # 3x3 convolutional layer with padding of 1
        # output_dim = (input_dim - kernel_size + 2 * padding) / stride + 1
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        # 3x3 convolutional layer with padding of 1
        # output_dim = (input_dim - kernel_size + 2 * padding) / stride + 1
        output = self.conv2(output)
        output = self.bn2(output)

        # Apply downsampling (if not None) to make sure the size remain the same
        # With this operation, the shortcut is no longer identity mapping
        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output


# Deeper bottleneck architecture (for more than 50 layers)
# for the concerns on the training time
class Bottleneck(nn.Module):
    # With Bottleneck architecture, there are 3 convolutional layers and the depth of
    # the last layer is 4 times of the first two layers
    expansion = 4

    def __init__(self, inplanes, outplanes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv3 = nn.Conv2d(outplanes, outplanes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # (identity) shortcut connection
        residual = x

        # Here dim represents height x width, size means dim x depth (number of feature maps)
        # 1x1 convolutional layer, output_dim = input_dim
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)

        # 3x3 convolutional layer with padding of 1
        # output_dim = (input_dim - kernel_size + 2 * padding) / stride + 1
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu(output)

        # 1x1 convolutional layer, output_dim = input_dim
        output = self.conv3(output)
        output = self.bn3(output)

        # Apply downsampling (if not None) to make sure the size remain the same
        # With this operation, the shortcut is no longer identity mapping
        if self.downsample is not None:
            residual = self.downsample(x)

        output += residual
        output = self.relu(output)

        return output


# Residual network construction
class ResNet(nn.Module):
    """
    Block_style: choose from BasicBlock and Bottleneck
    num_layers: a list contains numbers of every block to construct the final residual network
    num_classes: number of classes for the classification layer
    """
    def __init__(self, block_style, num_layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block_style, 64, num_layers[0])
        self.layer2 = self._make_layer(block_style, 128, num_layers[1], stride=2)
        self.layer3 = self._make_layer(block_style, 256, num_layers[2], stride=2)
        self.layer4 = self._make_layer(block_style, 512, num_layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block_style.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block_style, outplanes, num_layers, stride=1):
        downsample = None

        # Define the downsample method for shortcut connection to make sure the sizes of
        # tensors remain the same (shortcut no longer performs identity mapping)
        # This corresponds to option B of the original paper
        if stride != 1 or self.inplanes != outplanes * block_style.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, outplanes * block_style.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outplanes * block_style.expansion),
            )

        layers = [block_style(self.inplanes, outplanes, stride, downsample)]
        # The number of input channels of the remain parts changed
        self.inplanes = outplanes * block_style.expansion

        for i in range(1, num_layers):
            layers.append(block_style(self.inplanes, outplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 18-layers ResNet
def resnet18(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet18']))
    return model


# 34-layers ResNet
def resnet34(pretrained=False, **kwargs):

    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet34']))
    return model


# 50-layers ResNet
# Notice that the block style change to Bottleneck for the rest models
def resnet50(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet50']))
    return model


# 101-layers ResNet
def resnet101(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet101']))
    return model


# 152-layers ResNet
def resnet152(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['ResNet152']))
    return model
