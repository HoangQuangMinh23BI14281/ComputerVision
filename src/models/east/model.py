from torch import nn
import torch.nn.functional as F
import json
import torch
import math
import torchvision

class Extractor(nn.Module):
    def __init__(self, weights=None):
        super(Extractor, self).__init__()
        # Use torchvision's vgg16_bn with weights parameter
        vgg16_bn = torchvision.models.vgg16_bn(weights=weights)
        self.features = vgg16_bn.features


    def forward(self, x):
        out = list()
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                out.append(x)

        return out[1:]


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1024, out_channels=128, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=384, out_channels=64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=192, out_channels=32, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU()

        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU()

        self.init_weights()

    def forward(self, x):
        y = F.interpolate(x[3], scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[2]), dim=1)
        y = self.relu1(self.bn1(self.conv1(y)))
        y = self.relu2(self.bn2(self.conv2(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[1]), dim=1)
        y = self.relu3(self.bn3(self.conv3(y)))
        y = self.relu4(self.bn4(self.conv4(y)))

        y = F.interpolate(y, scale_factor=2, mode='bilinear', align_corners=True)
        y = torch.cat((y, x[0]), dim=1)
        y = self.relu5(self.bn5(self.conv5(y)))
        y = self.relu6(self.bn6(self.conv6(y)))

        y = self.relu7(self.bn7(self.conv7(y)))

        return y

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


class Output(nn.Module):
    def __init__(self):
        super(Output, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.sigmoid1 = nn.Sigmoid()

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=4, kernel_size=1)
        self.sigmoid2 = nn.Sigmoid()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1)
        self.sigmoid3 = nn.Sigmoid()

        self.scope = 512

        self.init_weights()

    def forward(self, x):
        score = self.sigmoid1(self.conv1(x)) 
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi

        geo = torch.cat((loc, angle), dim=1)

        return score, geo

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


class East(nn.Module):
    def __init__(self, weights=None):
        super(East, self).__init__()
        self.extractor = Extractor(weights)
        self.merge = Merge()
        self.output = Output()

    def forward(self, x):
        x = self.extractor(x)
        x = self.merge(x)
        x = self.output(x)

        return x


if __name__ == '__main__':
    from src.dataset.east_dataset import EastDataset
    # Example usage
    # dataset = EastDataset(image_paths[: 2], boxes[: 2])
    model = East()
    print("EAST model initialized")

