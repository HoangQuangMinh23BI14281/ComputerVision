import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import DeformConv2d

class FPN(nn.Module):
    def __init__(self, in_channels, inner_channels=256):
        super(FPN, self).__init__()
        self.conv_out = inner_channels
        self.in2_conv = nn.Conv2d(in_channels[0], self.conv_out, 1)
        self.in3_conv = nn.Conv2d(in_channels[1], self.conv_out, 1)
        self.in4_conv = nn.Conv2d(in_channels[2], self.conv_out, 1)
        self.in5_conv = nn.Conv2d(in_channels[3], self.conv_out, 1)
        
        self.p5_conv = nn.Conv2d(self.conv_out, self.conv_out // 4, 3, padding=1)
        self.p4_conv = nn.Conv2d(self.conv_out, self.conv_out // 4, 3, padding=1)
        self.p3_conv = nn.Conv2d(self.conv_out, self.conv_out // 4, 3, padding=1)
        self.p2_conv = nn.Conv2d(self.conv_out, self.conv_out // 4, 3, padding=1)

    def forward(self, features):
        c2, c3, c4, c5 = features
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.interpolate(in5, size=in4.shape[2:], mode='bilinear', align_corners=False)
        out3 = in3 + F.interpolate(out4, size=in3.shape[2:], mode='bilinear', align_corners=False)
        out2 = in2 + F.interpolate(out3, size=in2.shape[2:], mode='bilinear', align_corners=False)

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        p5 = F.interpolate(p5, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p4 = F.interpolate(p4, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p3 = F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=False)

        out = torch.cat([p5, p4, p3, p2], dim=1)
        return out

class DBHead(nn.Module):
    def __init__(self, in_channels, k=50):
        super(DBHead, self).__init__()
        self.k = k
        self.binarize = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 2, 2),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels // 4, 1, 2, 2),
            nn.Sigmoid()
        )

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))

    def forward(self, x):
        shrink_maps = self.binarize(x)
        if not self.training:
            return shrink_maps
        threshold_maps = self.thresh(x)
        # Nới rộng biên độ threshold map (0.3 đến 0.7)
        threshold_maps = threshold_maps * 0.4 + 0.3
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        # Returns [batch_size, 3, H, W] = [Probability map, Threshold map, Approximate binary map]
        return torch.cat((shrink_maps, threshold_maps, binary_maps), dim=1)

class DeformConvWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, groups=1):
        super(DeformConvWrapper, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
            
        self.offset_conv = nn.Conv2d(in_channels, 2 * groups * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.mask_conv = nn.Conv2d(in_channels, groups * kernel_size[0] * kernel_size[1], kernel_size=kernel_size, stride=stride, padding=padding)
        self.dcn = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        self._init_offset()

    def _init_offset(self):
        nn.init.constant_(self.offset_conv.weight, 0)
        if self.offset_conv.bias is not None:
            nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.mask_conv.weight, 0)
        if self.mask_conv.bias is not None:
            nn.init.constant_(self.mask_conv.bias, 0)

    def forward(self, x):
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        return self.dcn(x, offset, mask)

def apply_dcn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and (child.kernel_size == (3, 3) or child.kernel_size == 3):
            in_channels = child.in_channels
            out_channels = child.out_channels
            stride = child.stride[0] if isinstance(child.stride, tuple) else child.stride
            padding = child.padding[0] if isinstance(child.padding, tuple) else child.padding
            groups = child.groups
            bias = child.bias is not None
            dcn = DeformConvWrapper(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)
            setattr(module, name, dcn)
        else:
            apply_dcn(child)

class DBNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DBNet, self).__init__()
        # Backbone (ResNet18 / ResNet50)
        backbone = models.resnet18(pretrained=pretrained)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Thêm DCN vào c3, c4, c5 (layer2, layer3, layer4) của ResNet
        apply_dcn(self.layer2)
        apply_dcn(self.layer3)
        apply_dcn(self.layer4)
        
        # Neck (FPN)
        in_channels = [64, 128, 256, 512]
        self.neck = FPN(in_channels, inner_channels=256)
        
        # Head (Differentiable Binarization)
        self.head = DBHead(256)

        # Khởi tạo trọng số cho Neck và Head
        self._init_weights()

    def _init_weights(self):
        for m in self.neck.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Extract features from backbone
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # Pass through FPN
        features = self.neck([c2, c3, c4, c5])
        
        # Prediction
        out = self.head(features)
        return out

if __name__ == '__main__':
    # Test model shape
    model = DBNet(pretrained=False)
    dummy_input = torch.randn(2, 3, 640, 640)
    out = model(dummy_input)
    print("DBNet output shape:", out.shape) # Expect [2, 3, 640, 640] if training
