import torch
import torch.nn as nn
import torch.nn.functional as F
from FCNN import Fourier


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        out = F.relu(out)
        return out


class FourierBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(FourierBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.fourier1 = Fourier(N=30)
        self.fourier2 = Fourier(N=30)
        self.fourier3 = Fourier(N=30)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = self.fourier1(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))

        # out = self.fourier2(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)

        out = self.fourier3(out)
        # out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, resnetname, actf, num_classes=10, fourierblock=FourierBottleneck):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        if actf == 'fourier':
            self.fourier = Fourier(N=30)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer1 = self._make_fourier_layer(block, 64, num_blocks[0], stride=1, fourierblock=fourierblock)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.model = resnetname + '_' + actf

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _make_fourier_layer(self, block, planes, num_blocks, stride, fourierblock):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides[:-1]:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        layers.append(fourierblock(self.in_planes, planes, strides[-1]))
        self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        if 'fourier' in self.model:
            out = self.fourier(self.bn1(self.conv1(x)))
        else:
            out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(actf='fourier'):
    return ResNet(BasicBlock, [2, 2, 2, 2], 'res18', actf=actf)


def ResNet34(actf='fourier'):
    return ResNet(BasicBlock, [3, 4, 6, 3], 'res34', actf=actf)


def ResNet50(actf='fourier'):
    return ResNet(Bottleneck, [3, 4, 6, 3], 'res50', actf=actf)


def ResNet101(actf='fourier'):
    return ResNet(Bottleneck, [3, 4, 23, 3], 'res101', actf=actf)


def ResNet152(actf='fourier'):
    return ResNet(Bottleneck, [3, 8, 36, 3],'res152', actf=actf)


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
