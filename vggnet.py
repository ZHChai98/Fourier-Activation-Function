import torch.nn as nn
from FCNN import Fourier
from FCNN import Sin

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels=1, actf='fourier', classnum=10):
        super(VGG, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        if actf == 'fourier':
            self.fourier = Fourier(N=30)
        else:
            self.relu = nn.ReLU(inplace=True)

        # self.sin = Sin()
        # self.sins = [Sin()] * 20
        # self.fouriers = [Fourier()] * 20

        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, classnum)

        self.model = vgg_name + '_' + actf
        # self.softmax=nn.Softmax()

        self.fc1 = nn.Linear(7680, 512)

    def forward(self, x):
        # layer1
        # print(x.size())
        out = self.conv1(x)
        out = self.bn1(out)
        if 'fourier' in self.model:
            out = self.fourier(out)
        else:
            out = self.relu(out)
        # others
        out = self.features(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.classifier(out)
        # out=self.softmax(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        if len(cfg) <= 12:
            in_channels = 128
        else:
            in_channels = 64
        # for x in cfg:
        #     if x == 'M':
        #         layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #         # Fourier(N=30)
        #     else:
        #         layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
        #                    nn.BatchNorm2d(x),
        #                    nn.ReLU(inplace=True)
        #                    # Fourier(N=30)
        #                    # self.fourier
        #                    ]
        #         in_channels = x

        for i in range(1, len(cfg)):
            if cfg[i] == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                # Fourier(N=30)

            # elif i in [0, ]:
            #     layers += [nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1),
            #                nn.BatchNorm2d(cfg[i]),
            #                self.fourier
            #                # self.fouriers[i]
            #                ]
            #     in_channels = cfg[i]

            else:
                layers += [nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=1),
                           nn.BatchNorm2d(cfg[i]),
                           nn.ReLU(inplace=True)
                           # Fourier(N=30)
                           # self.sins[i]

                           # self.fourier
                           ]
                in_channels = cfg[i]

        # layers += [self.fourier]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11(in_channels=1, actf='fourier', classnum=30):
    return VGG('VGG11', in_channels, actf, classnum)


def VGG13(in_channels=1, actf='fourier', classnum=30):
    return VGG('VGG13', in_channels, actf, classnum)


def VGG16(in_channels=1, actf='fourier', classnum=30):
    return VGG('VGG16', in_channels, actf, classnum)


def VGG19(in_channels=1, actf='fourier', classnum=30):
    return VGG('VGG19', in_channels, actf, classnum)
