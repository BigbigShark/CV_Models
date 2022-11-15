import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, cfg: list, num_classes=10, init_weights=False):
        super(VGG, self).__init__()

        self.layers = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.layers(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x (512x7x7)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
                in_channels = v
        return nn.Sequential(*layers)

cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # M: MaxPooling
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11():
    return VGG(cfgs['vgg11'])

def vgg13():
    return VGG(cfgs['vgg13'])

def vgg16():
    return VGG(cfgs['vgg16'])

def vgg19():
    return VGG(cfgs['vgg19'])

# def VGGNet(model_name="vgg16", **kwargs):
#     assert  model_name in cfgs, "Warning: Model {} not in cfgs dict!".format(model_name)
#     cfg = cfgs[model_name]
#     model = VGG(cfg, **kwargs)
#
#     return model