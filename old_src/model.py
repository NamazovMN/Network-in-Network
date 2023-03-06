import torch.nn as nn

import torch


class NIN(nn.Module):
    def __init__(self, num_classes, test = False):
        super(NIN, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=160,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=160, out_channels=96,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, stride = 2, padding = 1),
            nn.Dropout(0.7),


            nn.Conv2d(in_channels=96, out_channels=192,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn. MaxPool2d(3, stride = 2, padding = 1),
            nn.Dropout(0.7),


            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=192, out_channels=self.num_classes,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.num_classes),
            nn.ReLU(),
            # nn.AvgPool2d(7,stride = 1, padding = 0) => For MNIST Dataset
            nn.AvgPool2d(8,stride = 1, padding = 0)

        )

        self.initialize_weights()

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_classes)
        return x
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
