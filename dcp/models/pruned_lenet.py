import torch.nn as nn
import math
import numpy as np

class Pruned_LeNet5(nn.Module):

    def __init__(self, pruning_rate):
        super(Pruned_LeNet5, self).__init__()

        o_out_channels = np.array([6, 16, 120])
        p_out_channels = (o_out_channels * pruning_rate).astype(np.int)

        self.features = nn.Sequential(
            nn.Conv2d(1, p_out_channels[0], kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(6, p_out_channels[1], kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, p_out_channels[2], kernel_size=(5, 5)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(p_out_channels[2], 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        output = self.features(img)
        output = output.view(img.size(0), self.classifier[0].in_features)
        output = self.classifier(output)
        return output