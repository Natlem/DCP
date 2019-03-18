import torch.nn as nn

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, img):
        output = self.features(img)
        output = output.view(img.size(0), self.classifier[0].in_features)
        output = self.classifier(output)
        return output