import torch
import torch.nn as nn 
import torch.nn.functional as F1


class AlexNetHalf(nn.Module):
    def __init__(self, input_shape= (1, 40 , 500), num_classes=10):
        super(AlexNetHalf, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 48, kernel_size=11, stride=(2, 3), padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.BatchNorm2d(48),
            
            nn.Conv2d(48, 128, kernel_size=5, stride=(2, 3), padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, stride=1, padding=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=(1, 2)),
            nn.BatchNorm2d(128),
        )
        
        # The following Flatten and Linear layers need to have their input size calculated
        # after a forward pass with a sample tensor.
        # We will use a dummy forward pass to determine the flattened size.
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            flattened_size = self.features(dummy_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
