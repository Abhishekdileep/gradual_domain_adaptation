import torch
import torch.nn as nn 
import torch.nn.functional as F1


class SimpleConvNet(nn.Module):
    """
    A simple Convolutional Neural Network model, equivalent to the likely
    Keras model `simple_softmax_conv_model`.
    """
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Reshape input if it's flat
        if x.dim() == 2:
            x = x.view(-1, 28, 28)
        # Add channel dimension if it's missing
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        out = self.pool1(self.relu1(self.conv1(x)))
        out = self.pool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1) # Flatten
        out = self.relu3(self.fc1(out))
        out = self.fc2(out)
        return out

class PyTorchConvModel(nn.Module):
    def __init__(self, num_labels, hidden_nodes=32, input_shape=(1, 28, 28)):
        """
        Initializes the layers of the model.

        Args:
            num_labels (int): The number of output classes.
            hidden_nodes (int): The number of filters in the convolutional layers.
            input_shape (tuple): The shape of the input tensor (C, H, W).
                                 Note: PyTorch uses Channels-First format.
        """
        super(PyTorchConvModel, self).__init__()
        
        # Note: Keras uses (H, W, C) while PyTorch uses (C, H, W)
        in_channels = input_shape[0]

        # --- Convolutional Layers ---
        # The Keras 'same' padding with a stride of 2 is approximated here
        # by setting padding=2. This keeps the output dimensions consistent.
        self.conv1 = nn.Conv2d(in_channels, hidden_nodes, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(hidden_nodes, hidden_nodes, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(hidden_nodes, hidden_nodes, kernel_size=5, stride=2, padding=2)
        
        # --- Regularization and Normalization Layers ---
        self.dropout = nn.Dropout(0.5)
        # BatchNorm2d is used for 2D inputs (i.e., after conv layers)
        self.batch_norm = nn.BatchNorm2d(hidden_nodes)

        # --- Classifier Layers ---
        self.flatten = nn.Flatten()
        
        # In PyTorch, you often need to calculate the input size for the first
        # linear layer manually.
        # Input: 28x28
        # After conv1 (stride 2): ceil(28/2) = 14x14
        # After conv2 (stride 2): ceil(14/2) = 7x7
        # After conv3 (stride 2): ceil(7/2) = 4x4
        # Flattened size = channels * height * width
        flattened_size = hidden_nodes * 4 * 4
        
        self.fc = nn.Linear(flattened_size, num_labels)

    def forward(self, x):
        """Defines the forward pass of the model."""
        # Convolutional block
        x = F1.relu(self.conv1(x))
        x = F1.relu(self.conv2(x))
        x = F1.relu(self.conv3(x))
        
        # Regularization and Normalization
        x = self.dropout(x)
        x = self.batch_norm(x)
        
        # Classifier
        x = self.flatten(x)
        logits = self.fc(x)
        
        # NOTE: The final Softmax activation is NOT applied here.
        # In PyTorch, loss functions like `nn.CrossEntropyLoss` expect raw logits
        # as they combine LogSoftmax and the loss calculation for better
        # numerical stability. You would apply softmax only during inference.
        
        return logits

def simple_softmax_conv_model(num_labels=10, hidden_nodes=32, input_shape=(1 ,28, 28)):
    '''
    Wrapper function similar to the Keras one  
    '''
    return PyTorchConvModel(num_labels, hidden_nodes, input_shape)
