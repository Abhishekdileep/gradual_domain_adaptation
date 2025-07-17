import torch
import torch.nn as nn 
import torch.nn.functional as F1

class Dcase_Baseline(nn.Module):
    def __init__(self, feature_vector_length=128, input_sequence_length=500, class_count=10):
        super(Dcase_Baseline, self).__init__()
        
        # Constants from original config
        self.convolution_kernel_size = 7
        self.convolution_dropout = 0.3
        
        # CNN Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=self.convolution_kernel_size,
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.dropout1 = nn.Dropout2d(p=self.convolution_dropout)
        
        # CNN Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=self.convolution_kernel_size,
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 100))
        self.dropout2 = nn.Dropout2d(p=self.convolution_dropout)
        
        # Calculate flattened size after conv layers
        # This needs to be computed based on input dimensions and pooling
        self.flattened_size = self._calculate_flattened_size(
            feature_vector_length, input_sequence_length
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(100, class_count)
        
        # Initialize weights
        self._initialize_weights()
    
    def _calculate_flattened_size(self, feature_length, sequence_length):
        """Calculate the size after convolution and pooling layers"""
        # Simulate forward pass to get dimensions
        x = torch.randn(1, 1, feature_length, sequence_length)
        
        # Conv1 + Pool1
        x = self.pool1(self.conv1(x))
        
        # Conv2 + Pool2
        x = self.pool2(self.conv2(x))
        
        return x.view(1, -1).size(1)
    
    def _initialize_weights(self):
        """Initialize weights similar to Keras glorot_uniform and uniform"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Glorot uniform initialization (Xavier uniform)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Uniform initialization for dense layers
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # CNN Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F1.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # CNN Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F1.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F1.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc2(x)
        x = F1.softmax(x, dim=1)
        
        return x


class Feature_extractor(nn.Module):
    def __init__(self, feature_vector_length=128, input_sequence_length=500, class_count=10):
        super(Feature_extractor, self).__init__()
        
        # Constants from original config
        self.convolution_kernel_size = 7
        self.convolution_dropout = 0.3
        
        # CNN Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=self.convolution_kernel_size,
            padding='same'
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(5, 5))
        self.dropout1 = nn.Dropout2d(p=self.convolution_dropout)
        
        # CNN Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=self.convolution_kernel_size,
            padding='same'
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(4, 100))
        self.dropout2 = nn.Dropout2d(p=self.convolution_dropout)
        self._initialize_weights()
        
        # Calculate flattened size after conv layers
        # This needs to be computed based on input dimensions and pooling

 
    def _initialize_weights(self):
        """Initialize weights similar to Keras glorot_uniform and uniform"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Glorot uniform initialization (Xavier uniform)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Uniform initialization for dense layers
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # CNN Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F1.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # CNN Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F1.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        return x 
    
class Classifier(nn.Module):

    def _initialize_weights(self):
        """Initialize weights similar to Keras glorot_uniform and uniform"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Glorot uniform initialization (Xavier uniform)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Uniform initialization for dense layers
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _calculate_flattened_size(self, feature_length, sequence_length ):
        """Calculate the size after convolution and pooling layers"""
        # Simulate forward pass to get dimensions
        x = torch.randn(1, 1, feature_length, sequence_length)
        
        # Conv1 + Pool1
        x = self.pool1(self.conv1(x))
        
        # Conv2 + Pool2
        x = self.pool2(self.conv2(x))
        
        return x.view(1, -1).size(1)
    

    def __init__(self, feature_vector_length, input_sequence_length, class_count):

        self.convolution_kernel_size = 7
        self.convolution_dropout = 0.3
        
        self.flattened_size = self._calculate_flattened_size(
            feature_vector_length, input_sequence_length
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 100)
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(100, class_count)
        
        # Initialize weights
        self._initialize_weights()

    def forward(self , x):
        x = self.fc1(x)
        x = F1.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc2(x)
        x = F1.softmax(x, dim=1)
        return x
    
class Wrapper(nn.Module):
    def __init__(self, net , classify ):
        
        torch.nn.Module.__init__(self  )
        # self.mel = mel
        self.net = net
        self.classify = classify 

    def forward(self, x):
        pass
        
