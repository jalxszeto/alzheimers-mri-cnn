"""CNN classifier for brain MRI slices (Alzheimer's severity)."""
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x_conv1 = F.relu(self.conv1(x))
        x_conv2 = F.relu(self.conv2(x_conv1))
        x_norm = self.norm(x_conv2)
        x_maxpool = self.maxpool(x_norm)
        return x_maxpool
        
class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_fc = F.relu(self.fc(x))
        x_norm = self.norm(x_fc)
        x_dropout = self.dropout(x_norm)
        return x_dropout

class CNNModel(nn.Module):
    def __init__(self, image_size, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_block1 = ConvBlock(16, 32)
        self.conv_block2 = ConvBlock(32, 64)

        self.conv_block3 = ConvBlock(64, 128)
        self.dropout1 = nn.Dropout(0.2)

        self.conv_block4 = ConvBlock(128, 256)
        self.dropout2 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        # The initial maxpool plus four ConvBlocks halve the spatial size 5 times
        conv_output_size = [image_size[index] // 2**5 for index in range(2)]
        flattened_size = 256 * conv_output_size[0] * conv_output_size[1]
        
        self.dense_block1 = DenseBlock(flattened_size, 512, 0.7)
        self.dense_block2 = DenseBlock(512, 128, 0.5)
        self.dense_block3 = DenseBlock(128, 64, 0.3)

        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x_conv1 = F.relu(self.conv1(x))
        x_conv2 = F.relu(self.conv2(x_conv1))
        x_maxpool = self.maxpool(x_conv2)

        x_conv_block1 = self.conv_block1(x_maxpool)
        x_conv_block2 = self.conv_block2(x_conv_block1)

        x_conv_block3 = self.conv_block3(x_conv_block2)
        x_dropout1 = self.dropout1(x_conv_block3)

        x_conv_block4 = self.conv_block4(x_dropout1)
        x_dropout2 = self.dropout2(x_conv_block4)

        x_flatten = self.flatten(x_dropout2)
        x_dense_block1 = self.dense_block1(x_flatten)
        x_dense_block2 = self.dense_block2(x_dense_block1)
        x_dense_block3 = self.dense_block3(x_dense_block2)

        return self.fc(x_dense_block3)
