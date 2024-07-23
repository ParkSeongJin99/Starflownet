import torch
import torch.nn as nn
import torch.nn.functional as F

class StarNet(nn.Module):
    def __init__(self):
        super(StarNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512 * 9 * 9, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.fc2(x)

        return x
