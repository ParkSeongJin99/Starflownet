import torch
import torch.nn as nn
import torch.nn.functional as F

class StarNet(nn.Module):
    def __init__(self):
        super(StarNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv6_1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv6_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6_4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv6_5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        #self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.fc1 = nn.Linear(512*16*16, 256)
        self.fc2 = nn.Linear(256, 1)

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6_1(x))
        x = F.relu(self.conv6_2(x))
        x = F.relu(self.conv6_3(x))
        x = F.relu(self.conv6_4(x))
        x = F.relu(self.conv6_5(x))

        x = self.global_max_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
