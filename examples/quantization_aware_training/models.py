import torch
import torch.nn as nn

class vanilla_cnn(nn.Module):
    """
    Contains basic conv ops for cifar-10 dataset
    """
    def __init__(self):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=8)
        )

        self.fcs = nn.Sequential(
            nn.Linear(4096,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self, x):

        x = self.convs(x)
        x = x.view(x.size(0),-1)
        x = self.fcs(x)
        return x 
