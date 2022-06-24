import numpy as np
import torch as th
from torch import nn
import torchvision

def echo2depth_loss(output, target):
    loss = th.mean(th.log(1 + th.abs(output - target)))
    return loss

class echo2depth(nn.Module):
    def __init__(self, num_channels=8):
        super().__init__()
        
        self.conv_1 = nn.Conv2d(num_channels, 8, kernel_size=(8, 8), stride=8)
        self.bn_1 = nn.BatchNorm2d(8)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(8, 128, kernel_size=(4, 4), stride=4)
        self.bn_2 = nn.BatchNorm2d(128)
        self.relu_2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=3)
        self.bn_3 = nn.BatchNorm2d(512)
        self.relu_3 = nn.ReLU()

        self.convt_1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.bn_t_1 = nn.BatchNorm2d(256)
        self.relu_t_1 = nn.ReLU()
        self.convt_2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.bn_t_2 = nn.BatchNorm2d(128)
        self.relu_t_2 = nn.ReLU()
        self.convt_3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.bn_t_3 = nn.BatchNorm2d(64)
        self.relu_t_3 = nn.ReLU()
        self.convt_4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.bn_t_4 = nn.BatchNorm2d(32)
        self.relu_t_4 = nn.ReLU()
        self.convt_5 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.bn_t_5 = nn.BatchNorm2d(16)
        self.relu_t_5 = nn.ReLU()
        self.convt_6 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.bn_t_6 = nn.BatchNorm2d(8)
        self.relu_t_6 = nn.ReLU()
        self.convt_7 = nn.ConvTranspose2d(8, 1, 2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)

        x = x.view(-1, 512, 1, 1)

        x = self.convt_1(x)
        x = self.bn_t_1(x)
        x = self.relu_t_1(x)

        x = self.convt_2(x)
        x = self.bn_t_2(x)
        x = self.relu_t_2(x)

        x = self.convt_3(x)
        x = self.bn_t_3(x)
        x = self.relu_t_3(x)

        x = self.convt_4(x)
        x = self.bn_t_4(x)
        x = self.relu_t_4(x)

        x = self.convt_5(x)
        x = self.bn_t_5(x)
        x = self.relu_t_5(x)

        x = self.convt_6(x)
        x = self.bn_t_6(x)
        x = self.relu_t_6(x)
        
        x = self.convt_7(x)
        x = self.sigmoid(x)

        return(x)
        
def main():
    model = echo2depth(num_channels=8)
    num_epochs = 100
    x = th.rand((16, 8, 128, 128))
    gt = th.rand((16, 1, 128, 128))

    y = model(x)

    loss = echo2depth_loss(y, gt)

    print(loss)


if __name__ == "__main__":
    main()