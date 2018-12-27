import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, input, reduction=4):
        mid_channel = int(input / reduction)
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(input, mid_channel),
                nn.LeakyReLU(inplace=True),
                nn.Linear(mid_channel, input),
                nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class ConvBlock(nn.Module):
    def __init__(self,input):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input,out_channels=input,kernel_size=(1,1,1))
        self.bn1 = nn.BatchNorm3d(num_features=input)
        self.conv2 = nn.Conv3d(in_channels=input, out_channels=input, kernel_size=(3,3,3),padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(num_features=input)
        self.conv3 = nn.Conv3d(in_channels=input, out_channels=input, kernel_size=(1,1,1))
        self.bn3 = nn.BatchNorm3d(num_features=input)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        return x

class SE_ResBlock(nn.Module):
    def __init__(self,channel):
        super(SE_ResBlock, self).__init__()
        self.conv = ConvBlock(input=channel)
        self.se_block = SELayer(input=channel)
        self.bn = nn.BatchNorm3d(num_features=channel)

    def forward(self,x):
        x1 = self.conv(x)
        x1 = self.se_block(x1)
        x = x + x1
        x = self.bn(x)
        x = F.relu(x)
        return x



def main():
    data = torch.Tensor(1,16,512,512,4).cuda()
    print(data.size())
    net = SE_ResBlock(channel=16)
    net.cuda()
    result = net(data)
    print(result.size())

if __name__ == '__main__':
    main()