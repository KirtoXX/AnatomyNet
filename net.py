from basic_layer import SE_ResBlock
import torch.nn as nn
import torch.nn.functional as F
import torch

class Conv3x3(nn.Module):
    def __init__(self,input,output):
        super(Conv3x3, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=input,out_channels=output,kernel_size=(3,3,3),padding=(1,1,1))
        self.bn1 = nn.BatchNorm3d(num_features=output)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        return x

class AnatomyNet(nn.Module):
    def __init__(self,classes=10):
        super(AnatomyNet,self).__init__()
        # stage 0
        self.conv0 = Conv3x3(1,32)
        self.pool = nn.MaxPool3d(kernel_size=(2,2,2))
        # stage 1
        self.se1 = SE_ResBlock(32)
        self.conv1 = Conv3x3(32,40)
        # stage 2
        self.se2 = SE_ResBlock(40)
        self.conv2 = Conv3x3(40, 48)
        # stage 3
        self.se3 = SE_ResBlock(48)
        self.conv3 = Conv3x3(48, 56)
        # stage 4
        self.se4 = SE_ResBlock(56)
        # stage 5
        self.conv5 = Conv3x3(48+56,48)
        self.conv5_1 = Conv3x3(48,48)
        self.se5 = SE_ResBlock(48)
        # stage 6
        self.conv6 = Conv3x3(40 +48, 40)
        self.conv6_1 = Conv3x3(40,40)
        self.se6 = SE_ResBlock(40)
        # stage 7
        self.conv7 = Conv3x3(32 + 40, 32)
        self.conv7_1 = Conv3x3(32, 32)
        self.Deconv = nn.ConvTranspose3d(in_channels=32,out_channels=16,kernel_size=(2,2,2),stride=2)
        # stage 8
        self.conv8 = Conv3x3(16+1,16)
        self.conv8_1 = Conv3x3(16,16)
        self.decoder = nn.Conv3d(in_channels=16,out_channels=classes,kernel_size=(3,3,3),padding=(1,1,1))

    def forward(self,x):
        #stage 0
        x0 = self.conv0(x)
        x0 = self.pool(x0)
        # stage1
        x1 = self.se1(x0)
        x1 = self.conv1(x1)
        # stage2
        x2 = self.se2(x1)
        x2 = self.conv2(x2)
        # stage3
        x3 = self.se3(x2)
        x3 = self.conv3(x3)
        # stage4
        x4 = self.se4(x3)
        # stage5
        x5 = torch.cat((x4,x2),1)
        x5 = self.conv5(x5)
        x5 = self.conv5_1(x5)
        x5 = self.se5(x5)
        # stage6
        x6 = torch.cat((x5, x1), 1)
        x6 = self.conv6(x6)
        x6 = self.conv6_1(x6)
        x6 = self.se6(x6)
        # stage7
        x7 = torch.cat((x6, x0), 1)
        x7 = self.conv7(x7)
        x7 = self.conv7_1(x7)
        x7 = self.Deconv(x7)
        x7 = F.relu(x7)
        # stage8
        x8 = torch.cat((x7,x), 1)
        x8 = self.conv8(x8)
        x8 = self.conv8_1(x8)
        x8 = self.decoder(x8)
        x8 = F.softmax(x8,dim=1)

        return x8

def main():
    net = AnatomyNet(classes=10)
    data = torch.Tensor(1,1,512,512,4)
    data = data.cuda()
    net.cuda()
    result = net(data)
    print(result.size())
    torch.save(net.state_dict(),'123.pick')

if __name__ == '__main__':
    main()




