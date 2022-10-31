import torch
import torch.nn as nn
from spectral import SpectralNorm

class Self_Attn(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim       
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps(B X C X W X H)
        returns :
            out : self attention value + input feature 
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1) # B X C X(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)        
        out = self.gamma*out + x

        return out, attention


class Generator(nn.Module):
    def __init__(self, z_dim, c_dim):
        super(Generator,self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.conditionExpand = nn.Sequential(
            nn.Linear(24, c_dim),
            nn.ReLU()
        )
        kernel_size = (4, 4)
        channels = [z_dim+c_dim, 512, 256, 128, 64]
        paddings = [(0,0), (1,1), (1,1), (1,1)]
        for i in range(1, len(channels)):
            setattr(self, 'l'+str(i), nn.Sequential(
                nn.ConvTranspose2d(channels[i-1], channels[i], kernel_size, stride=(2,2), padding=paddings[i-1]),
                nn.BatchNorm2d(channels[i]),
                nn.ReLU()
            ))
        self.l5 = nn.ConvTranspose2d(64, 3, kernel_size, stride=(2,2), padding=(1,1))
        self.tanh = nn.Tanh()

        self.attn1 = Self_Attn(128)
        self.attn2 = Self_Attn(64)

    def forward(self, z, c):
        z = z.view(-1, self.z_dim, 1, 1)
        c = self.conditionExpand(c).view(-1, self.c_dim, 1, 1)
        out = torch.cat((z,c), dim=1)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out, _ = self.attn1(out)
        out = self.l4(out)
        out, _ = self.attn2(out)
        out = self.l5(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.H, self.W, self.C = img_shape
        self.conditionExpand = nn.Sequential(
            nn.Linear(24, self.H*self.W*1),
            nn.LeakyReLU()
        )
        kernel_size = (4, 4)
        channels = [4, 64, 128, 256, 512]
        for i in range(1,len(channels)):
            setattr(self, 'l'+str(i),nn.Sequential(
                SpectralNorm(nn.Conv2d(channels[i-1], channels[i], kernel_size, stride=(2,2), padding=(1,1))),
                nn.BatchNorm2d(channels[i]),
                nn.LeakyReLU(0.1)
            ))
        self.l5 = nn.Conv2d(512, 1, kernel_size, stride=(1,1))
        self.attn1 = Self_Attn(256)
        self.attn2 = Self_Attn(512)

    def forward(self, X, c):
        c = self.conditionExpand(c).view(-1, 1, 64, 64)
        out = torch.cat((X,c), dim=1)
        out = self.l1(out)
        out = self.l2(out)
        out = self.l3(out)
        out, _ = self.attn1(out)
        out = self.l4(out)
        out, _ = self.attn2(out)
        out = self.l5(out)

        return out