# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class upsample_block(nn.Module):
    # https://www.cnblogs.com/kk17/p/10094160.html
    def __init__(self, in_channels, out_channels):
        super(upsample_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              3, stride=1, padding=1)
        self.shuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        return self.prelu(self.shuffler(self.conv(x)))


class Transformer(torch.nn.Module):
    # https://zhuanlan.zhihu.com/p/33345791
    # https://zhuanlan.zhihu.com/p/48508221
    # https://zhuanlan.zhihu.com/p/340149804
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(Transformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.K = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)

        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)


    def forward(self, x):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)
        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)

        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = x + W
        return output

class ChannelAttention(nn.Module):
    # https://zhuanlan.zhihu.com/p/32702350
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class CARB(nn.Module):   # Channel attention residual block
    def __init__(self, nChannels,reduction=16):
        super(CARB, self).__init__()
        self.relu1 = nn.PReLU()
        self.relu2 = nn.PReLU()
        self.conv1 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.conv4 = nn.Conv2d(nChannels, nChannels, kernel_size=3, padding=1, bias=False)
        self.ca1 = ChannelAttention(nChannels, reduction)
        self.ca2 = ChannelAttention(nChannels, reduction)

    def forward(self, x):
        out = self.conv2(self.relu1(self.conv1(x)))
        b1 = self.ca1(out) +x

        out = self.relu2(self.conv3(b1))
        b2 = self.ca2(self.conv4(out)) +b1
        return b2


class OutConv(nn.Module):
    def __init__(self,ichannels=64):
        super(OutConv, self).__init__()
        self.conv_tail = nn.Conv2d(in_channels=ichannels, out_channels=ichannels, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.conv_out = nn.Conv2d(ichannels, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, f):
        tail = self.relu(self.conv_tail(f))
        out = self.conv_out(tail)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_input = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.PReLU()
        self.conv_input2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.tf_block1 = Transformer(64, 64, 3, 1, 1)
        self.tf_block2 = Transformer(64, 64, 3, 1, 1)
        self.main_block = nn.Sequential(
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64)
        )
        self.main_block2 = nn.Sequential(
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64)
        )
        self.main_block3 = nn.Sequential(
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64),
            CARB(64)
        )
        self.u1 = upsample_block(64, 64*4)
        self.u2 = upsample_block(64, 64*4)
        self.u1e = upsample_block(64, 64*4)
        self.u2e = upsample_block(64, 64*4)
        self.out = OutConv(64)


    def forward(self, x):
        out = self.relu(self.conv_input(x))
        conv1 = self.conv_input2(out)
        # tf1 = self.tf_block1(conv1)
        main = self.main_block(conv1)
        tf1 = self.tf_block1(main)
        main = self.main_block2(tf1)
        tf2 = self.tf_block2(main)
        main = self.main_block3(tf2)
        # fusion = main+out    #https://blog.csdn.net/weixin_43840215/article/details/89815120
        u1 = self.u1(main)
        u2 = self.u2(u1)
        u1e = self.u1e(out)
        u2e = self.u2e(u1e)
        res = self.out(u2+u2e)
        return res


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)  # over smooth
        loss = torch.sum(error)
        return loss

class L1_Sobel_loss(nn.Module):
    """Sobel+L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Sobel_loss, self).__init__()
        self.eps = 1e-6
        self.sobel = Sobel2D()

    def forward(self, X, Y):
        sobel_x = self.sobel(X)
        sobel_y = self.sobel(Y)
        diff = torch.add(sobel_x, -sobel_y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss


class Sobel2D(nn.Module):
    def __init__(self):
        super(Sobel2D, self).__init__()
        self.conv_op_x = nn.Conv2d(1, 1, 3,stride=1, padding=1, bias=False)
        self.conv_op_y = nn.Conv2d(1, 1, 3,stride=1, padding=1, bias=False)

        sobel_kernel_x = np.array([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]], dtype='float32')
        sobel_kernel_y = np.array([[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]], dtype='float32')
        sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
        sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))

        self.conv_op_x.weight.data = torch.from_numpy(sobel_kernel_x)
        self.conv_op_y.weight.data = torch.from_numpy(sobel_kernel_y)
        self.conv_op_x.weight.requires_grad = False
        self.conv_op_y.weight.requires_grad = False

    def forward(self, x):
        edge_x = self.conv_op_x(x)
        edge_y = self.conv_op_y(x)
        edge = torch.abs(edge_x) + torch.abs(edge_y)
        return edge
