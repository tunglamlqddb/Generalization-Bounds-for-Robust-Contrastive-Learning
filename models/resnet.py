'''
ResNet with dual BN.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .base import GeneralSSL, ParamSequential, NormalizeByChannelMeanStd
from .lenet import Learner

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)])
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)])

        self.shortcut = (stride != 1) or (in_planes != self.expansion * planes)
        if self.shortcut:
            self.conv3 = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.bn3 = nn.ModuleList([nn.BatchNorm2d(self.expansion * planes), nn.BatchNorm2d(self.expansion * planes)])

    def forward(self, x, adv=False):
        out = F.relu(self.bn1[adv](self.conv1(x)))
        out = self.bn2[adv](self.conv2(out))

        if self.shortcut:
            out += self.bn3[adv](self.conv3(x))
    
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)])

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(planes), nn.BatchNorm2d(planes)])

        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(self.expansion * planes), nn.BatchNorm2d(self.expansion * planes)])

        self.shortcut = (stride != 1) or (in_planes != self.expansion * planes)
        if self.shortcut:
            self.conv4 = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            self.bn4 = nn.ModuleList([nn.BatchNorm2d(self.expansion * planes), nn.BatchNorm2d(self.expansion * planes)])

    def forward(self, x, adv=False):
        out = F.relu(self.bn1[adv](self.conv1(x)))
        out = F.relu(self.bn2[adv](self.conv2(out)))
        out = self.bn3[adv](self.conv3(out))

        if self.shortcut:
            out += self.bn4[adv](self.conv4(x))

        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False, full=False):
        '''
        @input full : whether to use the ImageNet version of ResNet
        '''
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if full:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.bn1 = nn.ModuleList([nn.BatchNorm2d(64), nn.BatchNorm2d(64)])
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3[0].weight, 0)
                    nn.init.constant_(m.bn3[1].weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2[0].weight, 0)
                    nn.init.constant_(m.bn2[1].weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return ParamSequential(*layers)

    def forward(self, x, adv=False, normalize=False):
        adv = int(adv)
        # if normalize:
        x = self.normalize(x)    # check if normalize here improve anything. Assume normalizr=True for now

        out = self.conv1(x)
        out = self.bn1[adv](out)
        out = F.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out, adv=adv)
        out = self.layer2(out, adv=adv)
        out = self.layer3(out, adv=adv)
        out = self.layer4(out, adv=adv)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))


class AlexNet(nn.Module):
    def __init__(self, full=False):
        super(AlexNet, self).__init__()
        if full: 
            self.input_size = 64
        else:
            self.input_size = 32
        self.ksize=[]
        self.conv1 = nn.Conv2d(3, 64, 4, bias=False)
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(64), nn.BatchNorm2d(64)])
        s=compute_conv_output_size(self.input_size,4)
        s=s//2
        self.ksize.append(4)
        self.conv2 = nn.Conv2d(64, 128, 3, bias=False)
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(128), nn.BatchNorm2d(128)])
        s=compute_conv_output_size(s,3)
        s=s//2
        self.ksize.append(3)
        self.conv3 = nn.Conv2d(128, 256, 2, bias=False)
        self.bn3 = nn.ModuleList([nn.BatchNorm2d(256),nn.BatchNorm2d(256)])
        s=compute_conv_output_size(s,2)
        s=s//2
        self.smid=s
        self.ksize.append(2)
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()
        self.drop1=torch.nn.Dropout(0.2)
        self.drop2=torch.nn.Dropout(0.5)

        self.fc1 = nn.Linear(256*self.smid*self.smid,2048, bias=False)
        self.bn4 = nn.ModuleList([nn.BatchNorm1d(2048), nn.BatchNorm1d(2048)])
        self.fc2 = nn.Linear(2048,2048, bias=False)
        self.bn5 = nn.ModuleList([nn.BatchNorm1d(2048), nn.BatchNorm1d(2048)])
        
        
    def forward(self, x, adv=False):
        adv = int(adv)

        bsz = x.size(0)
        x = self.conv1(x)
        x = self.maxpool(self.drop1(self.relu(self.bn1[adv](x))))

        x = self.conv2(x)
        x = self.maxpool(self.drop1(self.relu(self.bn2[adv](x))))

        x = self.conv3(x)
        x = self.maxpool(self.drop2(self.relu(self.bn3[adv](x))))
        
        x=x.view(bsz,-1)
        x = self.fc1(x)
        x = self.drop2(self.relu(self.bn4[adv](x)))

        x = self.fc2(x)
        x = self.drop2(self.relu(self.bn5[adv](x)))
            
        return x


def alexnet(**kwargs):
    # from torchvision import models
    # model_class = getattr(models, 'vgg16')
    # model = MyVGG(pretrained=False, num_classes=512)
    model = AlexNet(**kwargs)
    return model

def lenet(**kwargs):
    channels = 160
    cfg = [
        ('conv2d-nbias', [channels, 3, 3, 3, 2, 1], ''),
        ('relu', [True], ''),

        ('conv2d-nbias', [channels, channels, 3, 3, 2, 1], ''),
        ('relu', [True], ''),

        ('conv2d-nbias', [channels, channels, 3, 3, 2, 1], ''),
        ('relu', [True], ''),

        ('conv2d-nbias', [channels, channels, 3, 3, 2, 1], ''),
        ('relu', [True], ''),

        ('flatten', [], ''),
        ('rep', [], ''),

        ('linear-nbias', [640, 16 * channels], ''),
        ('relu', [True], ''),

        ('linear-nbias', [640, 640], ''),
    ]
    model = Learner(cfg)
    return model


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'alexnet': [alexnet, 2048],
    'lenet': [lenet, 640]
}


class ResNetSSL(GeneralSSL):
    """backbone + projection head"""
    def __init__(self, name='resnet50', feat_dim=128, num_classes=10, full=False):
        super(ResNetSSL, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun(full=full)
        
        # projection head
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim)
        )
        
        # classifier head
        self.fc = nn.Linear(dim_in, num_classes)
