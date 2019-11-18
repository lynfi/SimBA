'''Read models.'''
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import *
from normal import *


def readmodel(name, device):
    if name == 'resnet18':
        net = ResNet18()
    elif name == 'vgg16':
        net = VGG('VGG16')
    elif name == 'googlenet':
        net = GoogLeNet()
    elif name == 'mobilenetv2':
        net = MobileNetV2()
    elif name == 'preactresnet18':
        net = PreActResNet18()
    elif name == 'densenet_cifar':
        net = densenet_cifar()

    net = nn.Sequential(NormalizeLayer(), net)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    return net

def loadmodel(net, name):
    ckpname = ('./checkpoint/' + '_' + name + '.pth')
    checkpoint = torch.load(ckpname)
    net.load_state_dict(checkpoint['net'])


def loaddivmodel(net, name, filename):
    ckpname = ('./checkpoint/' + filename + '.pth')
    checkpoint = torch.load(ckpname)
    net.load_state_dict(checkpoint[name])