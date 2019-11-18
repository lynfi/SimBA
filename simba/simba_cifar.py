'''SimBA-DCT lingfei-version.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
from matplotlib import pyplot as plt
import copy

from models import *
from utils import progress_bar
from normal import *
from readmodel import *
import torch_dct as dct
import simbaf

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--gpu',
                    default=None,
                    type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--idx', default=-1, type=int, help='model num')
parser.add_argument('--bs', default=50, type=int, help='batch size')
parser.add_argument('--name', default='-1', type=str, help='name')
parser.add_argument('--targeted', action='store_true', help='targeted attack')
parser.add_argument('--pixel', action='store_true', help='No Use DCT')

args = parser.parse_args()
if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

transform_test = transforms.Compose([transforms.ToTensor()])

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.bs,
                                         shuffle=False,
                                         num_workers=2,
                                         pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

# Model
print('==> Building model..')

loadname = 'zhu'
resnet18 = readmodel('resnet18', device)
loaddivmodel(resnet18, 'resnet18', loadname)
resnet18.eval()

vgg16 = readmodel('vgg16', device)
loaddivmodel(vgg16, 'vgg16', loadname)
vgg16.eval()

preactresnet18 = readmodel('preactresnet18', device)
loaddivmodel(preactresnet18, 'preactresnet18', loadname)
preactresnet18.eval()

densenet_cifar = readmodel('densenet_cifar', device)
loaddivmodel(densenet_cifar, 'densenet_cifar', loadname)
densenet_cifar.eval()

if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()


def net(inputs):
    idx = torch.randint(4, [1])
    if args.idx > -1:
        idx = torch.tensor(args.idx)
    if idx == 0:
        outputs = resnet18(inputs)
    elif idx == 1:
        outputs = vgg16(inputs)
    elif idx == 2:
        outputs = preactresnet18(inputs)
    elif idx == 3:
        outputs = densenet_cifar(inputs)
    else:
        outputs = (resnet18(inputs) + vgg16(inputs) + preactresnet18(inputs) +
                   densenet_cifar(inputs)) / 4
    return outputs, idx


def test(epoch):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx, len(testloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))


def CWloss(outputs, targets):
    mask = torch.zeros_like(outputs)
    mask[range(len(targets)), targets] = 1
    correct_logit = (mask * outputs).sum(1)
    wrong_logit, _ = torch.max((1 - mask) * outputs - 1e4 * mask, 1)
    loss = correct_logit - wrong_logit
    return loss


def SimBA(max_iters=3 * 32 * 32,
          freq_dims=4,
          stride=1,
          epsilon=0.2,
          targeted=False,
          pixel_attack=False,
          image_size=32):
    all_queries = torch.empty(0)
    all_l2 = torch.empty(0)
    all_idx = torch.empty(0)
    success = 0
    total = 1e-10
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if batch_idx % 10 != 0:
                continue
            total += len(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            images = inputs.clone()
            # target or untarget attack
            if targeted:
                labels = torch.randint_like(targets, 10)
                while labels.eq(targets).sum() > 0:
                    labels[labels.eq(targets)] = torch.randint_like(
                        targets[labels.eq(targets)], 10)
            else:
                labels = targets.clone()

            adv = inputs.clone()
            # the original points will be queried, so it is 1
            queries = torch.ones(len(targets))

            if pixel_attack:
                indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
            else:
                indices = simbaf.block_order(image_size,
                                             3,
                                             initial_size=freq_dims,
                                             stride=stride)[:max_iters]
            for k in range(max_iters):
                dim = indices[k]
                c, w, h = simbaf.location(
                    image_size, dim)  # return the location of the dim
                outputs, net_idx = net(adv)
                loss = CWloss(outputs, labels)
                outputs = F.softmax(outputs, dim=1)

                if targeted:
                    remaining = loss <= 0
                    PASS = loss > 0
                else:
                    remaining = loss > 0
                    PASS = loss <= 0

                if PASS.sum() > 0:
                    all_queries = torch.cat([all_queries, queries[PASS]])
                    l2 = (adv[PASS] - images[PASS]).view(PASS.sum(),
                                                         -1).norm(2, 1)
                    all_l2 = torch.cat([all_l2, l2.cpu()])
                    all_idx = torch.cat(
                        [all_idx, net_idx.repeat(PASS.sum()).float()])
                    success += float(PASS.sum())
                    adv = adv[remaining].clone()
                    images = images[remaining].clone()
                    labels = labels[remaining].clone()
                    outputs = outputs[remaining].clone()
                    queries = queries[remaining].clone()

                # check if all images are misclassified and stop early
                if remaining.sum() == 0:
                    break

                diff = torch.zeros_like(images)
                diff[:, c, w, h] = 1  # bs * c * w * h
                if not pixel_attack:
                    diff = dct.idct_2d(diff).clone()
                diff = diff / diff.view(diff.shape[0], -1).norm(2, 1).view(
                    -1, 1, 1, 1) * epsilon

                left_adv = (adv - diff).clamp(0, 1)
                left_outputs, _ = net(left_adv)
                left_outputs = F.softmax(left_outputs, dim=1)
                idx = left_outputs[range(len(labels)), labels] < outputs[range(
                    len(labels)), labels]
                if targeted:
                    idx = ~idx

                adv[idx] = left_adv[idx].clone()
                # only increase query count further by 1 for images
                # that did not improve in adversarial loss
                queries += 1
                queries[~idx] += 1

                right_adv = (adv + diff).clamp(0, 1)
                right_outputs, _ = net(right_adv)
                right_outputs = F.softmax(right_outputs, dim=1)
                idx2 = right_outputs[range(len(
                    labels)), labels] < outputs[range(len(labels)), labels]
                if targeted:
                    idx2 = ~idx2
                idx2[idx] = 0  # these points should not be queried or updated
                adv[idx2] = right_adv[idx2].clone()

            progress_bar(
                batch_idx / 10,
                len(testloader) / 10,
                'quirese: %.2f | l2: %.2f | success %.2f%%' %
                (all_queries.mean(), all_l2.mean(),
                 100. * success / float(total)))

    state = {
        'all_queries': all_queries,
        'all_l2': all_l2,
        'success': success,
        'total': total
    }
    if not os.path.isdir('./checkpoint'):
        os.mkdir('./checkpoint')
    ckpname = ('./checkpoint/' + args.name + '_simba.pth')
    torch.save(state, ckpname)


if args.pixel:
    freq = 32
else:
    freq = 4
print(args)
test(0)
SimBA(max_iters=3 * 32 * 32,
      freq_dims=freq,
      stride=1,
      epsilon=0.2,
      targeted=args.targeted,
      pixel_attack=args.pixel,
      image_size=32)
