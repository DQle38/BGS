import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


__all__ = ['ResNet', 'resnet56']


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, taskcla=None, continual=None, num_classes=10, for_icarl=False):
        super(ResNet, self).__init__()
        self.taskcla = taskcla
        self.for_icarl = for_icarl
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        assert taskcla is not None
        if continual == 'task' or continual == 'class':
            self.fc = torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.fc.append(torch.nn.Linear(64, n))
        else:
            num_classes = self.taskcla[0][1]
            self.fc = torch.nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, task_id=None, get_inter=False):
        act1 = F.relu(self.bn1(self.conv1(x)))
        act2 = self.layer1(act1)
        act3 = self.layer2(act2)
        act4 = self.layer3(act3)
        h = F.avg_pool2d(act4, act4.size()[3])
        h = h.view(h.size(0), -1)
        normalized_feature = h / torch.norm(h, 2, 1).unsqueeze(1)
        h = normalized_feature if self.for_icarl else h

        if isinstance(self.fc, nn.ModuleList):
            if task_id is None:
                y = []
                for t, i in self.taskcla:
                    y.append(self.fc[t](h))
            else:
                tmp = 0
                for t, i in self.taskcla:
                    tmp += self.fc[t](h) * (task_id == t)[:, None]
                y = tmp
        else:
            y = self.fc(h)

        if get_inter:
            return act4, h, y
        else:
            return y
    
    def forward_head(self, h, task_id=None):
        if isinstance(self.fc, nn.ModuleList):
            if task_id is None:
                y = []
                for t, i in self.taskcla:
                    y.append(self.fc[t](h))
            else:
                tmp = 0
                for t, i in self.taskcla:
                    tmp += self.fc[t](h) * (task_id == t)[:, None]
                y = tmp
        else:
            y = self.fc(h)
        return y   
   
    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ResNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()


def resnet56(taskcla=None, continual=None, **kwargs):
    return ResNet(BasicBlock, [9, 9, 9], taskcla, continual, **kwargs)

