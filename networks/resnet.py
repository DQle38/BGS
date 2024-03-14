import torch
import torch.nn as nn
import torchvision
from torch.nn import Sequential
from torch import Tensor

__all__ = ["ResNet", "resnet18", "resnet50",]

model_urls = {
    "resnet18": 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample= None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer= None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, taskcla=None, continual=None, for_icarl=False):
        super(ResNet, self).__init__()

        self.taskcla = taskcla

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num_blocks = layers
        self.inplanes = 64
        self.dilation = 1
        self.for_icarl = for_icarl
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        #         self.layer0 = self._make_layer_before_block(self.inplanes, self._norm_layer)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        assert taskcla is not None
        if 'task' in continual or 'class' in continual:
            self.fc = torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.fc.append(torch.nn.Linear(512 * block.expansion, n))
        else:
            self.fc = torch.nn.Linear(512 * block.expansion, self.taskcla[0][1])

        for m in self.modules():
            if 'Conv2d' in type(m).__name__:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, inplanes=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        inplanes = self.inplanes if inplanes is None else inplanes
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample_modules = []
            conv = nn.Conv2d(inplanes, planes * block.expansion, stride=stride, kernel_size=1, bias=False)
            downsample_modules.append(conv)
            downsample_modules.append(norm_layer(planes * block.expansion))
            downsample = Sequential(*downsample_modules)

        layers = []

        kwargs = {'inplanes': inplanes,
                  'planes': planes,
                  'stride': stride,
                  'downsample': downsample,
                  # 'groups':self.groups,
                  # 'base_width':self.base_width,
                  # 'dialation':previous_dilation,
                  'norm_layer': norm_layer,
                  }

        # if hasattr(self, 'num_groups'):
        #     kwargs['num_groups'] = self.num_groups

        layers.append(block(**kwargs))

        self.inplanes = planes * block.expansion
        kwargs['inplanes'] = self.inplanes
        kwargs.pop('downsample')
        kwargs.pop('stride')
        for _ in range(1, blocks):
            layers.append(block(**kwargs))

        return Sequential(*layers)

    def _make_layer_before_block(self, inplanes, norm_layer):
        layers = []
        conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        layers.append(conv1)
        layers.append(norm_layer(self.inplanes))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        return Sequential(*layers)

    def _forward_impl(self, x, get_inter=False, task_id=None):
        # See note [TorchScript super()]
        #         h = self.layer0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        h = self.avgpool(b4)
        h = torch.flatten(h, 1)
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
            return (b1, b2, b3, b4), h, y
        else:
            return y

    def forward(self, x, get_inter=False, task_id=None):
        return self._forward_impl(x, get_inter, task_id=task_id)
    
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

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
        
    if pretrained:
        if arch == 'resnet18':
            dummy_model = torchvision.models.resnet18(pretrained=True)
        else:
            dummy_model = torchvision.models.resnet50(pretrained=True)

        dummy_keys = [i for i in model.state_dict().keys()]
        for key1, key2 in zip(dummy_keys, dummy_model.state_dict().keys()):
            if model.state_dict()[key1].shape == torch.tensor(1).shape:
                model.state_dict()[key1] = dummy_model.state_dict()[key2]
            else:
                if key1.startswith('fc'):
                    continue
                model.state_dict()[key1][:] = dummy_model.state_dict()[key2][:]
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)

    return model


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)
