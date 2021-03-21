from typing import Callable, Type
from math import gcd

import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import mobilenet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3

from fcn import FCN, FCNHead


__all__ = ['fcn_mobilenetv2', 'fcn_mobilenetv3', 'deeplabv3_mobilenetv3', 'deeplabv3_mobilenetv2']


def _segm_mobilenet(name, backbone_name, num_classes, aux, pretrained_backbone):
    backbone = mobilenet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        ).features

    return_layers = {'18': 'out'}
    if aux:
        return_layers['14'] = 'aux'
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 160
        aux_classifier = FCNHead(inplanes, num_classes)

    model_map = {
        'deeplabv3': (DeepLabHead, DeepLabV3),
        'fcn': (FCNHead, FCN),
    }
    inplanes = 1280
    classifier = model_map[name][0](inplanes, num_classes)
    base_model = model_map[name][1]

    model = base_model(backbone, classifier, aux_classifier)
    return model


def fcn_mobilenetv2(pretrained_backbone=True, 
                 num_classes=81, aux_loss=True, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _segm_mobilenet('fcn', 'mobilenet_v2', num_classes, aux_loss, pretrained_backbone, **kwargs)


def fcn_mobilenetv3(large=False, pretrained_backbone=True,
                 num_classes=81, aux_loss=True, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = 'mobilenet_v3_large' if large else 'mobilenet_v3_small'
    return _segm_mobilenet('fcn', backbone, num_classes, aux_loss, pretrained_backbone, **kwargs)


def deeplabv3_mobilenetv2(pretrained_backbone=True, 
                       num_classes=81, aux_loss=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _segm_mobilenet('deeplabv3', 'mobilenet_v2', num_classes, aux_loss, pretrained_backbone, **kwargs)


def deeplabv3_mobilenetv3(large=False, pretrained_backbone=True, 
                       num_classes=81, aux_loss=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = 'mobilenet_v3_large' if large else 'mobilenet_v3_small'
    return _segm_mobilenet('deeplabv3', 'mobilenet_v3', num_classes, aux_loss, pretrained_backbone, **kwargs)


# below section is functions for differential privacy
def _replace_child(
    root: nn.Module, child_name: str, converter: Callable[[nn.Module], nn.Module]
) -> None:
    """
    Converts a sub-module to a new module given a helper
    function, the root module and a string representing
    the name of the submodule to be replaced.

    Args:
        root: Root module whose sub module must be replaced.
        child_name: Name of submodule that must be replaced.
        converter: Function or a lambda that takes a module
            (the submodule to be replaced) and returns its
            replacement.
    """
    # find the immediate parent
    parent = root
    nameList = child_name.split(".")
    for name in nameList[:-1]:
        parent = parent._modules[name]
    # set to identity
    parent._modules[nameList[-1]] = converter(parent._modules[nameList[-1]])


def replace_all_modules(
    root: nn.Module,
    target_class: Type[nn.Module],
    converter: Callable[[nn.Module], nn.Module],
) -> nn.Module:
    """
    Converts all the submodules (of root) that have the same
    type as target_class, given a converter, a module root,
    and a target class type.

    This method is useful for replacing modules that are not
    supported by the Privacy Engine.

    Args:
        root: Model instance, potentially with sub-modules
        target_class: Target class that needs to be replaced.
        converter: Function or a lambda that converts an instance
            of a given target_class to another nn.Module.

    Returns:
        Module with all the target_class types replaced using the
        converter. root is modified and is equal to the return value.

    Example:
        >>>  from torchvision.models import resnet18
        >>>  from torch import nn
        >>>  model = resnet18()
        >>>  print(model.layer1[0].bn1)
        BatchNorm2d(64, eps=1e-05, ...
        >>>  model = replace_all_modules(model, nn.BatchNorm2d, lambda _: nn.Identity())
        >>>  print(model.layer1[0].bn1)
        Identity()
    """
    # base case
    if isinstance(root, target_class):
        return converter(root)

    for name, obj in root.named_modules():
        if isinstance(obj, target_class):
            _replace_child(root, name, converter)
    return root


def _batchnorm_to_instancenorm(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm module to the corresponding InstanceNorm module

    Args:
        module: BatchNorm module to be replaced

    Returns:
        InstanceNorm module that can replace the BatchNorm module provided
    """    
    def matchDim():
        if isinstance(module, nn.BatchNorm1d):
            return nn.InstanceNorm1d
        elif isinstance(module, nn.BatchNorm2d):
            print(module.num_features)            
            return nn.InstanceNorm2d
        elif isinstance(module, nn.BatchNorm3d):
            return nn.InstanceNorm3d

    return matchDim()(module.num_features)


def _batchnorm_to_groupnorm(module: nn.modules.batchnorm._BatchNorm) -> nn.Module:
    """
    Converts a BatchNorm ``module`` to GroupNorm module.
    This is a helper function.

    Args:
        module: BatchNorm module to be replaced

    Returns:
        GroupNorm module that can replace the BatchNorm module provided

    Notes:
        A default value of 32 is chosen for the number of groups based on the
        paper *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour*
        https://arxiv.org/pdf/1706.02677.pdf
    """
    # use 8 instead of 32, as 8 is the maximum divisor of channel numbers
    return nn.GroupNorm(min(8, module.num_features), module.num_features, affine=True)



def nullify_batchnorm_modules(root: nn.Module) -> nn.Module:
    """
    Replaces all the BatchNorm submodules (e.g. :class:`torch.nn.BatchNorm1d`,
    :class:`torch.nn.BatchNorm2d` etc.) in ``root`` with :class:`torch.nn.Identity`.

    Args:
        root: Module for which to replace BatchNorm submodules.

    Returns:
        Module with all the BatchNorm sub modules replaced with
        Identity. ``root`` is modified and is equal to the return value.

    Notes:
        Most of the times replacing a BatchNorm module with Identity
        will heavily affect convergence of the model.
    """
    return replace_all_modules(
        root, nn.modules.batchnorm._BatchNorm, lambda _: nn.Identity()
    )


def convert_batchnorm_modules(
    model: nn.Module,
    converter: Callable[
        [nn.modules.batchnorm._BatchNorm], nn.Module
    ] = _batchnorm_to_groupnorm, #_batchnorm_to_instancenorm
) -> nn.Module:
    """
    Converts all BatchNorm modules to another module
    (defaults to GroupNorm) that is privacy compliant.

    Args:
        model: Module instance, potentially with sub-modules
        converter: Function or a lambda that converts an instance of a
            Batchnorm to another nn.Module.

    Returns:
        Model with all the BatchNorm types replaced by another operation
        by using the provided converter, defaulting to GroupNorm if one
        isn't provided.

    Example:
        >>>  from torchvision.models import resnet50
        >>>  from torch import nn
        >>>  model = resnet50()
        >>>  print(model.layer1[0].bn1)
        BatchNorm2d module details
        >>>  model = convert_batchnorm_modules(model)
        >>>  print(model.layer1[0].bn1)
        GroupNorm module details
    """
    return replace_all_modules(model, nn.modules.batchnorm._BatchNorm, converter)

def check_bn_num_features(
    model: nn.Module,
    converter: Callable[
        [nn.modules.batchnorm._BatchNorm], nn.Module
    ] = _batchnorm_to_instancenorm,
) -> nn.Module:
    
    return replace_all_modules(model, nn.modules.batchnorm._BatchNorm, converter)

def _relu_2_tanh(module: nn.modules.activation.ReLU) -> nn.Module:
    if isinstance(module, nn.modules.activation.ReLU):
        return nn.Tanh()        

def convert_relu_tanh(
    model: nn.Module,
    converter: Callable[
        [nn.modules.activation.ReLU], nn.Module
    ] = _relu_2_tanh, #_batchnorm_to_instancenorm
) -> nn.Module:
    """
    Converts all RELU modules to TANH for DP-SGD.
    """
    return replace_all_modules(model, nn.modules.activation.ReLU, converter)