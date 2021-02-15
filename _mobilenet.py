from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import mobilenet
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, DeepLabV3
from torchvision.models.segmentation.fcn import FCN, FCNHead


__all__ = ['fcn_mobilenetv2', 'fcn_mobilenetv3', 'deeplabv3_mobilenetv3', 'deeplabv3_mobilenetv2']


def _segm_mobilenet(name, backbone_name, num_classes, aux, pretrained_backbone):    
    if backbone_name.endswith('2'):
        backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained_backbone).features
    else: # ends with 3
        backbone = mobilenet.__dict__[backbone_name](pretrained=pretrained_backbone, reduced_tail=True).features

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
    return _segm_mobilenet('fcn', 'mobilenet_v2', num_classes, aux_loss, pretrained_backbone)

def fcn_mobilenetv3(large=False, pretrained_backbone=True,
                 num_classes=81, aux_loss=True, **kwargs):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = 'mobilenet_v3_large' if large else 'mobilenet_v3_small'
    return _segm_mobilenet('fcn', backbone, num_classes, aux_loss, pretrained_backbone)

def deeplabv3_mobilenetv2(pretrained_backbone=True, 
                       num_classes=81, aux_loss=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _segm_mobilenet('deeplabv3', 'mobilenet_v2', num_classes, aux_loss, pretrained_backbone)


def deeplabv3_mobilenetv3(large=False, pretrained_backbone=True, 
                       num_classes=81, aux_loss=True, **kwargs):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    backbone = 'mobilenet_v3_large' if large else 'mobilenet_v3_small'
    return _segm_mobilenet('deeplabv3', backbone, num_classes, aux_loss, pretrained_backbone)