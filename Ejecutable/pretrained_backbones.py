"""
Pretrained Network Backbones for Single Colony Classification
==============================================================
Drop-in replacements for SingleColonyCNN using pretrained ImageNet weights.

Supported architectures:
    - ResNet-18, ResNet-34, ResNet-50 (and a slim ResNet-15-like variant)
    - VGG-11, VGG-16 (with batch norm)

All models accept [B, 3, 128, 128] input and output [B, n_classes] logits,
matching the SingleColonyCNN interface (including .get_pIDv()).

Usage:
    from pretrained_backbones import build_pretrained_model

    model = build_pretrained_model('resnet18', n_classes=32, pretrained=True)
    model = build_pretrained_model('resnet50', n_classes=32, pretrained=True)
    model = build_pretrained_model('vgg11_bn', n_classes=32, pretrained=True)
"""

import torch
import torch.nn as nn
from torchvision import models


# Registry of supported architectures
SUPPORTED_BACKBONES = [
    'resnet18', 'resnet34', 'resnet50',
    'vgg11_bn', 'vgg16_bn',
]


class PretrainedColonyCNN(nn.Module):
    """
    Wrapper that adapts any torchvision backbone for colony classification.

    The final classification layer is replaced to output `n_classes` logits.
    All other layers can be frozen or fine-tuned.

    Parameters
    ----------
    backbone_name : str
        One of SUPPORTED_BACKBONES.
    n_classes : int
        Number of output classes (default 32 for DeepColony).
    pretrained : bool
        Whether to load ImageNet-pretrained weights.
    freeze_backbone : bool
        If True, freeze all layers except the final classifier.
        Useful for fast fine-tuning with limited data.
    dropout : float
        Dropout probability before the final FC layer.
    """

    def __init__(self, backbone_name='resnet18', n_classes=32,
                 pretrained=True, freeze_backbone=False, dropout=0.5):
        super().__init__()
        self.backbone_name = backbone_name
        self.n_classes = n_classes

        weights = 'IMAGENET1K_V1' if pretrained else None

        if backbone_name.startswith('resnet'):
            self.backbone, fc_in = self._build_resnet(backbone_name, weights)
        elif backbone_name.startswith('vgg'):
            self.backbone, fc_in = self._build_vgg(backbone_name, weights)
        else:
            raise ValueError(
                f'Unknown backbone: {backbone_name}. '
                f'Supported: {SUPPORTED_BACKBONES}')

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fc_in, n_classes),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _build_resnet(self, name, weights):
        """Build a ResNet backbone, removing its FC layer."""
        builder = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
        }[name]
        base = builder(weights=weights)
        fc_in = base.fc.in_features
        # Remove original FC — use everything up to avgpool
        base.fc = nn.Identity()
        return base, fc_in

    def _build_vgg(self, name, weights):
        """Build a VGG backbone, removing its classifier."""
        builder = {
            'vgg11_bn': models.vgg11_bn,
            'vgg16_bn': models.vgg16_bn,
        }[name]
        base = builder(weights=weights)
        # VGG features → adaptive pool → flatten
        features = base.features
        pool = nn.AdaptiveAvgPool2d((4, 4))

        # Calculate flattened size: last conv channels * 4 * 4
        # VGG11: 512 * 4 * 4 = 8192, VGG16: same
        fc_in = 512 * 4 * 4

        backbone = nn.Sequential(features, pool, nn.Flatten())
        return backbone, fc_in

    def forward(self, x):
        features = self.backbone(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        return self.classifier(features)

    def get_pIDv(self, x):
        """Returns softmax probability vector — matches SingleColonyCNN API."""
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)


class SlimResNet(nn.Module):
    """
    A slim ~15-layer ResNet variant inspired by the architecture that
    previously worked in another student's thesis (ResNet-15-like).

    Uses fewer residual blocks than ResNet-18 for faster training on
    small colony images.

    Architecture:
        Conv7x7 → BN → ReLU → MaxPool
        → 2 BasicBlocks (64 filters)
        → 2 BasicBlocks (128 filters, stride=2)
        → 1 BasicBlock  (256 filters, stride=2)
        → AdaptiveAvgPool → FC(n_classes)

    ~15 weighted layers total (conv + FC).
    """

    def __init__(self, n_classes=32, dropout=0.5):
        super().__init__()

        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages (fewer blocks than ResNet-18)
        self.layer1 = self._make_layer(64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(256, n_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                         nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        # First block may downsample
        layers.append(_BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(_BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

    def get_pIDv(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)


class _BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (2 conv layers + skip connection)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                                stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                                stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                           stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


def build_pretrained_model(backbone_name, n_classes=32, pretrained=True,
                            freeze_backbone=False, dropout=0.5):
    """
    Factory function to build a pretrained colony classifier.

    Parameters
    ----------
    backbone_name : str
        'resnet18', 'resnet34', 'resnet50', 'vgg11_bn', 'vgg16_bn',
        or 'slim_resnet' for the ~15-layer variant.
    n_classes : int
        Number of species classes.
    pretrained : bool
        Use ImageNet weights (ignored for slim_resnet).
    freeze_backbone : bool
        Freeze pretrained layers, only train classifier.
    dropout : float
        Dropout rate before final FC.

    Returns
    -------
    nn.Module with .forward() and .get_pIDv() methods.
    """
    if backbone_name == 'slim_resnet':
        return SlimResNet(n_classes=n_classes, dropout=dropout)

    return PretrainedColonyCNN(
        backbone_name=backbone_name,
        n_classes=n_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )


def get_pretrained_transform(backbone_name, is_train=True):
    """
    Returns the appropriate transform for pretrained models.

    ImageNet-pretrained models expect normalized inputs, while the original
    SingleColonyCNN does not. This function returns the correct transforms
    so experiments can be run with the right preprocessing.
    """
    from torchvision import transforms

    # All pretrained models expect ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Import AdaptivePadding — assumed to be in the notebook's namespace
    # or passed externally. We define a standalone version here.
    from torchvision.transforms.functional import pad as F_pad

    class AdaptivePadding:
        def __init__(self, margin=20, fill=0):
            self.margin = margin
            self.fill   = fill

        def __call__(self, img):
            w, h     = img.size
            max_side = max(w, h)
            pad_w    = max_side - w + self.margin
            pad_h    = max_side - h + self.margin
            padding  = (pad_w // 2, pad_h // 2,
                        pad_w - pad_w // 2, pad_h - pad_h // 2)
            return transforms.functional.pad(img, padding, fill=self.fill)

    if is_train:
        return transforms.Compose([
            AdaptivePadding(20),
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            AdaptivePadding(20),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            normalize,
        ])
