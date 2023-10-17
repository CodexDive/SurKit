#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import oneflow as flow
from oneflow import nn

from . import activations
from .bayes_layer import BayesConv2d


class _DenseLayer(nn.Sequential):
    """
    One bayes dense layer within dense block, with bottleneck design.
    """
    def __init__(self, in_features, growth_rate, drop_rate=0., bn_size=8,
                 bottleneck=False):
        """

        Args:
            in_features ():
            growth_rate ():
            drop_rate ():
            bn_size ():
            bottleneck ():
        """
        # """
        #
        #
        # :param in_features:
        # :param growth_rate:         out feature maps of every dense layer
        # :param drop_rate:
        # :param bn_size:             Specifies maximum number of features is `bn_size` * `growth_rate`
        # :param bottleneck:          If True, enable bottleneck design
        # """
        super(_DenseLayer, self).__init__()
        self.log_post = None
        self.log_prior = None
        if bottleneck and in_features > bn_size * growth_rate:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', BayesConv2d(in_features, bn_size *
                                                 growth_rate, kernel_size=1, stride=1))
            self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
            self.add_module('relu2', nn.ReLU(inplace=True))
            self.add_module('conv2', BayesConv2d(bn_size * growth_rate, growth_rate,
                                                 kernel_size=3, stride=1, padding=1))
        else:
            self.add_module('norm1', nn.BatchNorm2d(in_features))
            self.add_module('relu1', nn.ReLU(inplace=True))
            self.add_module('conv1', BayesConv2d(in_features, growth_rate,
                                                 kernel_size=3, stride=1, padding=1))
        if drop_rate > 0:
            self.add_module('dropout', nn.Dropout2d(p=drop_rate))

    def forward(self, x):
        y = super(_DenseLayer, self).forward(x)
        self.log_prior = sum([layer.log_prior if 'log_prior' in dir(layer) else 0 for layer in self])
        self.log_post = sum([layer.log_post if 'log_post' in dir(layer) else 0 for layer in self])
        return flow.cat([x, y], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_features, growth_rate, drop_rate,
                 bn_size=4, bottleneck=False):
        super(_DenseBlock, self).__init__()
        self.log_post = None
        self.log_prior = None
        for i in range(num_layers):
            layer = _DenseLayer(in_features + i * growth_rate, growth_rate,
                                drop_rate=drop_rate, bn_size=bn_size,
                                bottleneck=bottleneck)
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, x):
        y = super(_DenseBlock, self).forward(x)
        self.log_prior = sum([layer.log_prior if 'log_prior' in dir(layer) else 0 for layer in self])
        self.log_post = sum([layer.log_post if 'log_post' in dir(layer) else 0 for layer in self])
        return y


class _Transition(nn.Sequential):
    def __init__(self, in_features, out_features, down, bottleneck=True,
                 drop_rate=0, upsample='nearest'):
        """
        Bayes transition layer, either downsampling or upsampling, both reduce
        number of feature maps, i.e. `out_features` should be less than
        `in_features`.

        :param in_features:         number of input feature maps
        :param out_features:        number of output feature maps
        :param down:                If True, downsampling, else upsampling
        :param bottleneck:          If True, enable bottleneck design
        :param drop_rate:
        :param upsample:            mode of upsample
        """

        super(_Transition, self).__init__()
        self.log_post = None
        self.log_prior = None
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(inplace=True))
        if down:
            # half feature resolution, reduce # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', BayesConv2d(in_features, out_features,
                                                     kernel_size=1, stride=1, padding=0))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
                # not using pooling, fully convolutional...
                self.add_module('conv2', BayesConv2d(out_features, out_features,
                                                     kernel_size=3, stride=2, padding=1))
                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('conv1', BayesConv2d(in_features, out_features,
                                                     kernel_size=3, stride=2, padding=1))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        else:
            # transition up, increase feature resolution, half # feature maps
            if bottleneck:
                # bottleneck impl, save memory, add nonlinearity
                self.add_module('conv1', BayesConv2d(in_features, out_features,
                                                     kernel_size=1, stride=1, padding=0))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

                self.add_module('norm2', nn.BatchNorm2d(out_features))
                self.add_module('relu2', nn.ReLU(inplace=True))
                # output_padding=0, or 1 depends on the image size
                # if image size is of the power of 2, then 1 is good
                if upsample is None:
                    self.add_module('convT2', nn.ConvTranspose2d(
                        out_features, out_features, kernel_size=3, stride=2,
                        padding=1, output_padding=1))
                elif upsample == 'bilinear':
                    self.add_module('upsample', nn.UpsamplingBilinear2d(scale_factor=2))
                    self.add_module('conv2', BayesConv2d(out_features, out_features,
                                                         3, 1, 1))
                elif upsample == 'nearest':
                    self.add_module('upsample', nn.UpsamplingNearest2d(scale_factor=2))
                    self.add_module('conv2', BayesConv2d(out_features, out_features,
                                                         3, 1, 1))

                if drop_rate > 0:
                    self.add_module('dropout2', nn.Dropout2d(p=drop_rate))
            else:
                self.add_module('convT1', nn.ConvTranspose2d(
                    out_features, out_features, kernel_size=3, stride=2,
                    padding=1, output_padding=1))
                if drop_rate > 0:
                    self.add_module('dropout1', nn.Dropout2d(p=drop_rate))

    def forward(self, x):
        y = super(_Transition, self).forward(x)
        self.log_prior = sum([layer.log_prior if 'log_prior' in dir(layer) else 0 for layer in self])
        self.log_post = sum([layer.log_post if 'log_post' in dir(layer) else 0 for layer in self])
        return y


class _LastDecoding(nn.Sequential):
    def __init__(self, in_features, out_channels, drop_rate=0., upsample='nearest'):
        """
        Last bayes transition up layer, which outputs directly the predictions.
        """
        super(_LastDecoding, self).__init__()
        self.log_post = None
        self.log_prior = None
        self.add_module('norm1', nn.BatchNorm2d(in_features))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv1', BayesConv2d(in_features, in_features // 2,
                                             kernel_size=3, stride=1, padding=1))
        if drop_rate > 0.:
            self.add_module('dropout1', nn.Dropout2d(p=drop_rate))
        self.add_module('norm2', nn.BatchNorm2d(in_features // 2))
        self.add_module('relu2', nn.ReLU(True))
        if upsample == 'nearest':
            self.add_module('upsample', nn.UpsamplingNearest2d(scale_factor=2))
        elif upsample == 'bilinear':
            self.add_module('upsample', nn.UpsamplingBilinear2d(scale_factor=2))
        self.add_module('conv2', BayesConv2d(in_features // 2, in_features // 4,
                                             kernel_size=3, stride=1, padding=1))
        self.add_module('norm3', nn.BatchNorm2d(in_features // 4))
        self.add_module('relu3', nn.ReLU(True))
        self.add_module('conv3', BayesConv2d(in_features // 4, out_channels,
                                             kernel_size=5, stride=1, padding=2))

    def forward(self, x):
        y = super(_LastDecoding, self).forward(x)
        self.log_prior = sum([layer.log_prior if 'log_prior' in dir(layer) else 0 for layer in self])
        self.log_post = sum([layer.log_post if 'log_post' in dir(layer) else 0 for layer in self])
        return y


class DenseED(nn.Module):
    def __init__(self, in_channels, out_channels, imsize, blocks, growth_rate=16,
                 init_features=48, drop_rate=0, bn_size=8, bottleneck=False,
                 out_activation=None, upsample='nearest'):
        """
        Bayes Dense Convolutional Encoder-Decoder Networks.
        Decoder: Upsampling + Conv instead of TransposeConv

        :param in_channels:             number of input channels
        :param out_channels:            number of output channels
        :param imsize:                  image size, assume squared image
        :param blocks:                  A list (of odd size) of integers
        :param growth_rate:
        :param init_features:           number of feature maps after first conv layer
        :param drop_rate:               dropout rate
        :param bn_size:                 bottleneck size for number of feature maps (not used)
        :param bottleneck:              use bottleneck for dense block or not (False)
        :param out_activation:          Output activation function, choices=[None, 'tanh', 'sigmoid', 'softplus']
        :param upsample:                upsample mode
        """

        super(DenseED, self).__init__()
        if len(blocks) > 1 and len(blocks) % 2 == 0:
            raise ValueError('length of blocks must be an odd number, but got {}'
                             .format(len(blocks)))
        enc_block_layers = blocks[: len(blocks) // 2]
        dec_block_layers = blocks[len(blocks) // 2:]

        self.features = nn.Sequential()
        pad = 3 if imsize % 2 == 0 else 2
        # First convolution, half image size ================
        # For even image size: k7s2p3, k5s2p2
        # For odd image size (e.g. 65): k7s2p2, k5s2p1, k13s2p5, k11s2p4, k9s2p3
        self.features.add_module('In_conv', BayesConv2d(in_channels, init_features,
                                                        kernel_size=7, stride=2, padding=pad))
        # Encoding / transition down ================
        # dense block --> encoding --> dense block --> encoding
        num_features = init_features
        for i, num_layers in enumerate(enc_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                bottleneck=bottleneck)
            self.features.add_module('EncBlock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            trans_down = _Transition(in_features=num_features,
                                     out_features=num_features // 2,
                                     down=True,
                                     drop_rate=drop_rate)
            self.features.add_module('TransDown%d' % (i + 1), trans_down)
            num_features = num_features // 2
        # Decoding / transition up ==============
        # dense block --> decoding --> dense block --> decoding --> dense block
        for i, num_layers in enumerate(dec_block_layers):
            block = _DenseBlock(num_layers=num_layers,
                                in_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                bottleneck=bottleneck)
            self.features.add_module('DecBlock%d' % (i + 1), block)
            num_features += num_layers * growth_rate
            # the last decoding layer has different convT parameters
            if i < len(dec_block_layers) - 1:
                trans_up = _Transition(in_features=num_features,
                                       out_features=num_features // 2,
                                       down=False,
                                       drop_rate=drop_rate,
                                       upsample=upsample)
                self.features.add_module('TransUp%d' % (i + 1), trans_up)
                num_features = num_features // 2

        # The last decoding layer =======
        self.features.add_module('LastTransUp', _LastDecoding(num_features, out_channels,
                                                              drop_rate=drop_rate, upsample=upsample))

        if out_activation is not None:
            self.features.add_module(out_activation, activations.get(out_activation))

    def forward(self, x):
        return self.features(x)

    def log_prior(self):
        # calculate the log prior over all the layers
        return sum([layer.log_prior if 'log_prior' in dir(layer) else 0 for layer in self.features])

    def log_post(self):
        # calculate the log posterior over all the layers
        return sum([layer.log_post if 'log_post' in dir(layer) else 0 for layer in self.features])
