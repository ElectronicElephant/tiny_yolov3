"""Darknet as YOLO backbone network."""
# pylint: disable=arguments-differ
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from collections import OrderedDict

__all__ = ['tiny_darknet', '_conv2d']


def _conv2d(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))

    return cell


class TinyYolov3(gluon.HybridBlock):
    """
    """
    def __init__(self, classes=1000, **kwargs):
        """ Network initialisation """
        super(TinyYolov3, self).__init__(**kwargs)

        # Network
        self.features = nn.HybridSequential()
        # backbone
        with self.features.name_scope():
            self.features.add(self.cbl(3, 16, 3, 1, 1))  # 416->416
            self.features.add(nn.MaxPool2D(2, 2))                # 416->208
            self.features.add(self.cbl(16, 32, 3, 1, 1))  # 208->208
            self.features.add(nn.MaxPool2D(2, 2))                # 208->104
            self.features.add(self.cbl(32, 64, 3, 1, 1))  # 104->104

            self.features.add(nn.MaxPool2D(2, 2))  # 104->52
            self.features.add(self.cbl(64, 128, 3, 1, 1))  # 52->52

            self.features.add(nn.MaxPool2D(2, 2))    # 52->26
            self.features.add(self.cbl(128, 256, 3, 1, 1))  # 26->26

            self.features.add(nn.MaxPool2D(2, 2))  # 26->13
            self.features.add(self.cbl(256, 512, 3, 1, 1))  # 13->13
            self.features.add(nn.MaxPool2D(3, 1, 1))  # 13->13
            self.features.add(self.cbl(512, 1024, 3, 1, 1))  # 13->13
            self.features.add(self.cbl(1024, 256, 1, 1, 0))  # 13->13

        # output
        self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        self.features(x)
        x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
        return self.output(x)

    def cbl(self, in_channels, out_channels, kernel, stride, padding, norm_layer=BatchNorm, norm_kwargs=None):
        """A common conv-bn-leakyrelu cell"""
        cell = nn.HybridSequential(prefix='')
        cell.add(nn.Conv2D(channels=out_channels, kernel_size=kernel,
                           strides=stride, padding=padding, in_channels=in_channels, use_bias=False))
        cell.add(norm_layer(axis=1, epsilon=1e-5, momentum=0.9, use_global_stats=False))
        cell.add(nn.LeakyReLU(0.1))

        return cell


def tiny_darknet():
    """Tiny Darknet v3 21 layer network.
    Reference: TencentYoutu <https://github.com/TencentYoutuResearch/ObjectDetection-OneStageDet>
    Parameters
    ----------
    Returns
    -------
    mxnet.gluon.HybridBlock
        Tiny Darknet network.

    """
    net = TinyYolov3()
    return net
