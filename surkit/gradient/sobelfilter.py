#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple


class SobelFilter(object):
    """
    Get image gradient, only support pytorch now.
    """
    def __init__(self, imsize, correct=True):
        # conv2d is cross-correlation, need to transpose the kernel here
        self.HSOBEL_WEIGHTS_3x3 = torch.FloatTensor(
            np.array([[-1, -2, -1],
                      [0, 0, 0],
                      [1, 2, 1]]) / 8.0).unsqueeze(0).unsqueeze(0)

        self.VSOBEL_WEIGHTS_3x3 = self.HSOBEL_WEIGHTS_3x3.transpose(-1, -2)

        self.VSOBEL_WEIGHTS_5x5 = torch.FloatTensor(
            np.array([[-5, -4, 0, 4, 5],
                      [-8, -10, 0, 10, 8],
                      [-10, -20, 0, 20, 10],
                      [-8, -10, 0, 10, 8],
                      [-5, -4, 0, 4, 5]]) / 240.).unsqueeze(0).unsqueeze(0)
        self.HSOBEL_WEIGHTS_5x5 = self.VSOBEL_WEIGHTS_5x5.transpose(-1, -2)

        modifier = np.eye(imsize)
        modifier[0:2, 0] = np.array([4, -1])
        modifier[-2:, -1] = np.array([-1, 4])
        self.modifier = torch.FloatTensor(modifier)
        self.correct = correct

    def grad_h(self, image, filter_size=3):
        """
        Get image gradient along horizontal direction, or x axis.
        Option to do replicate padding for image before convolution. This is mainly
        for estimate the du/dy, enforcing Neumann boundary condition.

        :param image:           input image; tensor
        :param filter_size:     different SobelFilter kernel size
        :return:                horizontal gradient
        """

        image_width = image.shape[-1]

        if filter_size == 3:
            replicate_pad = 1
            kernel = self.VSOBEL_WEIGHTS_3x3
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.VSOBEL_WEIGHTS_5x5
        image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0, bias=None) * image_width
        # modify the boundary based on forward & backward finite difference (three points)
        # forward [-3, 4, -1], backward [3, -4, 1]
        if self.correct:
            return torch.matmul(grad, self.modifier)
        else:
            return grad

    def grad_v(self, image, filter_size=3):
        """
        Get image gradient along vertical direction, or y axis.
        Option to do replicate padding for image before convolution. This is mainly
        for estimate the du/dx, enforcing Neumann boundary condition.

        :param image:           input image; tensor
        :param filter_size:     different SobelFilter kernel size
        :return:                horizontal gradient
        """
        image_height = image.shape[-2]
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.HSOBEL_WEIGHTS_3x3
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.HSOBEL_WEIGHTS_5x5
        image = F.pad(image, _quadruple(replicate_pad), mode='replicate')
        grad = F.conv2d(image, kernel, stride=1, padding=0, bias=None) * image_height
        # modify the boundary based on forward & backward finite difference
        if self.correct:
            return torch.matmul(self.modifier.t(), grad)
        else:
            return grad
