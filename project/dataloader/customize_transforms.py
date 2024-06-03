#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/project/dataloader/customize_transforms.py
Project: /workspace/code/project/dataloader
Created Date: Monday June 3rd 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday June 3rd 2024 1:34:06 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import torch 

class AddSaltAndPepperNoise(object):
    def __init__(self, prob=0.05):
        self.prob = prob

    def __call__(self, tensor):
        # 创建一个与图像大小相同的随机数掩码
        noise = torch.rand(tensor.size())
        
        # 创建一个新的张量副本，以便不修改原始图像张量
        noisy_tensor = tensor.clone()
        
        # 将噪声概率小于 prob/2 的像素设置为 0（椒噪声）
        noisy_tensor[noise < (self.prob / 2)] = 0
        
        # 将噪声概率大于 1 - prob/2 的像素设置为 1（盐噪声）
        noisy_tensor[noise > 1 - (self.prob / 2)] = 1
        
        return noisy_tensor

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class AddUniformNoise(object):
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def __call__(self, tensor):
        noise = torch.empty(tensor.size()).uniform_(self.low, self.high)
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0.0, 1.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(low={self.low}, high={self.high})"