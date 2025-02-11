# -*- coding: utf-8 -*-
"""

Add Attention module in unet3 archi

paper:  https://www.nature.com/articles/s41598-024-70019-z#Fig1

@author: ibra Ndiaye
"""
import torch
import torch.nn as nn


class DaulAttentionModule(nn.Module):
	"""
	Attn module class
	"""
	def __init__(self, in_dim):
		super(DaulAttentionModule, self).__init__()
		self.chanel_in = in_dim
		self.conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.softmax = nn.Softmax(dim=1)
		self.avg_pol_c = nn.AvgPool2d(kernel_size=1)
		self.fc = nn.Sequential(
			nn.Linear(in_features=in_dim, out_features=in_dim // 16, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(in_features=in_dim // 16, out_features=in_dim, bias=False),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		avg_h = nn.AvgPool2d(kernel_size=(x.size(2), 1))(x)
		avg_w = nn.AvgPool2d(kernel_size=(1, x.size(3)))(x)
		pam = self.softmax(torch.matmul(avg_w, avg_h))
		avg_c = nn.AvgPool2d(kernel_size=x.size(2))(x).view(x.size(0), x.size(1))
		cam = self.fc(avg_c)
		cam = cam.view(x.size(0), x.size(1), 1, 1)
		out = torch.add(torch.mul(pam, x), torch.mul(cam.expand_as(x), x))
		return out


class Bottleneck(nn.Module):
	"""
	Bottleneck block .
	"""
	
	def __init__(self, in_channels, out_channels):
		super(Bottleneck, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.inners = out_channels // 4
		
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)
		
		self.conv1 = nn.Sequential(
			nn.Conv2d(out_channels, self.inners, kernel_size=1, bias=False),
			nn.BatchNorm2d(self.inners),
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(self.inners, self.inners, kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(self.inners),
			nn.ReLU(inplace=True),
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(self.inners, out_channels, kernel_size=1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True),
		)
	
	def forward(self, x):
		x = self.conv(x)
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		return out
