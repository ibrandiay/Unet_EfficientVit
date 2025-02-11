import torch
import torch.nn as nn
from unet.unet3plus.init_weights import init_weights


class UnetConv2d(nn.Module):
	def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
		super(UnetConv2d, self).__init__()
		
		self.n = n
		self.ks = ks
		self.stride = stride
		self.padding = padding
		s = stride
		p = padding
		if is_batchnorm:
			for i in range(1, n + 1):
				conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
				                     nn.BatchNorm2d(out_size),
				                     nn.ReLU(inplace=True), )
				setattr(self, 'conv%d' % i, conv)
				in_size = out_size
		
		else:
			for i in range(1, n + 1):
				conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
				                     nn.ReLU(inplace=True), )
				setattr(self, 'conv%d' % i, conv)
				in_size = out_size
		
		# initialise the blocks
		for m in self.children():
			init_weights(m, init_type='kaiming')
	
	def forward(self, inputs):
		x = inputs
		for i in range(1, self.n + 1):
			conv = getattr(self, 'conv%d' % i)
			x = conv(x)
		
		return x


class UnetUp(nn.Module):
	def __init__(self, in_size, out_size, is_deconv, n_concat=2):
		super(UnetUp, self).__init__()
		# self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
		self.conv = UnetConv2d(out_size * 2, out_size, False)
		if is_deconv:
			self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
		else:
			self.up = nn.UpsamplingBilinear2d(scale_factor=2)
		
		# initialise the blocks
		for m in self.children():
			if m.__class__.__name__.find('unetConv2') != -1: continue
			init_weights(m, init_type='kaiming')
	
	def forward(self, inputs0, *input):
		# print(self.n_concat)
		# print(input)
		outputs0 = self.up(inputs0)
		for i in range(len(input)):
			outputs0 = torch.cat([outputs0, input[i]], 1)
		return self.conv(outputs0)


class UnetUpOrigin(nn.Module):
	def __init__(self, in_size, out_size, is_deconv, n_concat=2):
		super(UnetUpOrigin, self).__init__()
		# self.conv = unetConv2(out_size*2, out_size, False)
		if is_deconv:
			self.conv = UnetConv2d(in_size + (n_concat - 2) * out_size, out_size, False)
			self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
		else:
			self.conv = UnetConv2d(in_size + (n_concat - 2) * out_size, out_size, False)
			self.up = nn.UpsamplingBilinear2d(scale_factor=2)
		
		# initialise the blocks
		for m in self.children():
			if m.__class__.__name__.find('unetConv2') != -1: continue
			init_weights(m, init_type='kaiming')
	
	def forward(self, inputs0, *input):
		# print(self.n_concat)
		# print(input)
		outputs0 = self.up(inputs0)
		for i in range(len(input)):
			outputs0 = torch.cat([outputs0, input[i]], 1)
		return self.conv(outputs0)


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