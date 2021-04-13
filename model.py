import math
import torch
import argparse
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function
from op import FusedLeakyReLU, rasterize
from layers import *
''' SubModules '''
class StyledConv(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, style_dim, \
	upsample = False, blur_kernel = [1, 3, 3, 1], demodulate = True):
		super().__init__()
		self.conv = ModulatedConv2d(in_channel, \
				out_channel, \
				kernel_size, \
				style_dim, \
				upsample = upsample, \
				blur_kernel = blur_kernel, \
				demodulate = demodulate)
		self.noise = NoiseInjection()
		self.bias = None # nn.Parameter(torch.zeros(1, out_channel, 1, 1))
		# self.activate = ScaledLeakyReLU(0.2)
		self.activate = FusedLeakyReLU(out_channel)
	def forward(self, input, style, noise=None):
		out = self.conv(input, style)
		out = self.noise(out, noise=noise)
		if self.bias is not None:
			out = out + self.bias
		out = self.activate(out)
		return out
class StyledMapConv(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, style_dim, \
	upsample = False, blur_kernel = [1, 3, 3, 1], demodulate = True):
		super().__init__()
		self.conv = ModulatedConv2d(in_channel, \
				out_channel, \
				kernel_size, \
				style_dim, \
				upsample = upsample, \
				blur_kernel = blur_kernel, \
				demodulate = demodulate)
		self.noise = NoiseInjection()
		self.bias = None # nn.Parameter(torch.zeros(1, out_channel, 1, 1))
		# self.activate = ScaledLeakyReLU(0.2)
		self.activate = FusedLeakyReLU(out_channel)
	def forward(self, input, style, stylemap, noise=None):
		out = self.conv(input, style)
		out = out * stylemap[:,:1] + stylemap[:,1:2]
		out = self.noise(out, noise=noise)
		if self.bias is not None:
			out = out + self.bias
		out = self.activate(out)
		return out
class ToRGB(nn.Module):
	def __init__(self, in_channel, style_dim, upsample = True, blur_kernel = [1, 3, 3, 1]):
		super(ToRGB, self).__init__()
		if upsample:
			self.upsample = Upsample(blur_kernel)
		self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
		self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
	def forward(self, input, style, skip=None):
		out = self.conv(input, style)
		out = out + self.bias
		if skip is not None:
			skip = self.upsample(skip)
			out = out + skip
		return out
''' Network '''
class Generator(nn.Module):
	def __init__(self, size, style_dim, n_mlp, \
	channel_multiplier = 2, blur_kernel = [1, 3, 3, 1], lr_mlp = 0.01):
		super(Generator, self).__init__()
		self._initialize_styled(size,n_mlp,style_dim,channel_multiplier,lr_mlp)
		in_channel = self.channels[4]
		self.conv1 = StyledConv( \
			in_channel, in_channel, 3, style_dim, blur_kernel = blur_kernel)
		for i in range(3, self.log_size + 1):
			out_channel = self.channels[2 ** i]
			self.convs.append(StyledConv( \
				in_channel, out_channel, 3, style_dim, \
				upsample = True, blur_kernel = blur_kernel))
			self.convs.append(StyledConv( \
				out_channel, out_channel, 3, style_dim, blur_kernel = blur_kernel))
			self.to_rgbs.append(ToRGB(out_channel, style_dim))
			in_channel = out_channel
	def _initialize_styled(self, size, n_mlp, style_dim, channel_multiplier, lr_mlp):
		self.size = size
		self.style_dim = style_dim
		layers = [PixelNorm()]
		for i in range(n_mlp):
			layers.append(EqualLinear( \
				style_dim, style_dim, lr_mul = lr_mlp, activation = 'fused_lrelu'))
		self.style = nn.Sequential(*layers)
		self.channels = { \
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256 * channel_multiplier,
			128: 128 * channel_multiplier,
			256: 64 * channel_multiplier,
			512: 32 * channel_multiplier,
			1024: 16 * channel_multiplier}
		self.input = ConstantInput(self.channels[4])
		self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample = False)

		self.log_size = int(math.log(size, 2))
		self.num_layers = (self.log_size - 2) * 2 + 1
		self.convs = nn.ModuleList()
		self.upsamples = nn.ModuleList()
		self.to_rgbs = nn.ModuleList()
		self.noises = nn.Module()
		in_channel = self.channels[4]
		for layer_idx in range(self.num_layers):
			res = (layer_idx + 5) // 2
			shape = [1, 1, 2 ** res, 2 ** res]
			self.noises.register_buffer('noise_%d' % layer_idx, torch.randn(*shape))
		for i in range(3, self.log_size + 1):
			out_channel = self.channels[2 ** i]
			self.to_rgbs.append(ToRGB(out_channel, style_dim))
			in_channel = out_channel
		self.n_latent = self.log_size * 2 - 2
	def make_noise(self):
		device = self.input.input.device
		noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
		for i in range(3, self.log_size + 1):
			for _ in range(2):
				noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
		return noises
	def mean_latent(self, n_latent):
		latent_in = torch.randn(n_latent, self.style_dim, device = self.input.input.device)
		latent = self.style(latent_in).mean(0, keepdim = True)
		return latent
	def get_latent(self, input):
		return self.style(input)
	def forward(self, styles, return_latents = False, \
			inject_index = None, \
			truncation = 1, \
			truncation_latent = None, \
			input_is_latent = False, \
			noise = None, \
			randomize_noise=True):
		if not input_is_latent:
			styles = [self.style(s) for s in styles]
		if noise is None:
			if randomize_noise:
				noise = [None] * self.num_layers
			else:
				noise = [getattr(self.noises, 'noise_%d' % i) \
					if hasattr(self.noises, 'noise_%d' % i) else 0 \
					for i in range(self.num_layers)]
		if truncation < 1 and truncation_latent is not None:
			style_t = []
			for style in styles:
				style_t.append(
					truncation_latent + truncation * (style - truncation_latent))
			styles = style_t
		if len(styles) < 2:
			inject_index = self.n_latent
			if len(styles[0].shape) < 3:
				latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
			else:
				latent = styles[0]
		else:
			if inject_index is None:
				inject_index = np.random.choice(self.n_latent-2) + 1
			latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
			latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent-inject_index, 1)
			latent = torch.cat([latent, latent2], 1)
		out = self.input(latent)
		out = self.conv1(out, latent[:, 0], noise=noise[0])
		skip = self.to_rgb1(out, latent[:, 1])
		i = 1
		for conv1, conv2, noise1, noise2, to_rgb in zip( \
				self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], \
				self.to_rgbs):
			out = conv1(out, latent[:, i], noise = noise1)
			out = conv2(out, latent[:, i + 1], noise = noise2)
			skip = to_rgb(out, latent[:, i + 2], skip)
			i += 2
		image = skip
		if return_latents:
			return image, latent
		else:
			return image, None
class GeneratorWithMap(Generator):
	def __init__(self, size, style_dim, n_mlp, n_stylemap = 3, \
	channel_multiplier = 2, blur_kernel = [1, 3, 3, 1], lr_mlp = 0.01):
		super(Generator, self).__init__()
		super(GeneratorWithMap, self)._initialize_styled( \
			size, n_mlp, style_dim, channel_multiplier, lr_mlp)
		self.norm_to_style = nn.ModuleList()
		in_channel = self.channels[4]
		if n_stylemap != 3:
			self.norm1 = nn.Sequential( \
				ConvLayer(3, n_stylemap, 3), \
				ResBlock(n_stylemap, 2, downsample = False))
		else:
			self.norm1 = ResBlock(n_stylemap, 2, downsample = False)
		self.conv1 = StyledMapConv( \
			in_channel, in_channel, 3, style_dim, blur_kernel = blur_kernel)
		for i in range(3, self.log_size + 1):
			out_channel = self.channels[2 ** i]
			self.convs.append(StyledMapConv( \
				in_channel, out_channel, 3, style_dim, \
				upsample = True, blur_kernel = blur_kernel))
			self.convs.append(StyledMapConv( \
				out_channel, out_channel, 3, style_dim, blur_kernel = blur_kernel))
			if n_stylemap != 3:
				self.norm_to_style.append(ConvLayer(3, n_stylemap, 3))
				self.norm_to_style.append(ResBlock(n_stylemap, 4, downsample = False))
			else:
				self.norm_to_style.append(ResBlock(n_stylemap, 4, downsample = False))
			self.to_rgbs.append(ToRGB(out_channel, style_dim))
			in_channel = out_channel
	def make_noise(self):
		return super(GeneratorWithMap, self).make_noise()
	def mean_latent(self, n_latent):
		return super(GeneratorWithMap, self).mean_latent(latent)
	def get_latent(self, input):
		return self.style(input)
	def forward(self, styles, mesh, return_normals = False, \
			return_latents = False, \
			inject_index = None, \
			truncation = 1, \
			truncation_latent = None, \
			input_is_latent = False, \
			noise = None, \
			randomize_noise=True):
		if not input_is_latent:
			styles = [self.style(s) for s in styles]
		if noise is None:
			if randomize_noise:
				noise = [None] * self.num_layers
			else:
				noise = [getattr(self.noises, 'noise_%d' % i) \
					if hasattr(self.noises, 'noise_%d' % i) else 0 \
					for i in range(self.num_layers)]
		if truncation < 1 and truncation_latent is not None:
			style_t = []
			for style in styles:
				style_t.append(
					truncation_latent + truncation * (style - truncation_latent))
			styles = style_t
		if len(styles) < 2:
			inject_index = self.n_latent
			if len(styles[0].shape) < 3:
				latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
			else:
				latent = styles[0]
		else:
			if inject_index is None:
				inject_index = np.random.choice(self.n_latent-2) + 1
			latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
			latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent-inject_index, 1)
			latent = torch.cat([latent, latent2], 1)
		out = self.input(latent)
		norm_maps = [rasterize(mesh[0], mesh[1], mesh[2], \
			int(out.shape[2]), int(out.shape[3])).permute(0,3,1,2)]
		maps = self.norm1(norm_maps[-1])
		out = self.conv1(out, latent[:, 0], maps, noise=noise[0])
		skip = self.to_rgb1(out, latent[:, 1])
		i = 1
		for conv1, conv2, noise1, noise2, to_rgb in zip( \
				self.convs[::2], self.convs[1::2], \
				noise[1::2], noise[2::2], self.to_rgbs):
			norm_maps.append(rasterize(mesh[0], mesh[1], mesh[2], \
				2*int(out.shape[2]), 2*int(out.shape[3])).permute(0,3,1,2))
			if len(self.convs) == len(self.norm_to_style):
				maps = self.norm_to_style[i-1](norm_maps[-1])
				maps = self.norm_to_style[i](maps)
			else:
				maps = self.norm_to_style[i//2](norm_maps[-1])
			if isinstance(conv1, StyledMapConv):
				out = conv1(out, latent[:,i], maps[:,:2], noise = noise1)
			else:
				out = conv1(out, latent[:,i], noise = noise1)
			if isinstance(conv2, StyledMapConv):
				out = conv2(out, latent[:,i+1], maps[:,2:], noise = noise2)
			else:
				out = conv2(out, latent[:,i+1], noise = noise2)
			skip = to_rgb(out, latent[:,i+2], skip)
			i += 2
		image = skip
		if return_latents:
			if return_normals:
				return image, latent, norm_maps
			else:
				return image, latent, None
		elif return_normals:
			return image, None, norm_maps
		else:
			return image, None, None
class Discriminator(nn.Module):
	def __init__(self, size, channel_multiplier = 2, blur_kernel = [1, 3, 3, 1]):
		super(Discriminator, self).__init__()
		channels = { \
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256 * channel_multiplier,
			128: 128 * channel_multiplier,
			256: 64 * channel_multiplier,
			512: 32 * channel_multiplier,
			1024: 16 * channel_multiplier}
		convs = [ConvLayer(3, channels[size], 1)]
		log_size = int(math.log(size, 2))
		in_channel = channels[size]
		for i in range(log_size, 2, -1):
			out_channel = channels[2 ** (i - 1)]
			convs.append(ResBlock(in_channel, out_channel, blur_kernel))
			in_channel = out_channel
		self.convs = nn.Sequential(*convs)
		self.stddev_group = 4
		self.stddev_feat = 1
		self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
		self.final_linear = nn.Sequential( \
			EqualLinear(channels[4] * 4 * 4, channels[4], activation = 'fused_lrelu'), \
			EqualLinear(channels[4], 1))
	def forward(self, input):
		out = self.convs(input)
		batch, channel, height, width = out.shape
		group = min(batch, self.stddev_group)
		stddev = out.view(group, -1, self.stddev_feat, \
			channel//self.stddev_feat, height, width)
		stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
		stddev = stddev.mean([2, 3, 4], keepdim = True).squeeze(2)
		stddev = stddev.repeat(group, 1, height, width)
		out = torch.cat([out, stddev], 1)
		out = self.final_conv(out)
		out = out.view(batch, -1)
		out = self.final_linear(out)
		return out
class Regressor(nn.Module):
	def __init__(self, size, style_dim, n_mlp, \
	channel_multiplier = 2, blur_kernel = [1, 3, 3, 1], lr_mlp = 0.01):
		super(Regressor, self).__init__()
		self.size = size
		self.style_dim = style_dim
		self.channels = { \
			4: 512,
			8: 512,
			16: 512,
			32: 512,
			64: 256 * channel_multiplier,
			128: 128 * channel_multiplier,
			256: 64 * channel_multiplier,
			512: 32 * channel_multiplier,
			1024: 16 * channel_multiplier}
		self.convs = nn.ModuleList()
		self.log_size = int(math.log(size, 2))
		self.num_layers = (self.log_size - 2) * 2 + 1
		self.n_latent = self.log_size * 2 - 2
		self.downsamples = nn.ModuleList()
		self.from_rgbs = nn.ModuleList()
		self.convs = nn.ModuleList()
		in_channel = self.channels[size]
		channels = 2 * in_channel
		self.conv1 = ConvLayer(3, in_channel, 1)
		for i in range(self.log_size, 2, -1):
			out_channel = self.channels[2 ** i]
			self.convs.append(ConvLayer(in_channel, out_channel//2, 3))
			self.convs.append(ConvLayer(out_channel, out_channel, 3, downsample = True))
			self.from_rgbs.append(ConvLayer(3, out_channel//2))
			in_channel = out_channel
			channels += 2 * out_channel
		channels += 4 * 4 * out_channel
		layers = [EqualLinear(channels, style_dim, lr_mul = lr_mlp, activation = 'fused_lrelu')]
		for i in range(n_mlp-1):
			layers.append(EqualLinear( \
				style_dim, style_dim, lr_mul = lr_mlp, activation = 'fused_lrelu'))
		layers.append(PixelNorm())
		self.style = nn.Sequential(*layers)
	def forward(self, rgb):
		out = self.conv1(rgb)
		latents = torch.cat([out.mean([2,3]), out.var([2,3])], 1)
		for i in range(0,len(self.convs),2):
			out = self.convs[2*i](out)
			out = torch.cat([out, self.from_rgbs[i](rgb)], 1)
			out = self.convs[2*i+1](out)
			rgb = torch.nn.functional.interpolate(rgb, out.shape[2:4], mode = 'bilinear')
			latents = torch.cat([latents, out.mean([2,3]), out.var([2,3])], 1)
		latents = torch.cat([latents, out.view(int(out.shape[0]),-1)], 1)
		return self.style(latents)
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'StyleGAN2 trainer')
	parser.add_argument('--size', type = int, default = 1024, \
		help =	'image sizes of the model [%(default)d]')
	parser.add_argument('--latent', type = int, default = 512, \
		help =	'lantent dimension [%(default)d]')
	parser.add_argument('--n_mlp', type = int, default = 8, \
		help =	'latent converting network depth [%(default)d]')
	parser.add_argument('--channel_multiplier', type = int, default = 2, \
		help =	'channel multiplier factor for the model. config-f = 2, else = 1')
	args = parser.parse_args()
	G = Generator(args.size, args.latent, args.n_mlp, args.channel_multiplier)
	print('Generator:'); print(G)
	D = Discriminator(args.size, args.channel_multiplier)
	print('Discriminator:'); print(D)

