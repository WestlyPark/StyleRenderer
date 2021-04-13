import torch
import numpy as np
from torch import nn
from torch.autograd import Function
from torch.nn import functional as F
from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
def make_kernel(k):
	k = torch.tensor(k, dtype = torch.float32)
	if len(k.shape) == 1:
		k = k[None, :] * k[:, None]
	k /= k.sum()
	return k
class Normalize(torch.autograd.Function):
	@staticmethod
	def forward(ctx, vec, axis = -1, _type = 'L2', eps = 1e-8):
		ctx._type= _type
		ctx.axis = axis
		ctx.eps  = eps = abs(eps)
		if 'L2' in _type.upper():
			norm = torch.sqrt(torch.sum(vec * vec, axis, keepdim = True))
			norm = torch.clamp(norm, min = eps)
			vec = vec / norm
			ctx.save_for_backward(vec, norm)
		elif 'L1' in _type.upper():
			norm = torch.sum(vec, axis, keepdim = True)
			norm = torch.clamp(norm, min = eps)
			vec = vec / norm
			ctx.save_for_backward(vec, norm)
		elif 'LINF' in  _type.upper():
			norm, ind = torch.max(torch.abs(vec),axis,keepdim=True)
			norm = torch.clamp(norm, min = eps)
			vec = vec / norm
			ctx.save_for_backward(vec, norm, ind)
		return vec
	@staticmethod
	def backward(ctx, grad_v):
		if ctx.needs_input_grad[0]:
			if 'L2' in ctx._type.upper():
				vec, norm = ctx.saved_tensors
				grad_v = (grad_v - vec*torch.sum(vec*grad_v,ctx.axis,keepdim=True))/norm
			elif 'L1' in ctx._type.upper():
				vec, norm = ctx.saved_tensors
				grad_v = (grad_v - torch.sum(vec*grad_v,ctx.axis,keepdim=True))/norm
			elif 'LINF' in ctx._type.upper():
				vec, norm, ind = ctx.saved_tensors
				grad_v = grad_v / norm
				res =	torch.sum(vec*grad_v, ctx.axis, keepdim = True)
				res =	torch.gather(grad_v, ctx.axis, ind) + torch.where( \
					torch.gather(vec, ctx.axis, ind) < 0, res, -res)
				grad_v.scatter_(ctx.axis, ind, res)
		else:
			grad_v = None
		return tuple([grad_v]+[None]*(len(ctx.needs_input_grad)-1))
class BatchEigenMax(torch.autograd.Function):
	@staticmethod
	def forward(ctx, A):
		b = int(A.shape[0])
		n = int(A.shape[1])
		u = []; s = []
		for _ in A.view(b,n,-1).detach():
			u_, s_, v_ = torch.svd(_)
			u.append(u_.unsqueeze(0))
			s.append(s_.unsqueeze(0))
		u = torch.cat(u, 0)
		s = torch.cat(s, 0)
		i = torch.argmax(s, 1)
		s = torch.gather(s, 1, i.view(-1,1)).squeeze(-1)
		u = torch.gather(u, 2, \
			i.view(-1,1,1).expand(-1,n,1))
		u = torch.where(u[:,-1:,:] \
			.expand(-1,n,-1) < 0, -u, u) \
			.squeeze(-1)
		ctx.save_for_backward(u, s, A)
		return u, s
	@staticmethod
	def backward(ctx, du, ds):
		if ctx.needs_input_grad[0]:
			u, s, A = ctx.saved_tensors
			b = int(A.shape[0])
			n = int(A.shape[1])
			s = s.view(-1,1,1)
			su=-2 * s.view(-1,1) * u
			s2=(s * s).view(-1,1,1)
			K = torch.cat((torch.cat(( \
				torch.matmul(A, A.permute(0,2,1)),\
				su.unsqueeze(-1)),2), torch.cat(( \
				su.unsqueeze(1), \
				s2), 2)), 1)
			Kinv = torch.inverse(s2 * \
				torch.eye(n+1, dtype = K.dtype, \
				device = K.device).unsqueeze(0) - K)
			df = torch.matmul(torch.cat(( \
				du.view(-1,1,n), \
				ds.view(-1,1,1)), -1), Kinv)[:,0,:-1]
			d0 = torch.matmul(df.view(-1,1,n), A)
			d1 = torch.matmul(du.view(-1,1,n), A)
			return d0 * u.view(-1,n,1) + d1 * df.view(-1,n,1)
		else:
			return None
class PixelNorm(nn.Module):
	def __init__(self, eps = 1e-8):
		super(PixelNorm, self).__init__()
		self.eps = abs(eps)
	def forward(self, input):
		return input * torch.rsqrt(torch.mean(input*input,-1,keepdim = True) + self.eps)
class SpectralNorm(torch.nn.Module):
	def __init__(self, module, name = 'weight', power_iterations = 1, random_init = True):
		super(SpectralNorm, self).__init__()
		self.module = module
		self.name = name
		self.power_iterations = int(power_iterations)
		if not self._made_params():
			self._make_params(random_init)
	def _update_u_v(self):
		w = getattr(self.module, self.name + '_bar')
		height = int(w.shape[0])
		if self.power_iterations > 0:
			u = getattr(self.module, self.name + '_u')
			v = getattr(self.module, self.name + '_v')
			for _ in range(self.power_iterations):
				v.data = Normalize.apply( \
					torch.mv(torch.t(w.view(height,-1).data), u.data))
				u.data = Normalize.apply( \
					torch.mv(w.view(height,-1).data, v.data))
			# sigma = torch.dot(u.data,torch.mv(w.view(height,-1).data,v.data))
			sigma = u.dot(w.view(height, -1).mv(v))
		else:
			w_ = w.view(height, -1)
			width = int(w_.shape[1])
			if height > width:
				w_ = w_.permute(1,0).unsqueeze(0)
			else:
				w_ = w_.unsqueeze(0)
			_, sigma = BatchEigenMax.apply(w_)
		setattr(self.module, self.name, w / sigma.expand_as(w))
	def _made_params(self):
		try:
			u = getattr(self.module, self.name + '_u')
			v = getattr(self.module, self.name + '_v')
			w = getattr(self.module, self.name + '_bar')
			return True
		except AttributeError:
			return False
	def _make_params(self, random_init = True):
		w = getattr(self.module, self.name)

		height = w.data.shape[0]
		width = w.view(height, -1).data.shape[1]
		if self.power_iterations > 0:
			if random_init:
				u = w.data.new(height).normal_(0, 1)
				v = w.data.new(width).normal_(0, 1)
				u = Normalize.apply(u)
				v = Normalize.apply(v)
			else:
				u, _, v = torch.svd(w.data.view(height,-1))
				i = torch.argmax(_, 0)
				u = u[:, i]
				v = v[:, i]
			u = torch.nn.Parameter(u, requires_grad = False)
			v = torch.nn.Parameter(v, requires_grad = False)
			self.module.register_parameter(self.name + '_u', u)
			self.module.register_parameter(self.name + '_v', v)
		w_bar = torch.nn.Parameter(w.data)
		del self.module._parameters[self.name]
		self.module.register_parameter(self.name + '_bar', w_bar)
	def forward(self, *args):
		self._update_u_v()
		return self.module.forward(*args)
class Upsample(nn.Module):
	def __init__(self, kernel, factor=2):
		super(Upsample, self).__init__()
		self.factor = factor
		kernel = make_kernel(kernel) * (factor**2)
		self.register_buffer('kernel', kernel)
		p = kernel.shape[0] - factor
		pad0 = (p + 1) // 2 + factor - 1
		pad1 = p // 2
		self.pad = (pad0, pad1)
	def forward(self, input):
 		return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)
class Downsample(nn.Module):
	def __init__(self, kernel, factor = 2):
		super(Downsample, self).__init__()
		self.factor = factor
		kernel = make_kernel(kernel)
		self.register_buffer('kernel', kernel)
		p = kernel.shape[0] - factor
		pad0 = (p + 1) // 2
		pad1 = p // 2
		self.pad = (pad0, pad1)
	def forward(self, input):
		return upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)
class Blur(nn.Module):
	def __init__(self, kernel, pad, upsample_factor = 1):
		super(Blur, self).__init__()
		kernel = make_kernel(kernel)
		if upsample_factor > 1:
			kernel = kernel * (upsample_factor ** 2)
		self.register_buffer('kernel', kernel)
		self.pad = pad
	def forward(self, input):
		return upfirdn2d(input, self.kernel, pad = self.pad)
class EqualConv2d(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, \
	stride = 1, padding = 0, bias = True):
		super(EqualConv2d, self).__init__()
		self.weight = nn.Parameter(
			torch.randn(out_channel, in_channel, kernel_size, kernel_size))
		self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
		self.stride = stride
		self.padding = padding
		self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None
	def forward(self, input):
		out = F.conv2d(input, self.weight*self.scale, \
			bias = self.bias, stride = self.stride, padding = self.padding)
		return out
	def __repr__(self):
		return	'%s(%d, %d, %d, stride=%d, padding=%d)' % ( \
			self.__class__.__name__, self.weight.shape[1], self.weight.shape[0], \
			self.weight.shape[2], self.stride, self.padding)
class EqualLinear(nn.Module):
	def __init__(self, in_dim, out_dim, bias = True, \
	bias_init = 0, lr_mul = 1, activation = None):
		super(EqualLinear, self).__init__()
		self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
		if bias:
			self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
		else:
			self.bias = None
		self.activation = activation
		self.scale = (1 / math.sqrt(in_dim)) * lr_mul
		self.lr_mul = lr_mul
	def forward(self, input):
		if self.activation == 'fused_lrelu':
			out = F.linear(input, self.weight * self.scale)
			out = fused_leaky_relu(out, self.bias * self.lr_mul)
		else:
			out = F.linear(input, self.weight*self.scale, bias = self.bias*self.lr_mul)
			if self.activation == 'relu':
				out = F.relu(out)
			elif self.activation == 'lrelu':
				out = F.leaky_relu(out, negative_slope = 0.2)
			elif self.activation == 'selu':
				out = F.selu(out)
			elif self.activation == 'tanh':
				out = F.tanh(out)
		return out
	def __repr__(self):
		return '%s(%d, %d)' % ( \
			self.__class__.__name__, self.weight.shape[1], self.weight.shape[0])
class ScaledLeakyReLU(nn.Module):
	def __init__(self, negative_slope = 0.2):
		super(ScaledLeakyReLU, self).__init__()
		self.negative_slope = negative_slope
	def forward(self, input):
		out = F.leaky_relu(input, negative_slope = self.negative_slope)
		return out * math.sqrt(2)
class ModulatedConv2d(nn.Module):
	def __init__(self, in_channel, out_channel, kernel_size, style_dim, \
	demodulate = True, upsample = False, downsample = False, \
	blur_kernel=[1, 3, 3, 1]):
		super(ModulatedConv2d, self).__init__()
		self.eps = 1e-8
		self.kernel_size = kernel_size
		self.in_channel = in_channel
		self.out_channel = out_channel
		self.upsample = upsample
		self.downsample = downsample
		if upsample:
			factor = 2
			p = (len(blur_kernel) - factor) - (kernel_size - 1)
			pad0 = (p + 1) // 2 + factor - 1
			pad1 = p // 2 + 1
			self.blur = Blur(blur_kernel, pad = (pad0, pad1), upsample_factor = factor)
		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2
			self.blur = Blur(blur_kernel, pad = (pad0, pad1))
		fan_in = in_channel * kernel_size ** 2
		self.scale = 1 / math.sqrt(fan_in)
		self.padding = kernel_size // 2
		self.weight = nn.Parameter(torch.randn(1, \
			out_channel, in_channel, kernel_size, kernel_size))
		self.modulation = EqualLinear(style_dim, in_channel, bias_init = 1)
		self.demodulate = demodulate
	def __repr__(self):
		return '%s(%d, %d, %d, upsample=%s, downsample=%s)' % (
			self.__class__.__name__, self.in_channel, self.out_channel, self.kernel_size,\
			self.upsample, self.downsample)
	def forward(self, input, style):
		batch, in_channel, height, width = input.shape[:4]
		style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
		weight = self.scale * self.weight * style
		if self.demodulate:
			demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps)
			weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)
		weight = weight.view(-1, in_channel, self.kernel_size, self.kernel_size)
		if self.upsample:
			input = input.view(1, -1, height, width)
			weight = weight.view(batch, \
				self.out_channel, in_channel, self.kernel_size, self.kernel_size)
			weight = weight.transpose(1, 2).reshape(-1, \
				self.out_channel, self.kernel_size, self.kernel_size)
			out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)
			out = self.blur(out)
		elif self.downsample:
			input = self.blur(input)
			_, _, height, width = input.shape
			input = input.view(1, batch * in_channel, height, width)
			out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)
		else:
			input = input.view(1, batch * in_channel, height, width)
			out = F.conv2d(input, weight, padding=self.padding, groups=batch)
			_, _, height, width = out.shape
			out = out.view(batch, self.out_channel, height, width)
		return out
class NoiseInjection(nn.Module):
	def __init__(self):
		super(NoiseInjection, self).__init__()
		self.weight = nn.Parameter(torch.zeros(1))
	def forward(self, image, noise = None):
		if noise is None:
			batch, _, height, width = image.shape
			noise = image.new_empty(batch, 1, height, width).normal_()
		return image + self.weight * noise
class ConstantInput(nn.Module):
	def __init__(self, channel, size = 4):
		super(ConstantInput, self).__init__()
		self.input = nn.Parameter(torch.randn(1, channel, size, size))
	def forward(self, input):
		batch = input.shape[0]
		out = self.input.repeat(batch, 1, 1, 1)
		return out
class ConvLayer(nn.Sequential):
	def __init__(self, in_channel, out_channel, kernel_size, \
	downsample = False, blur_kernel = [1, 3, 3, 1], \
	bias = True, activate = 'lrelu'):
		layers = []
		if downsample:
			factor = 2
			p = (len(blur_kernel) - factor) + (kernel_size - 1)
			pad0 = (p + 1) // 2
			pad1 = p // 2
			layers.append(Blur(blur_kernel, pad=(pad0, pad1)))
			stride = 2
			self.padding = 0
		else:
			stride = 1
			self.padding = kernel_size // 2
		if 'sp' in activate.lower():
			layers.append(SpectralNorm(EqualConv2d( \
				in_channel, \
				out_channel, \
				kernel_size, \
				padding = self.padding, \
				stride = stride, \
				bias = bias)))
		else:
			layers.append(EqualConv2d( \
				in_channel, \
				out_channel, \
				kernel_size, \
				padding = self.padding, \
				stride = stride, \
				bias = bias))
			if activate == 'lrelu':
				if bias:
					layers.append(FusedLeakyReLU(out_channel))
				else:
					layers.append(ScaledLeakyReLU(0.2))
		super(ConvLayer, self).__init__(*layers)
class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, blur_kernel = [1, 3, 3, 1], downsample = True):
		super(ResBlock, self).__init__()
		self.conv1 = ConvLayer(in_channel, in_channel, 3)
		self.conv2 = ConvLayer(in_channel,out_channel, 3, downsample = downsample)
		self.skip = ConvLayer(in_channel, out_channel, 1, downsample = downsample, \
			activate = False, bias = False)
	def forward(self, input):
		out = self.conv1(input)
		out = self.conv2(out)
		skip = self.skip(input)
		out = (out + skip) / math.sqrt(2)
		return out
if __name__ == '__main__':
	pass
