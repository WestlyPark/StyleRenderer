import numpy as np
import torch
import sys
import os
from torch.nn import functional as F
from torch.autograd import Function
from torch.utils.cpp_extension import load

module_path = os.path.dirname(__file__)
rasterize_op = load( \
	'rasterize', \
	sources = [ \
		os.path.join(module_path, 'rasterize.cpp'), \
		os.path.join(module_path, 'rasterize.cu') \
	] \
)
class Rasterize(Function):
	@staticmethod
	def forward(ctx, v, tex, tri, h, w, perspective, eps):
		v = v.contiguous()
		tri = tri.contiguous()
		ind, coeff = rasterize_op.forward(v, tri, h, w, perspective, eps)
		h = int(ind.shape[-3])
		w = int(ind.shape[-2])
		c = 1 if len(tex.shape) == len(v.shape)-1 else int(tex.shape[-1])
		ctx.save_for_backward(v, tex, ind, coeff)
		ctx.perspective = perspective
		ctx.eps = eps
		tex_rast = torch.index_select(tex.view(-1,c), 0, ind.view(-1))
		if len(tex.shape) == len(v.shape) - 1:
			sz = [int(i) for i in ind.shape]
			axis = -1
		else:
			sz = [int(i) for i in ind.shape] + [-1]
			coeff = coeff.unsqueeze(-1)
			axis = -2
		return torch.sum(tex_rast.view(sz) * coeff, axis)
	@staticmethod
	def backward(ctx, grad_out):
		v, tex, ind, coeff = ctx.saved_tensors
		b = int(tex.shape[0])
		n = int(tex.shape[1])
		h = int(ind.shape[-3])
		w = int(ind.shape[-2])
		c = 1 if len(tex.shape) == len(v.shape)-1 else int(tex.shape[-1])
		def build_sparse_index(ind):
			i = torch.cat((ind.view(1,-1), \
				torch.LongTensor(range(b*h*w*3)).view(1,-1).to(ind.device)), 0)
			I = torch.sparse.FloatTensor(i, \
				torch.ones(b*h*w*3).to(ind.device), torch.Size([b*n, b*h*w*3]))
			return I
		if ctx.needs_input_grad[0]:
			grad_coeff = rasterize_op.backward(v, ind, ctx.perspective, ctx.eps)
			tex_rast = torch.index_select(tex.view(-1,c), 0, ind.view(-1))
			if len(tex.shape) == len(v.shape) - 1:
				diff = grad_out.unsqueeze(-1) * tex_rast.view(b,h,w,3)
			else:
				diff = torch.sum(grad_out.unsqueeze(-2)*tex_rast.view(b,h,w,3,c),-1)
			diff = torch.matmul(diff.view(-1,1,3), \
				grad_coeff.view(-1,3,9)).view(b,h,w,3,3)
			I = build_sparse_index(ind)
			grad_v = torch.sparse.mm(I, diff.view(-1,3) \
				.type(torch.float32)).type(diff.dtype)
			grad_v = grad_v.view(v.shape)
		else:
			grad_v = None
			I = None
		if ctx.needs_input_grad[1]:
			if I is None:
				I = build_sparse_index(ind)
			if len(tex.shape) == len(v.shape) - 1:
				diff = grad_out.unsqueeze(-1) * coeff
			else:
				diff = grad_out.unsqueeze(-2) * coeff.unsqueeze(-1)
			grad_t = torch.sparse.mm(I, diff.view(-1,c) \
				.type(torch.float32)).type(diff.dtype)
			grad_t = grad_t.view(tex.shape)
		else:
			grad_t = None
		return grad_v,grad_t,None,None,None,None,None
def rasterize(v, tex, tri, h = 256, w = 0, perspective = False, eps = 1e-6):
	return Rasterize.apply(v, tex, tri, h, w, perspective, eps)
if __name__ == '__main__':
	use_cuda = False
	v = np.array([ \
		[-1,-1, 0], \
		[-1, 1, 0], \
		[ 1, 0, 0]], np.float64)
	f = np.array([ \
		[2, 1, 0]], np.int64)
	t = np.array([ \
		[1, 0], \
		[0, 1], \
		[0, 0]], np.float64)
	v = torch.autograd.Variable(torch.from_numpy(v[np.newaxis]), requires_grad = True)
	f = torch.from_numpy(f)
	t = torch.autograd.Variable(torch.from_numpy(t[np.newaxis]), requires_grad = True)
	if use_cuda:
		v = v.cuda()
		f = f.cuda()
		t = t.cuda()
	o = rasterize(v, t, f, 5)
	for i in range(int(t.shape[-1])):
		print(o[0,:,:,i].detach().cpu().numpy()); print();
	test = torch.autograd.gradcheck(lambda x: rasterize(x[:,:,:3], x[:,:,3:], f, 5), \
		torch.cat((v, t), -1), eps = 1e-6, atol = 1e-6)
	print(test)
