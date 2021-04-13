import time
import torch
import argparse
import numpy as np
from torch.nn import functional as F
try:
	from tqdm import tqdm
except ImportError as e:
	def tqdm(lst):
		return lst
import lpips
from model import Generator
from utils_3d import normalize
def lerp(w, *args):
	if w.shape[-1] == len(args)-1:
		w = torch.cat((1 - torch.sum(w,-1,keepdim=True), w), -1)
	else:
		w = normalize(w, -1, 'L1')
	return sum([args[i] * w[...,i:i+1] for i in range(len(args))])
def lerp_norm(w, *args):
	args = [normalize(a, -1, 'L2') for a in args]
	return	normalize(lerp(w, *args), -1, 'L2')
class SLerp(torch.autograd.Function):
	@staticmethod
	def forward(ctx, w, *args, iterations = 3, eps = 1e-8):
		c = sum([args[i] * w[...,i:i+1] for i in range(len(args))])
		c = normalize(c, -1, 'L2', eps)
		for _ in range(int(iterations)):
			a = torch.cat([torch.acos( \
				torch.clamp((args[i]*c).sum(-1), \
				min = -1, max = 1)).unsqueeze(-1), \
				for i in range(len(args))], -1)
			w_= w * torch.where(a <= eps, 1+a*a/6, a/torch.sin(a))
			c = sum([args[i] * w_[...,i:i+1] for i in range(len(args))])
			c = normalize(c, -1, 'L2', eps)
		ctx.eps = eps
		ctx.save_for_backward(w, c, *args)
		return c
	@staticmethod
	def backward(ctx, grad_c):
		w = ctx.saved_tensors[0]
		c = ctx.saved_tensors[1]
		num = int(w.shape[-1])
		dim = int(c.shape[-1])
		args = torch.cat([a.unsqueeze(-2)for a in ctx.saved_tensors[2:]],-2)
		grad_arg = [None for i in range(num)]
		cs = torch.clamp((args*c.unsqueeze(-2)).sum(-1), min=-1, max=1)
		a  = torch.acos(cs)
		sn = torch.sin(a)
		mul= torch.where(sn <= ctx.eps, 1+a*a/6, a/sn)
		c_ = (args * (w * mul).unsqueeze(-1)).sum(-2)
		norm=torch.sqrt((c_ * c_).sum(-1)).view(-1,1,1)
		dw = w * torch.where(sn <= ctx.eps, -(1+2*a*a/5)/3, (a*cs-sn)/(sn*sn*sn))
		dW =	dw.unsqueeze(-1).unsqueeze(-1) * \
			torch.matmul(args.view(-1,dim,1),args.view(-1,1,dim)) \
			.view([int(i) for i in args.shape]+[dim])
		K = torch.cat([torch.cat([ \
			norm * torch.eye(dim, \
				dtype = c.dtype, \
				device = c.device)
				.unsqueeze(0) - dW.sum(-3), \
			c.view(-1,dim,1)], -1), torch.cat([ \
			c.view(-1,1,dim), \
			torch.zeros_like(norm)], -1)], -2)
		Kinv = torch.inverse(K)
		if ctx.needs_input_grad[0]:
			grad_w = torch.matmul(Kinv[:,:dim,:dim], \
				mul.unsqueeze(-2) * args.permute(0,2,1))
			grad_w = (grad_w * grad_c.unsqueeze(-1)).sum(-2)
		else:
			grad_w = None
		mul = (w * mul).view(-1,num,1)
		dw  = dw.view(-1,num,1)
		c   = c.view(-1,1,dim)
		for i in range(num):
			if ctx.needs_input_grad[i+1]:
				grad_arg[i] = mul[:,i:i+1,:] * Kinv[:,:dim,:dim] + \
					dw[:,i:i+1,:] * torch.matmul(torch.matmul( \
					Kinv[:,:dim,:dim], args[:,i].unsqueeze(-1)), c)
				grad_arg[i] = (grad_arg[i] * grad_c.unsqueeze(-1)).sum(-2)
		return tuple([grad_w] + grad_arg + \
			[None]*(len(ctx.needs_input_grad)-1-len(grad_arg)))
def slerp(w, *args):
	args = [normalize(a, -1, 'L2') for a in args]
	if len(args) == 2:
		if w.shape[-1] > 1:
			w = w[...,1:2] / torch.sum(w,-1,keepdim=True)
		a = torch.acos((args[0]*args[1]).sum(-1,keepdim=True))
		w0= torch.sin(a *(1-w))
		w1= torch.sin(a * w)
		c = w0 * args[0] + w1 * args[1]
		return normalize(c, -1, 'L2')
	elif w.shape[-1] == len(args)-1:
		w = torch.cat((1 - torch.sum(w,-1,keepdim=True), w), -1)
 	return SLerp.apply(w, *args)
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Perceptual Path Length calculator')
	parser.add_argument('--space', choices = ['z', 'w'], \
		help =	'space that PPL calculated with')
	parser.add_argument('--batch', type = int, default = 64, \
		help =	'batch size for the models [%(default)d]')
	parser.add_argument('--n_sample', type = int, default = 5000, \
		help =	'number of the samples for calculating PPL [%(default)d]')
	parser.add_argument('--size', type = int, default = 256, \
		help =	'output image sizes of the generator [%(default)d]')
	parser.add_argument('--eps', type = float, default = 1e-4, \
		help =	'epsilon for numerical stability [%(default)f]')
	parser.add_argument('--crop', action = 'store_true', \
		help =	'apply center crop to the images')
	parser.add_argument('--sampling', default = 'end', choices = ['end', 'full'], \
		help =	'set endpoint sampling method')
	parser.add_argument('--gpu', type = int, default = 0, \
		help =	'use gpu id to test')
	parser.add_argument('--seed', type = int, default = -1, \
		help =	'random seed for sample')
	parser.add_argument('ckpt', metavar = 'CHECKPOINT', \
		help =	'path to the model checkpoints')
	args = parser.parse_args()
	if args.seed < 0:
		args.seed = int(time.time())
	torch.manual_seed(args.seed)
	if torch.cuda.is_available() and \
	args.gpu >= 0 and args.gpu < torch.cuda.device_count():
		torch.cuda.manual_seed(args.seed)
		device = 'cuda:%d' % args.gpu
	else:
		device = 'cpu'
	latent_dim = 512
	ckpt = torch.load(args.ckpt)
	g = Generator(args.size, latent_dim, 8).to(device)
	if 'g_ema' in ckpt.keys():
		g.load_state_dict(ckpt['g_ema'])
	elif 'g' in ckpt.keys():
		g.load_state_dict(ckpt['g'])
	else:
		exit(1)
	g.eval()
	percept = lpips.PerceptualLoss(
		model = 'net-lin', net = 'vgg', use_gpu = device.startswith('cuda'))
	distances = []
	n_batch = args.n_sample // args.batch
	resid = args.n_sample - (n_batch * args.batch)
	batch_sizes = [args.batch] * n_batch + [resid]
	with torch.no_grad():
		for batch in tqdm(batch_sizes):
			noise = g.make_noise()
			inputs = torch.randn([batch * 2, latent_dim], device = device)
			if args.sampling == 'full':
				t = torch.rand(batch, device = device)
			else:
				t = torch.zeros(batch, device = device)
			if 'w' in args.space.lower():
				latent = g.get_latent(inputs)
				latent_t0, latent_t1 = latent[::2], latent[1::2]
				latent_e0 = lerp(t[:, None],latent_t0,latent_t1)
				latent_e1 = lerp(t[:, None]+args.eps,latent_t0,latent_t1)
				latent_e = torch.stack([latent_e0,latent_e1],1).view(*latent.shape)
			elif 'z' in args.space.lower():
				inputs_t0, inputs_t1 = inputs[::2], inputs[1::2]
				latent_t0 = g.get_latent(slerp(t[:,None],inputs_t0,inputs_t1))
				latent_t1 = g.get_latent(slerp(t[:,None]+args.eps,inputs_t0,inputs_t1))
				latent_e = torch.stack([latent_t0,latent_t1],1).view(*latent.shape)
			else:
				latent_e = g.get_latent(inputs)
			image, _ = g([latent_e], input_is_latent = True, noise = noise)
			if args.crop:
				c = image.shape[2] // 8
				image = image[:, :, c*3:c*7, c*2:c*6]
			factor = image.shape[2] // 256
			if factor > 1:
				image = F.interpolate(image, size = (256,256), \
					mode = 'bilinear', align_corners = False)
			dist = percept(image[::2], image[1::2]).view(image.shape[0]//2)/ \
				(args.eps*args.eps)
			distances.append(dist.to('cpu').numpy())
	distances = np.concatenate(distances, 0)
	lo = np.percentile(distances, 1, interpolation = 'lower')
	hi = np.percentile(distances, 99, interpolation = 'higher')
	filtered_dist = np.extract(np.logical_and(lo<=distances, distances<=hi), distances)
	print("ppl:", filtered_dist.mean())
