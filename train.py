import os
import sys
import time
import numpy as np
import importlib
import argparse

import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms, utils
try:
	from tqdm import tqdm
except ImportError as e:
	def tqdm(lst,**kwargs):
		return lst
try:
	import wandb
except ImportError as e:
	wandb = None
from model import Generator, GeneratorWithMap, Discriminator
from face_model import LinearMorphableModel, load_bfm
from dataset import MultiResolutionDataset
from utils_3d import augment, random_apply_pose3D, mesh_point_normal
timer = [time.time()]
def tic():
	global timer
	timer += [time.time()]
def toc(show_tag = None):
	global timer
	t = time[-1] if len(timer) > 0 else 0
	t = time.time() - t
	if not show_tag is None:
		print('%s %f s' % (show_tag, t))
		sys.stdout.flush()
	timer = time[:-1] if len(timer) > 1 else [timer.time()]
	return t
def log_init(folder = '.', keys = [], size = 0):
	try:
		import tensorflow as tf
		fid = tf.summary.FileWriter(folder)
		var = {}
		for k in keys:
			if 'img' in k.lower():
				var[k] = tf.placeholder(tf.float32,(1,size,size,3))
			else:
				var[k] = tf.placeholder(tf.float32,())
				tf.summary.scalar(k, var[k])
		config = tf.ConfigProto()
		sess = tf.Session(config = config)
		return (fid, tf.summary.merge_all(), sess, var)
	except ModuleNotFoundError as e:
		try:
			import tensorboardX as tb
			fid = tb.SummaryWriter(logdir = folder)
		except ModuleNotFoundError as e::
			fid = open(os.path.join(folder, 'logger.txt'), 'w')
			return fid
def log_write(fid, data, step = 0):
	try:
		import tensorflow as tf
		fid, summary, sess, var = fid
		feed_dict = {}
		for k in var.keys():
			if k in data.keys():
				feed_dict[var[k]] = data[k]
			else:
				feed_dict[var[k]] = 0
		summary = sess.run(summary, feed_dict = feed_dict)
		fid.add_summary(summary, step)
	except ModuleNotFoundError as e:
		try:
			fid.add_scalars('train', data, step)
		except AttributeError as e:
			s = 'Step: %d\t' % step
			for i,(k, v) in enumerate(data.items()):
				try:
					v = float(v)
					s += '%s:%3.5f' % (k, float(v))
					if i != len(data) - 1:
						s += '\t'
					else:
						s += '\n'
				except:
					continue
			fid.write(s)
			fid.flush()
def log_close(fid):
	try:
		import tensorflow as tf
		fid, summary, sess, var = fid
		sess.close()
	except:
		if fid is not None:
			fid.close()
def requires_grad(model, flag = True):
	for p in model.parameters():
		p.requires_grad = True if flag else False
def accumulate(model1, model2, decay = 0.999):
	par1 = dict(model1.named_parameters())
	par2 = dict(model2.named_parameters())
	for k in par1.keys():
		par1[k].data.mul_(decay).add_(par2[k].data, alpha = 1 - decay)
def d_logistic_loss(real_pred, fake_pred):
	real_loss = F.softplus(-real_pred)
	fake_loss = F.softplus(fake_pred)

	return real_loss.mean() + fake_loss.mean()
def d_r1_loss(real_pred, real_img):
	grad_real, = autograd.grad( \
		outputs = real_pred.sum(), inputs = real_img, create_graph = True)
	grad_penalty = (grad_real * grad_real).reshape(grad_real.shape[0], -1).sum(1).mean()
	return grad_penalty
def g_nonsaturating_loss(fake_pred):
	loss = F.softplus(-fake_pred).mean()
	return loss
def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01, lambda_ = 1.):
	noise = torch.randn_like(fake_img) / np.sqrt(fake_img.shape[2] * fake_img.shape[3])
	if not isinstance(latents, list):
		latents = [latents]
	lambda_ = list(np.reshape(lambda_,-1))
	lambda_+= [1] * (len(latents) - len(lambda_))
	path_lengths = 0
	grads = autograd.grad( \
		outputs= (fake_img*noise).sum(), inputs = latents, \
		create_graph = True, allow_unused = True)
	for l, grad in zip(lambda_,grads):
		if grad is not None:
			grad = grad.view(int(grad.shape[0]),-1)
			path_lengths += torch.sqrt((grad * grad).sum(1)) * l
	path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)
	path_penalty = (path_lengths - path_mean).pow(2).mean()
	return path_penalty, path_mean.detach(), path_lengths
def make_noise(batch, latent_dim, n_noise, device):
	if n_noise <= 0:
		return torch.randn(batch, latent_dim, device = device)
	else:
		return torch.randn(n_noise, batch, latent_dim, device = device).unbind(0)
def mixing_noise(batch, latent_dim, prob, device):
	if prob > 0 and np.random.rand() < prob:
		return make_noise(batch, latent_dim, 2, device)
	else:
		return [make_noise(batch, latent_dim, 0, device)]
def initialize(args):
	if args.seed < 0:
		args.seed = int(time.time())
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if torch.cuda.is_available() and not args.cpu:
		torch.cuda.manual_seed(args.seed)
		tmp = args.gpu.split(',')
		args.gpu = []
		for i in tmp:
			try:
				i = max(int(i),0)
				if i < torch.cuda.device_count():
					args.gpu.append(i)
			except ValueError as e:
				continue
		args.gpu = list(np.unique(args.gpu))
		args.distributed = (args.local_rank >= 0 and len(args.gpu) > 1)
		try:
			import distributed
		except (ModuleNotFoundError,ImportError) as e:
			args.distributed = False
		if len(args.gpu) > 0:
			if args.distributed:
				args.device = 'cuda:%d' % args.gpu[args.locak_rank]
				distributed.initialize(args)
			else:
				args.device = 'cuda'
	else:
		args.gpu = []
		args.device = 'cpu'
		args.distributed = False
	return args
def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, \
		face, extra_model, logger = None):
	def sample_data():
		while True:
			for batch in loader:
				yield batch
	def reduce_sum(tensor):
		if args.distributed:
			import distributed
			return distributed.reduce_sum(tensor)
		else:
			return tensor
	def get_world_size():
		if args.distributed:
			import distributed
			return distributed.get_world_size()
		else:
			return 1
	pbar = range(args.iter)
	if args.rank == 0:
		pbar = tqdm(pbar, initial = args.start_iter, \
			dynamic_ncols = True, smoothing = 0.01)
	mean_path_length = 0
	d_loss_val = 0
	r1_loss = torch.tensor(0.0, device = args.device)
	g_loss_val = 0
	path_loss = torch.tensor(0.0, device = args.device)
	path_lengths = torch.tensor(0.0, device = args.device)
	mean_path_length_avg = 0
	loss_dict = {}
	if args.distributed:
		g_module = generator.module
		d_module = discriminator.module
		f_module = face[0].module
		extra_module = {k:model.module for k,model in extra_model.keys()}
		tri = face[1]
	else:
		g_module = generator
		d_module = discriminator
		f_module = face[0]
		extra_module = extra_model
		tri = face[1]
	accum = 0.5 ** (32 / (10 * 1000))
	ada_augment = torch.tensor([0.0, 0.0], device = args.device)
	ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
	ada_aug_step = args.ada_target / args.ada_length
	r_t_stat = 0
	with torch.no_grad():
		sample_z = torch.randn(args.n_sample, args.latent, device = args.device)
		sample_face= f_module.random_input(args.n_sample).to(args.device)
		sample_v = f_module(sample_face)
		sample_n = mesh_point_normal(sample_v, tri)
		sample_v = random_apply_pose3D(v = sample_v)
	bits = max(int(np.floor(np.log(max(args.iter,1))/np.log(10)))+1, 6)
	fmt = '%%0%dd' % bits
	nrow = int(np.ceil(np.sqrt(args.n_sample)))
	while nrow > 1:
		if args.n_sample % nrow == 0:
			break
		else:
			--nrow
	for idx in pbar:
		i = idx + args.start_iter
		if i > args.iter:
			print('Done!'); break
		real_img = next(sample_data()).to(args.device)
		# Optimize D
		requires_grad(generator, False)
		requires_grad(discriminator, True)
		noise = mixing_noise(args.batch, args.latent, args.mixing, args.device)
		sample_f = f_module.random_input(args.batch).to(args.device)
		with torch.no_grad():
			vert = random_apply_pose3D(v = f_module(sample_f))
			norm = mesh_point_normal(vert, tri)
		fake_img,_,_ = generator(noise, (vert,norm,tri))
		if args.augment:
			real_img_aug = augment(real_img, ada_aug_p)
			fake_img = augment(fake_img, ada_aug_p)
		else:
			real_img_aug = real_img
		fake_pred = discriminator(fake_img)
		real_pred = discriminator(real_img_aug)
		d_loss = d_logistic_loss(real_pred, fake_pred)

		loss_dict['d'] = d_loss
		loss_dict['real_score'] = real_pred.mean()
		loss_dict['fake_score'] = fake_pred.mean()

		discriminator.zero_grad()
		d_loss.backward()
		d_optim.step()
		if args.augment and args.augment_p <= 0:
			ada_augment_data = torch.tensor( \
				(torch.sign(real_pred).sum().item(), real_pred.shape[0]),\
				device = args.device)
			ada_augment += reduce_sum(ada_augment_data)
			if ada_augment[1] > 255:
				pred_signs, n_pred = ada_augment.tolist()
				r_t_stat = pred_signs / n_pred
				sign = 1 if r_t_stat > args.ada_target else -1
				ada_aug_p += sign * ada_aug_step * n_pred
				ada_aug_p = min(1, max(0, ada_aug_p))
				ada_augment.mul_(0)
		d_regularize = (i % args.d_reg_every == 0)
		if d_regularize:
			real_img.requires_grad = True
			real_pred = discriminator(real_img)
			r1_loss = d_r1_loss(real_pred, real_img)

			discriminator.zero_grad()
			(args.r1/2*r1_loss*args.d_reg_every + 0*real_pred[0]).backward()
			d_optim.step()
		# Optimize G
		loss_dict['r1'] = r1_loss
		requires_grad(generator, True)
		requires_grad(discriminator, False)
		half_batch = (args.batch+1) // 2
		res_batch = args.batch - half_batch
		if 'feat' in extra_model.keys():
			noise = mixing_noise(half_batch, args.latent, args.mixing, args.device)
			noise = [torch.cat([noi, noi[:res_batch]], 0) for noi in noise]
			sample_f = f_module.random_input(args.batch).to(args.device)
			sample_f[half_batch:,:f_module.dim[0]] = sample_f[:res_batch,:f_moduel.dim[0]]
		else:
			noise = mixing_noise(args.batch, args.latent, args.mixing, args.device)
			sample_f = f_module.random_input(args.batch).to(args.device)
		with torch.no_grad():
			vert = random_apply_pose3D(v = f_module.forward(sample_f))
			norm = mesh_point_normal(vert, tri)
			if 'lmk' in extra_model.keys():
				vert[1::2] = vert[:res_batch*2:2]
				norm[1::2] = norm[:res_batch*2:2]
		fake_img, _, norms = generator(noise, (vert,norm,tri), return_normals = True)
		if args.augment and not 'lmk' in extra_model.keys():
			fake_img = augment(fake_img, ada_aug_p)
		fake_pred = discriminator(fake_img)
		g_loss = g_nonsaturating_loss(fake_pred)
		loss_dict['g'] = g_loss
		if 'sfs' in extra_model.keys():
			norms = norms[-1]
			mask = ((norms*norms).sum(1,keepdim=True) > 1e-3).type(norms.dtype)
			loss_dict['sfs'] = nn.functional.smooth_l1_loss( \
				mask * extra_module['sfs'](fake_img)[0], \
				mask * norms)
			g_loss += loss_dict['sfs'] * .01
		if 'feat' in extra_model.keys():
			feat = extra_module['feat'](fake_img)
			loss_dict['feature'] = torch.mean((feat[:res_batch] - feat[half_batch:])**2)
			g_loss += loss_dict['feature'] * .001
		if 'lmk' in extra_model.keys():
			lmk = extra_module['lmk'](fake_img)
			loss_dict['lmk'] = nn.functional.smooth_l1_loss(lmk[:res_batch*2:2], lmk[1::2])
			g_loss += loss_dict['lmk'] * .00001
		generator.zero_grad()
		g_loss.backward()
		g_optim.step()

		g_regularize = (i % args.g_reg_every == 0)
		if g_regularize:
			path_batch_size = max(1, args.batch//args.path_batch_shrink)
			noise = mixing_noise(path_batch_size, args.latent, \
				args.mixing, args.device)
			v = torch.autograd.Variable(vert[:path_batch_size], requires_grad = True)
			n = torch.autograd.Variable(norm[:path_batch_size], requires_grad = True)
			fake_img, latents, normals = generator(noise, (v, n, tri), \
				return_latents = True, return_normals = True)
			path_loss, mean_path_length, path_lengths = g_path_regularize( \
				fake_img, [latents] + normals, mean_path_length)
			generator.zero_grad()
			weighted_path_loss = args.path_regularize * \
				args.g_reg_every * path_loss
			if args.path_batch_shrink:
				weighted_path_loss += 0 * fake_img[0, 0, 0, 0]
			weighted_path_loss.backward()
			g_optim.step()
			mean_path_length_avg = ( \
				reduce_sum(mean_path_length).item() / get_world_size())
		loss_dict['path'] = path_loss
		loss_dict['path_length'] = path_lengths.mean()

		accumulate(g_ema, g_module, accum)
		loss_reduced = {k: reduce_sum(l).item() / get_world_size() \
			for k, l in loss_dict.items()}
		d_loss_val = loss_reduced['d']
		g_loss_val = loss_reduced['g']
		r1_val = loss_reduced['r1']
		path_loss_val = loss_reduced['path']
		real_score_val = loss_reduced['real_score']
		fake_score_val = loss_reduced['fake_score']
		path_length_val = loss_reduced['path_length']
		if args.rank == 0:
			pbar.set_description(( \
				'd: %.4f; g: %.4f; r1: %.4f; ' + \
				'path: %.4f; mean path: %.4f; augment: %.4f') % (\
				d_loss_val, g_loss_val, r1_val, \
				path_loss_val, mean_path_length_avg, ada_aug_p))
		if args.wandb:
			wandb.log({ \
				'Generator': g_loss_val, \
				'Discriminator': d_loss_val, \
				'Augment': ada_aug_p, \
				'Rt': r_t_stat, \
				'R1': r1_val, \
				'Path Length Regularization': path_loss_val, \
				'Mean Path Length': mean_path_length, \
				'Real Score': real_score_val, \
				'Fake Score': fake_score_val, \
				'Path Length': path_length_val})
		if logger is not None:
			log_write(logger, { \
				'Generator': g_loss_val, \
				'Discriminator': d_loss_val, \
				'Augment': ada_aug_p, \
				'Rt': r_t_stat, \
				'R1': r1_val, \
				'Path Length Regularization': path_loss_val, \
				'Mean Path Length': mean_path_length, \
				'Real Score': real_score_val, \
				'Fake Score': fake_score_val, \
				'Path Length': path_length_val}, i)
		if (i+1) % 100 == 0:
			with torch.no_grad():
				g_ema.eval()
				sample, _, norms = g_ema([sample_z], (sample_v,sample_n,tri), \
					return_normals = True)
				utils.save_image(sample, \
					os.path.join('sample', (fmt + '.png') % i), \
					nrow = nrow, padding = 0, \
					normalize = True, range = (-1, 1))
				utils.save_image(norms[-1], \
					os.path.join('sample', (fmt + '_norm.png') % i), \
					nrow = nrow, padding = 0, \
					normalize = True, range = (-1, 1))
		if (i+1-1) % 10000 == 0:
			torch.save({ \
				'g': g_module.state_dict(), \
				'd': d_module.state_dict(), \
				'g_ema': g_ema.state_dict(), \
				'g_optim': g_optim.state_dict(), \
				'd_optim': d_optim.state_dict(), \
				'args': args, \
				'ada_aug_p': ada_aug_p}, \
				os.path.join('checkpoint', (fmt + '.pt') % i))
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'StyleGAN2 trainer')
	parser.add_argument('path', type = str, help = 'path to the lmdb dataset')
	parser.add_argument('--gpu', type = str, default = '0', \
		help =	'gpu ids to use')
	parser.add_argument('--cpu', action = 'store_true', \
		help =	'use cpu to train')
	parser.add_argument('--iter', type = int, default = 800000, \
		help =	'total training iterations [%(default)ld]')
	parser.add_argument('--bfm', type = str, default = '/data/BaselFaceModel.mat', \
		help =	'Basel Face Model path')
	parser.add_argument('--batch', type = int, default = 16, \
		help =	'batch sizes for each gpus [%(default)d]')
	parser.add_argument('--n_sample', type = int, default = 64, \
		help =	'number of the samples generated to evaluate during training [%(default)d]')
	parser.add_argument('--size', type = int, default = 256, \
		help =	'image sizes for the model [%(default)d]')
	parser.add_argument('--latent', type = int, default = 512, \
		help =	'latent dimension [%(default)d]')
	parser.add_argument('--n_mlp', type = int, default = 8, \
		help =	'latent converting network depth [%(default)d]')
	parser.add_argument('--sfs_net', default = 'thirdparty.face_normals.resnet_unet',\
		help =	'Shape From Shading pretrained Net')
	parser.add_argument('--lmk_net', default = 'thirdparty' + \
			'.pytorch_face_landmark.models.pfld_compressed', \
		help =	'Landmark Regression Network')
	parser.add_argument('--feat_net', default = 'thirdparty' + \
			'.facenet-pytorch.models.inception_resnet_v1', \
		help =	'Face Recognition Network')
	parser.add_argument('--r1', type = float, default = 10, \
		help =	'weight of the r1 regularization [%(default)f]')
	parser.add_argument('--path_regularize', type = float, default = 2, \
		help =	'weight of the path length regularization [%(default)f]')
	parser.add_argument('--path_batch_shrink', type = int, default = 2, \
		help =	'batch size reducing factor for the path length regularization'+ \
			' (reduce memory consumption)')
	parser.add_argument('--d_reg_every', type = int, default = 16, \
		help =	'interval of the applying r1 regularization [%(default)d]')
	parser.add_argument('--g_reg_every', type = int, default = 4, \
		help =	'interval of the applying path length regularization')
	parser.add_argument('--mixing', type = float, default = .9, \
		help =	'probability of latent code mixing [%(default)f]')
	parser.add_argument('--ckpt', type = str, default = '', \
		help =	'path to the checkpoints to resume training')
	parser.add_argument('--lr', type = float, default = 0.002, \
		help =	'learning rate [%(default)f]')
	parser.add_argument('--channel_multiplier', type = int, default = 2, \
		help =	'channel multiplier factor for the model. config-f = 2, else = 1')
	parser.add_argument('--wandb', action = 'store_true', \
		help =	'use weights and biases logging')
	parser.add_argument('--local_rank', type = int, default = -1, \
		help =	'local rank for distributed training')
	parser.add_argument('--augment', action = 'store_true', \
		help =	'apply non leaking augmentation')
	parser.add_argument('--augment_p', type = float, default = 0, \
		help =	'probability of applying augmentation. ' + \
			'0 = use adaptive augmentation')
	parser.add_argument('--ada_target', type = float, default = .6, \
		help =	'target augmentation probability for adaptive augmentation ')
	parser.add_argument('--ada_length', type = int, default = 500 * 1000, \
		help =	'target duraing to reach augmentation probability ' + \
			'for adaptive augmentation')
	parser.add_argument('--ada_every', type = int, default = 256, \
		help =	'probability update interval of the adaptive augmentation')
	parser.add_argument('--seed', type = int, default = -1, \
		help =	'random seed for training and sample')
	args = parser.parse_args()

	args = initialize(args)
	args.start_iter = 0
	generator = GeneratorWithMap(args.size, args.latent, args.n_mlp, \
		channel_multiplier = args.channel_multiplier).to(args.device)
	discriminator = Discriminator(args.size, \
		channel_multiplier = args.channel_multiplier).to(args.device)
	g_ema = GeneratorWithMap(args.size, args.latent, args.n_mlp, \
		channel_multiplier = args.channel_multiplier).to(args.device)
	face, tri = load_bfm(args.bfm)
	face = face.to(args.device)
	tri = tri.to(args.device)
	g_ema.eval()
	accumulate(g_ema, generator, 0)
	extra_model = {}
	try:
		model = importlib.import_module(args.sfs_net)
		model = model.ResNetUNet(n_class = 3).to(args.device)
		model.load_state_dict(torch.load(os.path.join( \
			*(list(args.sfs_net.split('.'))+['..','data','model.pth']))))
		model.eval()
		extra_model['sfs'] = model
	except ModuleNotFoundError:
		print('Not loading sfs_net'); sys.stdout.flush()
	try:
		model = importlib.import_module(args.lmk_net)
		model = model.PFLDInference().to(args.device)
		model.load_state_dict(torch.load(os.path.join( \
			*(list(args.lmk_net.split('.'))[:-1]+ \
			['..','checkpoint','pfld_model_best.pth.tar'])))['state_dict'])
		model.eval()
		extra_model['lmk'] = model
	except ModuleNotFoundError:
		print('Not loading landmark net'); sys.stdout.flush()
	try:
		model = importlib.import_module(args.feat_net)
		model = model.InceptionResnetV1(pretrained = 'vggface2').to(args.device)
		model.eval()
		extra_model['feat'] = model
	except ModuleNotFoundError:
		print('Not loading face recognition model'); sys.stdout.flush()
	g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
	d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
	g_optim = optim.Adam(generator.parameters(), \
		lr = args.lr * g_reg_ratio,
		betas = (0 ** g_reg_ratio, 0.99 ** g_reg_ratio))
	d_optim = optim.Adam(discriminator.parameters(), \
		lr = args.lr * d_reg_ratio,
		betas = (0 ** d_reg_ratio, 0.99 ** d_reg_ratio))
	if os.path.exists(args.ckpt):
		print("load model:", args.ckpt)
		ckpt = torch.load(args.ckpt, map_location = lambda storage, loc: storage)
		try:
			ckpt_name = os.path.basename(args.ckpt)
			args.start_iter = int(os.path.splitext(ckpt_name)[0])
		except ValueError:
			args.start_iter = 0
		if 'g' in ckpt.keys():
			generator.load_state_dict(ckpt['g'])
		if 'd' in ckpt.keys():
			discriminator.load_state_dict(ckpt['d'])
		if 'g_ema' in ckpt.keys():
			g_ema.load_state_dict(ckpt['g_ema'])
		elif 'g' in ckpt.keys():
			g.load_state_dict(ckpt['g'])
		if 'g_optim' in ckpt.keys():
			g_optim.load_state_dict(ckpt['g_optim'])
		if 'd_optim' in ckpt.keys():
			d_optim.load_state_dict(ckpt['d_optim'])
	transform = transforms.Compose([ \
		transforms.RandomHorizontalFlip(), \
		transforms.ToTensor(), \
		transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5),inplace = True)])
	dataset = MultiResolutionDataset(args.path, transform, args.size)
	if args.distributed:
		from distributed import construct_ddp, get_rank, get_dataloader
		args.rank = get_rank()
		loader = get_dataloader(dataset)
		generator = construct_ddp(generator)
		discriminator = construct_ddp(discriminator)
		face = construct_ddp(face)
		extra_model = {k:construct_ddp(model) for k,model in extra_model.items()}
	else:
		args.rank = 0
		loader = data.DataLoader(dataset, batch_size = args.batch, \
			sampler = data.RandomSampler(dataset), drop_last = True)
	if wandb is None: args.wandb = False
	logger = None
	if args.rank == 0:
		if args.wandb:
			wandb.init(project = 'stylegan 2')
		else:
			logger = log_init(keys = ['Generator', 'Discriminator', \
				'Augment', 'Rt', 'R1', 'Path Length Regularization', \
				'Mean Path Length', 'Real Score', 'Fake Score', \
				'Path Length'])
	train(args, loader, generator, discriminator, \
		g_optim, d_optim, g_ema, (face,tri), extra_model, logger)
	if logger is not None:
		log_close(logger)
