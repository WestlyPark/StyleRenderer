from scipy import linalg
import numpy as np
import argparse
import pickle
import torch
import time
from torch import nn
try:
	from tqdm import tqdm
except ImportError as e:
	def tqdm(lst):
		return lst
from model import Generator
from calc_inception import load_patched_inception_v3
@torch.no_grad()
def extract_feature_from_samples(generator, inception, truncation, truncation_latent, \
		batch_size, n_sample, device):
	n_batch = n_sample // batch_size
	resid = n_sample - (n_batch * batch_size)
	batch_sizes = [batch_size] * n_batch + [resid]
	features = []
	for batch in tqdm(batch_sizes):
		latent = torch.randn(batch, 512, device = device)
		img, _ = g([latent], truncation=truncation, truncation_latent=truncation_latent)
		feat = inception(img)[0].view(img.shape[0], -1)
		features.append(feat.to('cpu'))
	features = torch.cat(features, 0)
	return features

def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps = 1e-6):
	cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp = False)
	if not np.isfinite(cov_sqrt).all():
		print("product of cov matrices is singular")
		offset = np.eye(sample_cov.shape[0]) * eps
		cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))
	if np.iscomplexobj(cov_sqrt):
		if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
			m = np.max(np.abs(cov_sqrt.imag))
			raise ValueError("Imaginary component %f" % m)
		cov_sqrt = cov_sqrt.real
	mean_diff = sample_mean - real_mean
	mean_norm = mean_diff @ mean_diff
	trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)
	fid = mean_norm + trace
	return fid
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Calculate FID scores')
	parser.add_argument('--truncation', type = float, default = 1, \
		help =	'truncation factor [%(default)f]')
	parser.add_argument('--truncation_mean', type = int, default = 4096, \
		help =	'number of samples to calculate mean for truncation [%(default)d]')
	parser.add_argument('--batch', type = int, default = 64, \
		help =	'batch size for the generator [%(default)d]')
	parser.add_argument('--n_sample', type = int, default = 50000, \
		help =	'number of the samples for calculating FID')
	parser.add_argument('--size', type = int, default = 256, \
		help =	'image sizes for generator [%(default)d]')
	parser.add_argument('--inception', type = str, required = True, \
		help =	'path to precomputed inception embedding')
	parser.add_argument('ckpt', metavar = 'CHECKPOINT', \
		help =	'path to generator checkpoint')
	parser.add_argument('--gpu', type = int, default = 0, \
		help =	'use gpu id to test')
	parser.add_argument('--seed', type = int, default = -1, \
		help =	'random seed for generating images')
	args = parser.parse_args()
	if args.seed < 0:
		args.seed = int(time.time())
	torch.manual_seed(args.seed)
	if torch.cuda.is_available() and \
	args.gpu >= 0 and args.gpu < torch.cuda.device_count():
		torch.cuda.manual_seed(args.seed)
		args.device = 'cuda:%d' % args.gpu
	else:
		args.device = 'cpu'
	ckpt = torch.load(args.ckpt)
	g = Generator(args.size, 512, 8).to(args.device)
	g.load_state_dict(ckpt["g_ema"])
	g = nn.DataParallel(g)
	g.eval()
	if args.truncation < 1:
		with torch.no_grad():
			mean_latent = g.mean_latent(args.truncation_mean)
	else:
		mean_latent = None
	inception = nn.DataParallel(load_patched_inception_v3()).to(device)
	inception.eval()
	features = extract_feature_from_samples(g, inception, args.truncation, \
		mean_latent, args.batch, args.n_sample, args.device).numpy()
	print('extracted %d features' % features.shape[0])
	sample_mean = np.mean(features, 0)
	sample_cov = np.cov(features, rowvar=False)
	with open(args.inception, 'rb') as f:
		embeds = pickle.load(f)
		real_mean = embeds["mean"]
		real_cov = embeds["cov"]
	fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
	print('fid: %f' % fid)
