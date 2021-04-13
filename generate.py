import os
import time
import math
import torch
import argparse
from model import Generator
from torchvision import utils
try:
	from tqdm import tqdm
except ImportError as e:
	def tqdm(lst):
		return lst
def generate(args, g_ema, mean_latent, folder = 'sample'):
	bits = max(int(math.floor(math.log(max(args.pics,1))/math.log(10)))+1, 6)
	fmt = '%%0%dd.png' % bits
	with torch.no_grad():
		g_ema.eval()
		for i in tqdm(range(args.pics)):
			sample_z = torch.randn(args.sample, args.latent, device = args.device)
			sample, _ = g_ema([sample_z], \
				truncation = args.truncation, \
				truncation_latent = mean_latent)
			utils.save_image(sample, \
				os.path.join(folder, fmt % i), \
				nrow = 1, normalize = True, range = (-1, 1))
	return args.pics
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Generate samples from the generator')
	parser.add_argument('--size', type = int, default = 1024, \
		help =	'output image size of the generator [%(default)d]')
	parser.add_argument('--sample', type = int, default = 1, \
		help =	'number of samples to be generated for each image [%(default)d]')
	parser.add_argument('--pics', type = int, default = 20, \
		help =	'number of images to be generated [%(default)d]')
	parser.add_argument('--truncation', type = float, default = 1, \
		help =	'truncation ratio [%(default)f]')
	parser.add_argument('--truncation_mean', type = int, default = 4096, \
		help =	'number of vectors to calculate mean for the truncation [%(default)d]')
	parser.add_argument('--ckpt', type = str, default = 'stylegan2-ffhq-config-f.pt', \
		help =	'path to the model checkpoint [%(default)s]')
	parser.add_argument('--output', type = str, default = 'sample', \
		help =	'output folder [%(default)s]')
	parser.add_argument('--gpu', type = int, default = 0, \
		help =	'use gpu id to test')
	parser.add_argument('--seed', type = int, default = -1, \
		help =	'random seed for generating images')
	parser.add_argument('--channel_multiplier', type = int, default = 2, \
		help =	'channel multiplier of the generator. config-f = 2, else = 1')
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
	args.latent = 512
	args.n_mlp = 8
	g_ema = Generator(args.size, args.latent, args.n_mlp, \
		channel_multiplier = args.channel_multiplier).to(args.device)
	ckpt = torch.load(args.ckpt)
	if 'g_ema' in ckpt.keys():
		g_ema.load_state_dict(ckpt['g_ema'])
	elif 'g' in ckpt.keys():
		g_ema.load_state_dict(ckpt['g'])
	else:
		exit(1)
	if args.truncation < 1:
		with torch.no_grad():
			mean_latent = g_ema.mean_latent(args.truncation_mean)
	else:
		mean_latent = None
	generate(args, g_ema, mean_latent)
