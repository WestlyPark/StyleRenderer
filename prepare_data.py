from functools import partial
from io import BytesIO
import numpy as np
import argparse
import math
import lmdb
import sys
import os
try:
	from tqdm import tqdm
except ImportError as e:
	def tqdm(lst):
		return lst
try:
	from PIL import Image
	from torchvision.transforms import functional as trans_fn
	def read_img(file_name):
		try:
			img = Image.open(file_name).convert('RGB')
			return img
		except (FileNotFoundError,OSError) as e:
			return None
	def resize_img(img, size, resample = ''):
		if 'near' in resample.lower():
			resample = Image.NEAREST
		elif 'area' or 'box' in resample.lower():
			resample = Image.BOX
		elif 'ham' in resample.lower():
			resample = Image.HAMMING
		elif 'lin' in resample.lower():
			resample = Image.BILINEAR
		elif 'cub' in resample.lower():
			resample = Image.BICUBIC
		else:
			resample = Image.LANCZOS
		return trans_fn.center_crop(trans_fn.resize(img, size, resample), size)
	def decode_img(img_bytes):
		buffer = BytesIO(img_bytes)
		return Image.open(buffer)
	def save_img(img, file_name = None, quality = 100):
		if isinstance(img, np.ndarray):
			img = Image.fromarray(img)
		if file_name is None:
			buffer = BytesIO()
			img.save(buffer, format = 'jpeg', quality = quality)
			return buffer.getvalue()
		else:
			img.save(file_name, quality = quality)
			return os.path.exists(file_name)
except (ModuleNotFoundError,ImportError) as e:
	import cv2
	def read_img(file_name):
		img = cv2.imread(file_name)
		if img is not None:
			return img[:,:,::-1]
		else:
			return None
	def resize_img(img, size, resample):
		size = np.reshape(size,-1).astype(np.uint32)
		if len(size) == 1:
			h = w = size[0]
		else:
			h, w = size[:2]
		if 'near' in resample.lower():
			resample = cv2.INTER_NEAREST
		elif 'area' or 'box' in resample.lower():
			resample = cv2.INTER_AREA
		elif 'lin' in resample.lower():
			resample = cv2.INTER_LINEAR
		elif 'cub' in resample.lower():
			resample = cv2.INTER_CUBIC
		else:
			resample = cv2.INTER_LANCZOS4
		return cv2.resize(img, (w,h), interpolation = resample)
	def decode_img(img_bytes):
		file_bytes = np.asarray(bytearray(img_bytes.read()), dtype = np.uint8)
		return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
	def save_img(img, file_name = None):
		if file_name is None:
			success, buffer = cv2.imencode('.jpg', img)
			return BytesIO(buffer).getvalue()
		else:
			if img.max() < 2:
				img *= 255
			cv2.imwrite(file_name, img)
			return os.path.exists(file_name)
def resize_worker(img_file, sizes, resample):
	img = read_img(img_file[1])
	imgs= []
	if img is None:
		sizes = []
	for size in sizes:
		imgs.append(save_img(resize_img(img, size, resample)))
	return img_file[0], imgs
def prepare(env, dataset, n_worker, sizes = (128, 256, 512, 1024), resample = 'lanczos'):
	resize_fn = partial(resize_worker, sizes = sizes, resample = resample)
	files = sorted(dataset.imgs, key = lambda x: x[0])
	files = [(i, file) for i, (file, label) in enumerate(files)]
	total = 0
	bits = max(int(math.floor(math.log(max(len(files),1))/math.log(10)))+1, 5)
	fmt = '%%d-%%0%dd' % bits
	try:
		import multiprocessing
		with multiprocessing.Pool(n_worker) as pool:
			for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
				if len(imgs) == len(sizes):
					for size, img in zip(sizes, imgs):
						key = (fmt % (size,i)).encode('utf-8')
					with env.begin(write = True) as txn:
						txn.put(key, img)
					total += 1
			with env.begin(write = True) as txn:
				txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))
	except ModuleNotFoundError as e:
		for f in tqdm(files):
			i, imgs = resize_fn(f)
			if len(imgs) == len(sizes):
				for size, img in zip(sizes, imgs):
					key = (fmt % (size,i)).encode('utf-8')
				with env.begin(write = True) as txn:
					txn.put(key, img)
				total += 1
		with env.begin(write = True) as txn:
			txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'Preprocess images for model training')
	parser.add_argument('--out', type = str, help = 'filename of the result lmdb dataset')
	parser.add_argument('--size', type = str, default = '128,256,512,1024', \
		help = 'resolutions of images for the dataset')
	parser.add_argument('--n_worker', type = int, default = 8, \
		help = 'number of workers for preparing dataset')
	parser.add_argument('--resample', type = str, default = 'lanczos', \
		help = 'resampling methods for resizing images')
	parser.add_argument('path', type = str, help = 'path to the image dataset')
	args = parser.parse_args()
	sizes = []
	for s in args.size.split(','):
		try:
			size = int(s.strip())
			sizes += [size]
		except ValueError as s:
			pass
	print('Make dataset of image sizes:' + ','.join('%d'%s for s in sizes))
	try:
		from torchvision import datasets
		imgset = datasets.ImageFolder(args.path)
	except:
		from dataset import ImgDataset
		imgset = ImgDataset(args.path)
	with lmdb.open(args.out, map_size = 1024 ** 4, readahead = False) as env:
		prepare(env, imgset, args.n_worker, sizes = sizes, resample = args.resample)
