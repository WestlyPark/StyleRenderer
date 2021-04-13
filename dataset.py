import os
import sys
import lmdb
import numpy as np
from torch.utils.data import Dataset
from prepare_data import decode_img
from torchvision import transforms
class ImgDataset(Dataset):
	def __init__(self, folder, transform = transforms.ToTensor(), \
	resolution = 256, recurrent = True, \
	exts = ['.jpg','.jpeg','.png','.ppm','.bmp','.pgm','.tif','.tiff','.webp']):
		super(ImgDataset, self).__init__()
		if sys.version_info[0] == 3:
			from functools import reduce
		if isinstance(exts, str):
			exts = [exts]
		if os.path.isdir(folder):
			queue = [folder]
			files = []
		elif os.path.exists(folder):
			if reduce(lambda x,y: x or y, [ext.lower() == \
			folder[-len(ext):].lower() for ext in exts]):
				queue = []
				files = [folder]
		else:
			queue = files = []
		while len(queue) > 0:
			folder = queue[0]
			files += [os.path.join(folder,f) for f in os.listdir(folder) \
				if reduce(lambda x,y: x or y, \
				[ext.lower() == f[-len(ext):].lower() for ext in exts])]
			if recurrent:
				queue = queue[1:] + [ \
				os.path.join(folder,f) for f in os.listdir(folder) \
				if f!='.' and f!='..'and os.path.isdir( \
				os.path.join(folder,f))]
			else:
				queue = queue[1:]
		self.imgs = [(f, 0) for f in files]
		self.transform = transform
		self.resolution = resolution
	def __len__(self):
		return len(self.imgs)
	def __getitem__(self, index):
		from prepare_data import read_img, resize_img
		img = read_img(self.imgs[index][0])
		if isinstance(img, np.ndarray):
			h, w = img.shape[:2]
		else:
			h, w = img.size
		if h != self.resolution or w != self.resolution:
			img = resize_img(img, self.resolution)
		if self.transform is not None:
			img = self.transform(img)
		return img
class MultiResolutionDataset(Dataset):
	def __init__(self, path, transform = transforms.ToTensor(), resolution = 256):
		super(MultiResolutionDataset, self).__init__()
		self.env = lmdb.open(path, \
			max_readers = 32, \
			readonly = True, \
			lock = False, \
			readahead = False, \
			meminit = False)
		if not self.env:
			raise IOError('Cannot open lmdb dataset', path)
		with self.env.begin(write = False) as txn:
			self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
			keys = list(txn.cursor().iternext(values=False))
		bits = max(int(np.floor(np.log(max(self.length,1))/np.log(10)))+1, 5)
		res = []
		for key in keys:
			try:
				r = int(key.decode('utf-8').split('-')[0])
				res.append(r)
			except ValueError as e:
				pass
		res = list(np.unique(res))
		if not resolution in res:
			raise KeyError('No specified resolution', res)
		self.fmt = '%d-%%0%dd' % (resolution, bits)
		self.transform = transform
	def __len__(self):
		return self.length
	def __getitem__(self, index):
		with self.env.begin(write = False) as txn:
			key = (self.fmt % index).encode('utf-8')
			img_bytes = txn.get(key)
			img = decode_img(img_bytes)
		if self.transform is not None:
			img = self.transform(img)
		return img
if __name__ == '__main__':
	dataset = None
	if len(sys.argv) > 1:
		if os.path.exists(os.path.join(sys.argv[1],'data.mdb')):
			dataset = MultiResolutionDataset(sys.argv[1])
			print('Image number: %d' % len(dataset))
			I = dataset[0]
			print('Image size: [%d x %d]' % I.shape[1:3])
		elif os.path.isdir(sys.argv[1]):
			dataset = ImgDataset(sys.argv[1])
	if dataset is not None:
		print('Image number: %d' % len(dataset))
		I = dataset[0]
		print('Image size: [%d x %d]' % I.shape[1:3])
