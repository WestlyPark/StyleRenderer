import numpy as np
import argparse
import cv2
import sys
import os
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
def draw_landmarks(img, lmks = [], disp = False, color = (0,255,0), circle_size = 3, max_size = 720):
	try:
		img = np.array(img)
		sz = img.shape
	except:
		return
	if len(sz) == 2:
		img = img.reshape((sz[0],sz[1],1))
		sz = img.shape
	if sz[2] == 1:
		img_ = np.tile(img.copy(), (1,1,3))
	elif sz[2] >= 3:
		img_ = img[:,:,:3].copy()
	if img_.max() > 2:
		img_[img_ < 0]  = 0
		img_[img_ > 255]= 255
		img_ = img_.astype('uint8')
	else:
		img_[img_ < 0] = 0
		img_[img_ > 1] = 1
		img_ = (img_ * 255).astype('uint8')
	lmks = np.array(lmks)
	if len(lmks.shape) != 2:
		lmks = lmks.reshape(-1, 2)
	elif lmks.shape[0] == 2 and lmks.shape[1] != 2:
		lmks = lmks.T
	if len(lmks) > 0 and lmks[:,:2].max() < 1:
		lmks[:,0] *= sz[1]
		lmks[:,1] *= sz[0]
	rt = 1
	if max_size > 0 and (sz[0] > max_size or sz[1] > max_size):
		rt = max_size / float(max(sz[:2]))
		img_ = cv2.resize(img_, (int(rt*sz[1]), int(rt*sz[0])))
	for lmk in lmks:
		cv2.circle(img_, (int(lmk[0]*rt),int(lmk[1]*rt)),circle_size,color,-1)
	if disp:
		try:
			cv2.imshow('Landmarks', img_)
			cv2.waitKey()
		except:
			import matplotlib.pyplot as plt
			plt.imshow(img_[:,:,::-1])
			plt.show()
	return img_
class LandmarksReader:
	def __init__(self, file_name = os.path.join(BASE_DIR, 'tmp.txt')):
		with open(file_name, 'r') as f:
			lines = f.read().splitlines()
		data = [[float(f) for f in line.split(' ') \
			if f[-1] >= '0' and f[-1] <= '9'] for line in lines if len(line) > 0]
		names = [[img for img in line.split(' ') \
			if len(img) > 4 and img[-4:].lower() in ['.png','.jpg','.bmp']] \
			for line in lines if len(line) > 0]
		self.data = np.array([data[i] for i in range(len(names)) if len(names[i]) > 0])
		self.names = [name[0] for name in names if len(name) > 0]
		order = np.argsort(self.names)
		self.names = [self.names[i] for i in order]
		self.data = self.data[order,:]
	def detect(self, img_name):
		if len(self.data) == 0:
			return None
		for i in range(len(self.names)):
			name = self.names[i]
			if len(img_name) >= len(name) and img_name[-len(name):] == name:
				return self.data[i,:].reshape(-1,2)
		return None
class LandmarksDetectorExec:
	def __init__(self, model_name = 'track_images')):
		self.exec_name = model_name
	def detect(self, img):
		if isinstance(img, str):
			img_names = [os.path.basename(img)]
			argv = os.path.abspath(img)
		elif hasattr(img, '__len__') and len(img) > 0 and isinstance(img[0],str):
			img_names = [os.path.basename(img[i]) for i in range(len(img))]
			argv = os.path.abspath(os.path.dirname(img[0]))
		else:
			argv = os.path.join(BASE_DIR, 'tmp.png')
			cv2.imwrite(argv, img)
			img_names = ['tmp.png']
		os.system('cd %s; ./%s %s/tmp.txt %s' % (os.path.dirname(self.exec_name), \
			os.path.basename(self.exec_name), BASE_DIR, argv))
		model= LandmarksReader(os.path.join(BASE_DIR, 'tmp.txt'))
		lmks = [model.detect(img_name) for img_name in img_names]
		os.system('rm %s/tmp\.*' % BASE_DIR)
		return lmks[0] if len(lmks) == 1 else np.array(lmks)
class LandmarksDetectorDlib:
	def __init__(self, model_name = os.path.join(BASE_DIR, 'thirdparty', \
			'face_normals','data','shape_predictor_68_face_landmarks.dat')):
		import dlib
		self.detector = dlib.get_frontal_face_detector()
		self.predictor= dlib.shape_predictor(model_name)
		self.max_size = 640
		self.rt = 1
	def preprocess(self, img):
		if img.max() < 2:
			img = (img * 255).astype('uint8')
		elif img.dtype == np.float32 or img.dtype == np.float64:
			img[img < 0]  = 0
			img[img > 255]= 255
			img = img.astype('uint8')
		if len(img.shape) > 2 and img.shape[2] == 3:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			gray = img.reshape(img.shape[0], img.shape[1])
		if gray.shape[1] > self.max_size or gray.shape[0] > self.max_size:
			self.rt = float(self.max_size) / max(gray.shape)
			w = int(self.rt * gray.shape[1])
			h = int(self.rt * gray.shape[0])
			gray = cv2.resize(gray, (w, h), interpolation = cv2.INTER_AREA)
		return gray
	def detect(self, img):
		self.rt = 1
		img = self.preprocess(img)
		rects = self.detector(img, 1)
		lmks = []
		for (i, rect) in enumerate(rects):
			shape = self.predictor(img, rect)
			lmks += [np.array([[shape.part(i).x, shape.part(i).y] \
				for i in range(shape.num_parts)])]
		if len(lmks) > 0:
			return lmks[0] / self.rt
		else:
			return None
class LandmarksDetectorPytorch:
	def __init__(self, model = 'PLFD', checkpoint = os.path.join(BASE_DIR, 'thirdparty', \
			'pytorch_face_landmark','checkpoint','pfld_model_best.pth.tar'), \
			detector = 'MTCNN'):
		import torch
		sys.path.append(os.path.join(BASE_DIR,'thirdparty','pytorch_face_landmark'))
		if model == 'MobileNet':
			from models.basenet import MobileNet_GDConv
			self.model = MobileNet_GDConv(136)
			self.model = torch.nn.DataParallel(self.model)
			self.model.load_state_dict(torch.load(checkpoint)['state_dict'])
			self.model.eval()
			self.size = 224
		elif model == 'MobileFaceNet':
			from models.mobilefacenet import MobileFaceNet
			self.model = MobileFaceNet([112, 112], 136)
			self.model.load_state_dict(torch.load(checkpoint)['state_dict'])
			self.model.eval()
			self.size = 112
		elif model == 'PLFD':
			from models.pfld_compressed import PFLDInference
			self.model = PFLDInference()
			self.model.load_state_dict(torch.load(checkpoint)['state_dict'])
			self.model.eval()
			self.size = 112
		if detector == 'MTCNN':
			from MTCNN import detect_faces
			self.detect_fun = lambda x: detect_faces(x[:,:,::-1])
		elif detector == 'FaceBoxes':
			from FaceBoxes import FaceBoxes
			self.detector = FaceBoxes()
			self.detect_fun = lambda x: self.detector.face_boxex(x)
		elif detector == 'Retinaface':
			from Retinaface import Retinaface
			self.detector = Retinaface.Retinaface()
			self.detect_fun = lambda x: self.detector(x)
		else:
			import dlib
			self.detector = dlib.get_frontal_face_detector()
			self.detect_fun = lambda x: self.detector(cv2.cvtColor(x,cv2.COLOR_BGR2GRAY))
	def preprocess(self, img):
		import torch
		if img.max() < 2:
			img = (img * 255).astype(np.uint8)
		elif img.dtype == np.float32 or img.dtype == np.float64:
			img[img < 0]  = 0
			img[img > 255]= 255
			img = img.astype(np.uint8)
		rects = self.detect_fun(img)
		if isinstance(rects, tuple):
			rects = rects[0] # MTCNN
			if len(rects) == 0: return []
			x1,y1,x2,y2 = rects[0,:4]
		else:
			if len(rects) == 0: return []
			rect = rects[0]
			try:
				x1,y1,x2,y2 = rect[:4]
			except:
				x1 = rect.left()
				y1 = rect.top()
				x2 = rect.right()
				y2 = rect.bottom()
		w = x2 - x1 + 1
		h = y2 - y1 + 1
		size = int(min([w, h])*1.2)
		cx = x1 + w//2
		cy = y1 + h//2
		x1 = cx - size//2
		x2 = x1 + size
		y1 = cy - size//2
		y2 = y1 + size
		dx = max(0, -x1)
		dy = max(0, -y1)
		x1 = max(0, x1)
		y1 = max(0, y1)
		edx = max(0, x2 - img.shape[1])
		edy = max(0, y2 - img.shape[0])
		x2 = min(img.shape[1], x2)
		y2 = min(img.shape[0], y2)
		self.bbox = [int(x1), int(x2), int(y1), int(y2)]
		cropped = img[self.bbox[2]:self.bbox[3], self.bbox[0]:self.bbox[1]]
		if dx > 0 or dy > 0 or edx > 0 or edy > 0:
			cropped = cv2.copyMakeBorder(cropped,int(dy),int(edy),int(dx),int(edx), \
				cv2.BORDER_CONSTANT, 0)
		cropped_face = cv2.resize(cropped, (self.size, self.size))
		input_ = np.expand_dims(cropped_face.astype(np.float32).transpose([2,0,1]), 0) / 255.
		return [torch.from_numpy(input_).float()]
	def detect(self, img):
		img = self.preprocess(img)
		if len(img) == 0: return None
		lmks = []
		for i in img:
			lmk = self.model(i)[0].cpu().data.numpy()
			lmk = lmk.reshape(-1, 2)
			lmk[:,0] = lmk[:,0] * (self.bbox[1]-self.bbox[0]) + self.bbox[0]
			lmk[:,1] = lmk[:,1] * (self.bbox[3]-self.bbox[2]) + self.bbox[2]
			lmks += [lmk]
		return lmks[0]
class SkinSegmentationGrabcut:
	def __init__(self, lmks_num = 68, refine = None):
		if os.path.exists(os.path.join(BASE_DIR,'models','lmk%d.obj'%lmks_num)):
			import re
			with open(os.path.join(BASE_DIR,'models','lmk%d.obj'%lmks_num), 'r') as f:
				tri = re.findall('f'+' +([0-9]*)'*3, f.read())
				tri = np.array([[int(f) for f in fi] for fi in tri], 'uint32')-1
			self.tri = tri
		else:
			self.tri = None
		if not refine is None:
			if not hasattr(refine, '__len__'):
				refine = [refine]
			if hasattr(refine, '__len__'):
				self.ksize = [ \
					refine[0] if len(refine) > 0 else 0,\
					refine[1] if len(refine) > 1 else ( \
					refine[0] if len(refine) > 0 else 0)]
		else:
			self.ksize = None
	def segment(self, img, lmks):
		mask = np.zeros_like(img[:,:,0]).astype('uint8')
		if self.tri is not None:
			for f in self.tri:
				lmk = lmks[f,:].astype('int32')
				cv2.fillPoly(mask, np.expand_dims(lmk,0), 1)
		else:
			hull = cv2.convexHull(lmks.astype(np.int32))
			cv2.fillPoly(mask, np.expand_dims(hull.reshape(-1,2),0), 1)
		if not self.ksize is None:
			if img.max() < 2:
				img = img * 255
			if len(img.shape) < 3:
				img = np.expand_dims(img, -1)
			img[img < 0] = 0; img[img > 255] = 255
			img = img.astype(np.uint8)
			if img.shape[-1] == 1:
				img = np.tile(img, (1,1,3))
			maskf =	cv2.erode(mask, \
				cv2.getStructuringElement(cv2.MORPH_RECT, ( \
					int(self.ksize[0]*img.shape[1]), \
					int(self.ksize[0]*img.shape[0]))), \
				iterations = 1, borderType = cv2.BORDER_CONSTANT)
			maskb =	cv2.dilate(mask, \
				cv2.getStructuringElement(cv2.MORPH_RECT, ( \
					int(self.ksize[1]*img.shape[1]), \
					int(self.ksize[1]*img.shape[0]))), \
				iterations = 1, borderType = cv2.BORDER_CONSTANT)
			maski =	cv2.GC_BGD *(1-maskb) + \
				cv2.GC_FGD * maskf + \
				cv2.GC_PR_BGD *(maskb - mask) + \
				cv2.GC_PR_BGD *(mask - maskf)
			maski =	cv2.grabCut(img, maski, (0, 0, img.shape[1], img.shape[0]), \
				np.zeros((1,65), 'float'), \
				np.zeros((1,65), 'float'), \
				iterCount = 5, mode = cv2.GC_INIT_WITH_MASK)[0]
			mask = (maski == cv2.GC_FGD) + (maski == cv2.GC_PR_FGD)
		return mask > 0
class SkinSegmentationPytorch:
	def __init__(self, model = 'FCNResNet101', checkpoint = os.path.join(BASE_DIR, 'thirdparty', \
			'SemanticSegmentation','pretrained', 'model_segmentation_skin_30.pth'), \
			threshold = .5):
		import torch
		from torchvision import transforms
		sys.path.append(os.path.join(BASE_DIR,'thirdparty','SemanticSegmentation'))
		state_dict = torch.load(checkpoint)
		if model == 'FCNResNet101':
			from semantic_segmentation.models.fcn import FCNResNet101
			category_prefix = '_categories.'
			categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
			categories = [k[len(category_prefix):] for k in categories]
			self.model = FCNResNet101(categories)
		elif model == 'BiSeNetV2':
			category_prefix = '_categories.'
			categories = [k for k in state_dict.keys() if k.startswith(category_prefix)]
			categories = [k[len(category_prefix):] for k in categories]
			from semantic_segmentation.models.bisenetv2 import BiSeNetV2
			self.model = BiSeNetV2(categories)
		self.model.load_state_dict(state_dict)
		self.model.eval()
		self.transform = transforms.Normalize( \
			mean = (.485, .456, .406), \
			std  = (.229, .224, .225))
		self.th = (threshold if threshold < 1 else 1) if threshold > 0 else 0
	def segment(self, img, *kwargs):
		import torch
		if img.max() > 2.:
			img = img.astype(np.float32) / 255.
		sz = img.shape
		img = img[:(sz[0]//32)*32,:(sz[1]//32)*32]
		img = img.reshape(img.shape[0],img.shape[1],-1)
		if img.shape[-1] == 1:
			img = np.tile(img,[1,1,3])
		else:
			img = img[:,:,::-1]
		img = np.transpose(img,[2,0,1]).copy()
		input_ = self.transform(torch.from_numpy(img).float())
		with torch.no_grad():
			out = torch.sigmoid(self.model(input_.unsqueeze(0))['out'])
		mask = out[0,0].cpu().data.numpy()
		if sz[0] > mask.shape[0] or sz[1] > mask.shape[1]:
			mask = np.pad(mask, ((0,sz[0]-mask.shape[0]),(0,sz[1]-mask.shape[1])), \
				'constant', constant_values = (0,0))
		return mask > self.th
class RecognitionFeature:
	def __init__(self, model_type = 'vggface2'):
		sys.path.append(os.path.join(BASE_DIR,'thirdparty','facenet-pytorch','models'))
		from inception_resnet_v1 import InceptionResnetV1
		from mtcnn import MTCNN
		self.model = InceptionResnetV1(pretrained = model_type)
		self.model = self.model.eval()
		self.detector = MTCNN(image_size = 160, margin = 0, min_face_size = 20, \
			thresholds = [0.6, 0.7, 0.7], factor = 0.709, post_process = True)
		self.size = 160
		self.color_range = [-1, 1]
	def detect(self, img):
		x_aligned, prob = self.detector(img, return_prob = True)
		if x_aligned is None: return None
		embeddings = self.model(x_aligned.unsqueeze(0))
		embeddings = embeddings[0].detach().cpu().numpy()
		return embeddings
def solve_ortho(src, dst, max_iter = 0, eps = 1e-9):
	n = src.shape[0]
	src_mean = src.mean(0).reshape(1,-1)
	dst_mean = dst.mean(0).reshape(1,-1)
	src_ = src - np.tile(src_mean, (n, 1))
	dst_ = dst - np.tile(dst_mean, (n, 1))
	u, w, vt = np.linalg.svd(src_)
	w_inv = [1./w[i] if w[i] > eps else w[i] for i in range(len(w))]
	R = vt.T.dot(np.diag(w_inv)).dot(u[:,0:vt.shape[0]].T).dot(dst_)
	u, w, vt = np.linalg.svd(R)
	vt_ = np.eye(3).astype(u.dtype)
	vt_[:2,:2] = vt
	if np.linalg.det(vt_)*np.linalg.det(u) < 0:
		vt_[2,2] = -1
	R_ = u.dot(vt_)
	w = ((R * R_[:,:2]).sum() / (R_[:,:2]*R_[:,:2]).sum())
	T = np.concatenate([w * R_, np.concatenate([ \
		dst_mean - src_mean.dot(w*R_[:,:2]), [[0]]],1)], 0).T
	T[2,3] = 1./ np.maximum(w, eps)
	if max_iter > 0:
		from scipy.optimize import leastsq
		def fun(x, src, dst):
			R = cv2.Rodrigues(x[:3])[0]
			src_ = x[3] * src.dot(R[:,:2]) + x[4:6].reshape(1,-1)
			return (src_ - dst).reshape(-1)
		def jac(x, src, dst):
			R, dR = cv2.Rodrigues(x[:3])[0]
			J = np.zeros([len(src)*2, len(x)], x.dtype)
			J[:,0]  = x[3] * src.dot(dR[0,:].reshape(3,3)[:,:2])
			J[:,1]  = x[3] * src.dot(dR[1,:].reshape(3,3)[:,:2])
			J[:,2]  = x[3] * src.dot(dR[2,:].reshape(3,3)[:,:2])
			J[:,3]  = src.dot(R[:,:2])
			J[:,4:6]= np.tile(np.eye(2),(len(src),1))
			return J
		x0 = cv2.Rodrigues(R_)[0].reshape(-1)
		x0 = np.concatenate([x0,[w],T[:2,3]])
		res = leastsq(fun, x0, args = (src, dst), \
			Dfun = jac, ftol = eps, maxfev = int(max_iter))
		x = res[0]
		T[:3,:3]= x[3] * cv2.Rodrigues(x[:3])[0].T
		T[:2,3] = x[4:6]
	return T
def solve_affine(src, dst, max_iter = 0, eps = 1e-9):
	J = np.zeros([len(src)*2, 4], src.dtype)
	J[:,0]  = src[:,:2].reshape(-1)
	J[:,1]  = np.concatenate((-src[:,1:2],src[:,:1]),1).reshape(-1)
	J[:,2:4]= np.tile(np.eye(2),(len(src),1))
	u, w, vt = np.linalg.svd(J)
	w_inv = [1./w[i] if w[i] > eps else w[i] for i in range(len(w))]
	x0 = dst.reshape(-1).dot(u[:,:4]).dot(np.diag(w_inv)).dot(vt)
	T = np.array([ \
		[x0[0],-x0[1],x0[2]],\
		[x0[1], x0[0],x0[3]]], x0.dtype)
	if max_iter > 0:
		from scipy.optimize import leastsq
		def fun(x, src, dst):
			R = np.array([[x[0],x[1]],[-x[1],x[0]]], x.dtype)
			src_ = src.dot(R) + x[2:4].reshape(1,-1)
			return (src_ - dst).reshape(-1)
		def jac(x, src, dst):
			return J
		res = leastsq(fun, x0, args = (src, dst), \
			Dfun = jac, ftol = eps, maxfev = int(max_iter))
		x = res[0]
		T[0,0] = T[1,1] = x0[0]
		T[1,0] = x0[1]; T[0,1] = -x0[1]
		T[:2,:] = x0[2:4]
	return T
def euler_mat_inv(R, _type = 'yxz', eps = 1e-9):
	tp = [ord(t)-ord('x') for t in _type.lower()]
	permute = 2 * ((tp[0] - tp[1]) % 3) - 3
	if tp[0] == tp[2] and tp[0] != tp[1]: # zxz type
		i = tp[0]; j = tp[1]; k = 3-tp[0]-tp[1]
		D = max(min(R[i,i],1),-1)
		r = np.array([ \
			np.arctan2(R[i,j], permute * R[i,k]), \
			np.arccos(D), \
			np.arctan2(R[j,i],-permute * R[k,i])], R.dtype)
		if 1 - D <= eps:
			r[2] = np.arctan2(-permute*R[j,k],R[j,j]) - r[0]
		elif 1 + D <= eps:
			r[2] = np.arctan2(permute*R[j,k], R[j,j]) + r[0]
		return r
	elif len(set(tp).difference([0,1,2])) == 0: # zyx type
		i = tp[0]; j = tp[1]; k = tp[2]
		D = max(min(R[k,i],1),-1)
		r = np.array([ \
			np.arctan2(permute * R[k,j], R[k,k]), \
			np.arcsin(-permute * D), \
			np.arctan2(permute * R[j,i], R[i,i])])
		if 1 - D <= eps:
			r[2] = np.arctan2(-permute* R[j,k], R[j,j]) - r[0]
		elif 1 + D <= eps:
			r[2] = np.arctan2(permute * R[j,k], R[j,j]) + r[0]
		return r
	else:
		return np.zeros(3, R.dtype)
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Preprocess for Faces')
	parser.add_argument('path', type = str, help = 'path to image/images folder')
	parser.add_argument('--lmk', default = 'Exec', \
		help =	'Landmarks Detection Method')
	parser.add_argument('--bfm', default = '/data/BaselFaceModel.mat', \
		help =	'Morphable Face Model')
	parser.add_argument('--mask', default = '', \
		help =	'Skin Segmentation Model')
	parser.add_argument('--disp', action = 'store_true', \
		help =	'Show Landmarks')
	parser.add_argument('--output', default = '', \
		help =	'Output folder for processed images')
	args = parser.parse_args()
	from dataset import ImgDataset
	data = ImgDataset(args.path)
	if 'exe' in args.lmk.lower():
		detector = LandmarksDetectorExec()
		base_lmk = [0]*68
	elif 'dlib' in args.lmk.lower():
		detector = LandmarksDetectorDlib()
		base_lmk = [0]*68
	elif 'torch' in args.lmk.lower():
		detector = LandmarksDetectorPytorch()
		base_lmk = [0]*68
	elif os.path.exists(args.lmk) and args.lmk[-4:].lower() == '.txt':
		detector = LandmarksReader(args.lmk)
		base_lmk = [0]*(args.lmk.shape[1]//2)
	if 'torch' in args.mask.lower():
		mask = SkinSegmentationPytorch()
	else:
		mask = SkinSegmentationGrabcut(refine = [.01, .5])
	if args.output != '':
		try:
			if not os.path.isdir(args.output):
				os.mkdir(args.output)
			import scipy.io as sio
			import torch
			model = sio.loadmat(args.bfm)
			v =(model['v'] - model['v'].mean(1).reshape(-1,1)).T * 1e-5
			c = model['tex'].T
			f = model['tri'][0,0].astype(np.int64)
			if 'landmarks%d' % len(base_lmk) in model.keys():
				base_lmk = v[model['landmarks%d' % len(base_lmk)] \
					.reshape(-1).astype(np.uint64)-f.min(),:]
				mesh = None
			else:
				f = f - f.min()
				v = torch.from_numpy(np.expand_dims(v.astype(np.float32), 0))
				c = torch.from_numpy(np.expand_dims(c.astype(np.float32), 0))
				f = torch.from_numpy(f)
				base_lmk = []
				mesh = (v, c, f)
		except ModuleNotFoundError as e:
			mesh = None; args.output = ''
			base_lmk = []
		base_img = None
	for img_name in data.imgs:
		if not isinstance(detector, LandmarksReader):
			img = cv2.imread(img_name[0])
			if img is None: continue
		else:
			img = img_name[0]
		lmk = detector.detect(img)
		if len(lmk) <= 0: continue
		if args.output != '':
			if mesh is not None:
				from op import rasterize
				base_img = rasterize(mesh[0], mesh[1], mesh[2], \
					img.shape[0], img.shape[1]).numpy()[0,:,:,::-1]
				base_lmk = detector.detect(base_img)
				mesh = None
			elif base_img is None:
				base_img = img
				if len(base_lmk) > 0 and len(base_lmk[0]) == 3:
					base_lmk[:,0] = (1 + base_lmk[:,0]) * img.shape[1]/2
					base_lmk[:,1] = (1 - base_lmk[:,1]) * img.shape[0]/2
					base_lmk[:,2] = -base_lmk[:,2] * \
						(img.shape[1] + img.shape[0]) / 4
			if len(base_lmk) == len(lmk):
				if len(base_lmk[0]) == 3:
					T = solve_ortho(base_lmk, lmk)
					f = np.cbrt(np.linalg.det(T[:3,:3]))
					rot = euler_mat_inv(T[:3,:3]/f, 'yxz')
					c, s = f*np.cos(rot[2]), f*np.sin(rot[2])
					tx,ty= T[:2,3]
					T = np.array([[c,-s,tx],[s,c,ty],[0,0,1]],T.dtype)
				else:
					T = solve_affine(base_lmk[:,:2], lmk)
				if T.shape[0] == 2:
					T = np.concatenate((T,[[0,0,1]]), 0)
				Tinv = np.linalg.inv(T)[:2,:]
				img_ = cv2.warpAffine(img, Tinv, \
					(base_img.shape[1],base_img.shape[0]), \
					flags = cv2.INTER_LINEAR, \
					borderMode = cv2.BORDER_REFLECT)
				cv2.imwrite(os.path.join(args.output, \
					os.path.basename(img_name[0])), img_)
		if args.disp:
			alpha = .3
			msk = mask.segment(img, lmk)
			img =(1-alpha)*img + msk.reshape(msk.shape[0],msk.shape[1],1)*255*alpha
			draw_landmarks(img, lmk, True)

