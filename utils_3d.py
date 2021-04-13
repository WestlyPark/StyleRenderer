import numpy as np
import torch
import sys
import os
from layers import Normalize
def normalize(vec, axis = -1, _type = 'L2', eps = 1e-8):
	return Normalize.apply(vec, axis, _type, eps)
def translate_mat(t):
	if len(t.shape) == 1:
		batch = 1
		t = t.view(1,-1)
		reshape = True
	else:
		batch = int(t.shape[0])
		t = t.view(batch,-1)
		reshape = False
	n = int(t.shape[1])
	mat = torch.cat([
		torch.zeros([batch,n*(n+1)],dtype=t.dtype,device=t.device), \
		t, torch.ones_like(t[:,:1])], dim = 1)
	if reshape:
		return mat.view(n+1,n+1).permute(1,0)
	else:
		return mat.view(-1,n+1,n+1).permute(0,2,1)
def rotate_mat(theta): # rotate in z axis for image
	sz = [int(i) for i in theta.shape]
	c = torch.cos(theta).view(-1,1)
	s = torch.sin(theta).view(-1,1)
	i = torch.ones(len(c),1,dtype = c.dtype,device = c.device)
	o = torch.zeros(len(c),1,dtype = c.dtype,device = c.device)
	mat = torch.cat([c, -s, o, s, c, o, o, o, i], 1)
	return mat.view(sz+[3,3])
def scale_mat(s, dim = None):
	sz = [int(i) for i in s.shape]
	if dim is None or int(dim) <= 0:
		dim = sz[-1]
		s = s.view(-1,dim)
	else:
		dim = int(dim)
		s = s.view(-1,1).expand(-1,dim)
	return torch.diag_embed(torch.cat([s, \
		torch.ones_like(s[:,:1])], 1))
def euler_mat(angle, _type = 'yxz'):
	if len(angle.shape) == 1:
		batch = 1; reshape = True
		angle = angle.view(1,-1)
	else:
		reshape = False
		batch = angle.shape[0]
	c = torch.cos(angle)
	s = torch.sin(angle)
	T = None
	one = torch.ones(len(c),1,dtype = c.dtype,device = c.device)
	zero= torch.zeros(len(c),1,dtype = c.dtype,device = c.device)
	for i in range(3):
		if _type[i].lower() == 'x':
			R = torch.cat(( \
				 one,      zero,       zero, \
				zero,c[:,i:i+1],-s[:,i:i+1], \
				zero,s[:,i:i+1], c[:,i:i+1]),-1).view(-1,3,3)
		elif _type[i].lower() == 'y':
			R = torch.cat(( \
				 c[:,i:i+1],zero,s[:,i:i+1], \
				       zero, one,      zero, \
				-s[:,i:i+1],zero,c[:,i:i+1]),-1).view(-1,3,3)
		elif _type[i].lower() == 'z':
			R = torch.cat(( \
				c[:,i:i+1],-s[:,i:i+1],zero, \
				s[:,i:i+1], c[:,i:i+1],zero, \
				      zero,       zero, one),-1).view(-1,3,3)
		else:
			continue
		if T is None:
			T = R
		else:
			T = torch.matmul(R, T)
	if reshape:
		return T.view(3,3)
	else:
		return T
class Rodrigues(torch.autograd.Function):
	@staticmethod
	def forward(ctx, rvec, eps):
		ctx.eps = eps = abs(eps)
		if len(rvec.shape) == 1:
			batch = 1; reshape = True
			rvec = rvec.view(1,-1)
		else:
			reshape = False
			batch = rvec.shape[0]
		dim = 3
		rr = torch.matmul(rvec.view(-1,dim,1), rvec.view(-1,1,dim))
		r2 = rr[:,0,0] + rr[:,1,1] + rr[:,2,2]
		r = torch.sqrt(r2)
		c = torch.cos(r)
		s = torch.sin(r)
		sc = torch.where(r <= eps, 1 - r2/6, s/r)
		cc = torch.where(r <= eps,.5 - r2/24,(1-c)/r2)
		zero = torch.zeros(batch, 1, dtype = rvec.dtype, device = rvec.device)
		I = torch.eye(dim, device = rvec.device).type(rvec.dtype) \
			.view(1,dim,dim).expand(batch,-1,-1)
		rx = torch.cat(( \
			 zero,       -rvec[:,2:3], rvec[:,1:2], \
			 rvec[:,2:3], zero,       -rvec[:,0:1], \
			-rvec[:,1:2], rvec[:,0:1], zero), 1).view(-1,dim,dim)
		ctx.save_for_backward(c, sc, cc, r, rr, rx)
		if reshape:
			return	c.view(1,1).expand(dim,dim) * I.squeeze()  + \
				cc.view(1,1).expand(dim,dim)* rr.squeeze() + \
				sc.view(1,1).expand(dim,dim)* rx.squeeze()
		else:
			return	c.view(-1,1,1).expand(-1,dim,dim) * I  + \
				cc.view(-1,1,1).expand(-1,dim,dim)* rr + \
				sc.view(-1,1,1).expand(-1,dim,dim)* rx
	@staticmethod
	def backward(ctx, grad_out):
		if ctx.needs_input_grad[0]:
			reshape =  len(grad_out.shape) <= 2
			c, sc, cc, r, rr, rx = ctx.saved_tensors
			eps = ctx.eps
			r2 = rr[:,0,0] + rr[:,1,1] + rr[:,2,2]
			batch = int(c.shape[0])
			dim   = 3
			dcc = torch.where(r <= eps,-1./12+ r2/180,(sc-2*cc)/r2)
			dsc = torch.where(r <= eps,-1./3 + r2/30, (c -  sc)/r2)
			I = torch.eye(dim, device = grad_out.device).type(grad_out.dtype) \
				.view(1,dim,dim).expand(batch,-1,-1)
			dRdr =	dcc.view(-1,1,1).expand(-1,dim,dim)* rr + \
				dsc.view(-1,1,1).expand(-1,dim,dim)* rx - \
				sc.view(-1,1,1).expand(-1,dim,dim) * I
			grad_out = grad_out.view(-1,dim,dim)
			dr = (grad_out * dRdr).sum([1,2])
			rc = torch.cat([ \
				cc.view(-1,1) * rx[:,2:3,1], \
				cc.view(-1,1) * rx[:,0:1,2], \
				cc.view(-1,1) * rx[:,1:2,0]], dim = 1)
			grad_in = torch.cat([ \
				(dr * rx[:,2,1] + \
				(rc *(grad_out[:,0,:] + grad_out[:,:,0])).sum(1) + \
				 sc *(grad_out[:,2,1] - grad_out[:,1,2])).view(-1,1), \

				(dr * rx[:,0,2] + \
				(rc *(grad_out[:,1,:] + grad_out[:,:,1])).sum(1) + \
				 sc *(grad_out[:,0,2] - grad_out[:,2,0])).view(-1,1), \

				(dr * rx[:,1,0] + \
				(rc *(grad_out[:,2,:] + grad_out[:,:,2])).sum(1) + \
				 sc *(grad_out[:,1,0] - grad_out[:,0,1])).view(-1,1)], dim = 1)
			if reshape: grad_in = grad_in.squeeze()
		else:
			grad_in = None
		return grad_in,None
def rodrigues(rvec, eps = 1e-8):
	return Rodrigues.apply(rvec, eps)
def random_apply_color(p = [.2,.3,0,.15,.5], img = None):
	# p = [bright, contrast, luma_flip, hue, saturation]
	batch = len(img) if img is not None and len(img.shape) >= 4 else 1
	if not isinstance(p, torch.Tensor):
		p = torch.Tensor(p)
	p = torch.abs(p.reshape(-1)[:5])
	if len(p) < 5:
		p = torch.cat((p, torch.zeros(5-len(p),dtype=p.dtype,device=p.device)))
	z = torch.cat([ \
		torch.normal(mean = 0, std = p[:2].unsqueeze(0).expand(batch,-1)),\
		torch.rand(batch,1,dtype=p.dtype,device=p.device), \
		torch.normal(mean = 0, std = p[3:].unsqueeze(0).expand(batch,-1))], 1)
	br  = z[:,0:1]
	con = torch.exp(z[:,1:2])
	luma=(z[:,2:3] < p[2])
	hue = z[:,3:4]
	sat = torch.exp(z[:,4:5]).unsqueeze(-1)
	o = torch.zeros_like(br)
	i = torch.ones_like(br)
	eye = torch.eye(3,dtype=z.dtype,device=z.device).unsqueeze(0)
	C = torch.cat([con,o,o,con*br,o,con,o,con*br,o,o,con,con*br], 1).view(-1,3,4)
	C = torch.matmul(eye - luma.view(-1,1,1).type(z.dtype)*2./3, C)
	C = torch.matmul(rodrigues(torch.cat([hue,hue,hue],1)/np.sqrt(3)), C)
	C = torch.matmul(eye*sat + torch.ones_like(eye)*(1-sat)/3., C)
	if img is None:
		return C[0]
	else:
		sz = [int(i) for i in img.shape]
		while len(img.shape) <= 3:
			img = img.unsqueeze(0)
		img = img.view(batch,-1,sz[-1]*sz[-2])
		C = C.type(img.dtype).to(img.device)
		img = torch.matmul(C[:,:3,:3], img) + C[:,:3,3:4]
		return img.view(sz)
def random_apply_pose2D_img(p = [.1,.1,.05,.15,0,.5], \
		img = None, cam = None, output_size = [], pad = 'zeros'):
	# p = [tx, ty, r_z, s_var, s_mean, flip_r]
	batch = len(img) if img is not None and len(img.shape) >= 4 else 1
	if cam is not None:
		sz = [int(i) for i in cam.shape]
		if len(sz) <= 2:
			cam = cam.unsqueeze(0)
		elif img is None:
			batch = len(cam)
	if img is not None and len(img.shape) >= 2:
		hi = int(img.shape[-2])
		wi = int(img.shape[-1])
	elif cam is not None:
		hi = max(int(cam[:,1,2].max().detach().cpu().numpy() * 2),0)
		wi = max(int(cam[:,0,2].max().detach().cpu().numpy() * 2),0)
	else:
		hi = wi = 0; img = None
	output_size = np.reshape(output_size,-1).astype(np.uint32)
	if len(output_size) == 0:
		ho = hi; wo = wi
	elif len(output_size) == 1:
		ho = wo = int(output_size[0])
	else:
		ho = int(output_size[0])
		wo = int(output_size[1])
	if not isinstance(p, torch.Tensor):
		p = torch.Tensor(p)
	p = torch.abs(p.reshape(-1)[:6])
	if len(p) < 6:
		p = torch.cat((p, torch.zeros(6-len(p),dtype=p.dtype,device=p.device)))
	z = torch.cat([ \
		torch.normal(mean = 0, std = p[:3].unsqueeze(0).expand(batch,-1)), \
		torch.normal(mean = p[4:5].unsqueeze(0).expand(batch,1), \
			std = p[3:4].unsqueeze(0).expand(batch,-1)), \
		torch.rand(batch,1,dtype=p.dtype,device=p.device)], 1)
	flip = (z[:,4:5] < p[-1])
	f = torch.exp(z[:,3:4])
	s = torch.sin(z[:,2:3])
	c = torch.cos(z[:,2:3])
	tx= z[:,0:1]
	ty= z[:,1:2]
	o = torch.zeros_like(f)
	i = torch.ones_like(f)
	if img is None:
		if cam is None:
			c *= f; s *= f
			T = torch.cat([c, -s, tx, s, c, ty, o, o, i], 1).view(-1,3,3)
			return T[0]
		else:
			if cam[:,:2,2].max().cpu() < .75 and wi > 0 and hi > 0:
				cam = cam * torch.tensor([[[wi],[hi],[1]]], \
					dtype = cam.dtype, device = cam.device)
				normalized = True
			else:
				normalized = False
			cam_out = cam * torch.cat( \
				[f,f,i,f,f,i,i,i,i], 0).view(-1,3,3) + torch.cat( \
				[o,o,tx*cam[:,0,0:1],o,o,-ty*cam[:,1,1:2],o,o,o], 0).view(-1,3,3)
			if normalized and ho > 0 and wo > 0:
				cam_out = cam_out / torch.tensor([[[wo],[ho],[1]]], \
					dtype = cam.dtype, device = cam.device)
				cam_out[:,0,2:3] = torch.where(flip, 1-cam_out[:,0,2:3], cam_out[:,0,2:3])
			else:
				cam_out[:,0,2:3] = torch.where(flip,wo-cam_out[:,0,2:3], cam_out[:,0,2:3])
			return cam_out.view(sz[:-2]+[3,3])
	elif img is not None:
		sz = [int(i) for i in img.shape]
		while len(img.shape) <= 3:
			img = img.unsqueeze(0)
		y, x = torch.meshgrid(torch.linspace(0,ho,ho),torch.linspace(0,wo,wo))
		x, y =	x.reshape(1,-1).type(z.dtype).to(z.device), \
			y.reshape(1,-1).type(z.dtype).to(z.device)
		corners = None
		try:
			add = float(pad)
			pad = 'zeros'
		except (TypeError,ValueError) as e:
			add = 0
			if pad is None: pad = ''
			if 'z' in pad.lower():
				pad = 'zeros'
			elif 'b' in pad.lower() or 'replicate' in pad.lower():
				pad = 'border'
			elif 'r' in pad.lower():
				pad = 'reflection'
			else:
				corners = 1; pad = 'zeros'
		if cam is not None:
			if cam[:,:2,2].max().cpu() < .75:
				cam = cam * torch.tensor([[[wo],[ho],[1]]], \
					dtype = cam.dtype, device = cam.device)
			cam = cam.type(z.dtype)
			cam_out = cam + torch.cat( \
				[o,o,tx*cam[:,0,0:1],o,o,-ty*cam[:,1,1:2],o,o,o], 0).view(-1,3,3)
			x = x.expand(batch,-1); x = torch.where(flip.expand(-1,ho*wo),wo-x,x)
			y = y.expand(batch,-1)
			if corners is not None:
				corners=(x[:,[0,wo-1,wo*(ho-1),ho*wo-1]], \
					 y[:,[0,wo-1,wo*(ho-1),ho*wo-1]])
				cam_inv=torch.inverse(cam_out)
				corners=cam_inv[:,0,0]*corners[0]+cam_inv[:,0,1]*corners[1]+cam_inv[:,0,2], \
					cam_inv[:,1,0]*corners[0]+cam_inv[:,1,1]*corners[1]+cam_inv[:,1,2]
				corners=c*corners[0]+s*corners[1], -s*corners[0]+c*corners[1]
				corners=cam[:,0,0]*corners[0] - cam[:,0,1]*corners[1], \
					cam[:,1,0]*corners[0] - cam[:,1,1]*corners[1]
				bd_min= torch.min(torch.cat([cam[:,0,2:3],wi-cam[:,0,2:3]],1),1)[0], \
					torch.min(torch.cat([cam[:,1,2:3],hi-cam[:,1,2:3]],1),1)[0]
				fmax = torch.max(torch.abs(torch.cat([ \
					corners[0]/bd_min[0], \
					corners[1]/bd_min[1]], 1)),1)[0]
				f = torch.where(f < fmax, fmax, f)
			cam_out *= torch.cat([f,f,i,f,f,i,i,i,i], 0).view(-1,3,3)
			cam_inv = torch.inverse(cam_out)
			z = cam_inv[:,2,0] * x + cam_inv[:,2,1] * y + cam_inv[:,2,2]
			x, y =	(cam_inv[:,0,0] * x + cam_inv[:,0,1] * y + cam_inv[:,0,2]) / z, \
				(cam_inv[:,1,0] * x + cam_inv[:,1,1] * y + cam_inv[:,1,2]) /-z
			cam_out[:,0,2:3] = torch.where(flip, wo-cam_out[:,0,2:3], cam_out[:,0,2:3])
		else:
			x = (x - (wo/2.)) / (max(wo,ho)/2.)
			y = ((ho/2.) - y) / (max(wo,ho)/2.)
			x = x.expand(batch,-1); x = torch.where(flip.expand(-1,ho*wo),-x,x)
			y = y.expand(batch,-1)
			x -= tx
			y -= ty
			if corners is not None:
				corners=(x[:,[0,wo-1,wo*(ho-1),ho*wo-1]], \
					 y[:,[0,wo-1,wo*(ho-1),ho*wo-1]])
				corners=((c*corners[0]+s*corners[1])*max(wo,ho)/float(wi), \
					(-s*corners[0]+c*corners[1])*max(wo,ho)/float(hi))
				fmax = torch.max(torch.abs(torch.cat(corners,1)),1)[0]
				f = torch.where(f < fmax, fmax, f)
			x /= f
			y /= f
		x, y = c*x+s*y, -s*x+c*y
		if cam is not None:
			z = cam[:,2,0] * x + cam[:,2,1] *-y + cam[:,2,2]
			x, y =	(cam[:,0,0] * x + cam[:,0,1] *-y + cam[:,0,2]) / z, \
				(cam[:,1,0] * x + cam[:,1,1] *-y + cam[:,1,2]) / z
			x = (x/(wi/2.)) - 1
			y = (y/(hi/2.)) - 1
			cam = cam_out
		else:
			x = x * max(wo,ho)/float(wi)
			y =-y * max(wo,ho)/float(hi)
		x = x.view(-1,ho,wo,1).type(img.dtype)
		y = y.view(-1,ho,wo,1).type(img.dtype)
		if img.is_cuda:
			x = x.to(img.device)
			y = y.to(img.device)
		img = torch.nn.functional.grid_sample(img, torch.cat([x,y],-1), \
			mode = 'bilinear', padding_mode = pad)
		if add != 0:
			img = img + add - torch.nn.functional.grid_sample( \
				add*torch.ones_like(img[:1,:1]),torch.cat([x,y],-1), \
				mode = 'bilinear', padding_mode = pad)
		img = img.view(sz[:-2] + [ho,wo])
		if cam is None:
			return img
		else:
			return img, cam
def augment(img, augment_ratio = .5):
	sz = [int(i) for i in img.shape]
	while len(img.shape) < 4:
		img = img.unsqueeze(0)
	img_ = random_apply_pose2D_img(img = img, pad = None)
	img_ = random_apply_color(img = img_)
	sz_ = [int(i) for i in img_.shape]
	p = torch.rand(sz_[0],1,1,1, \
		dtype = img.dtype, device = img.device)
	return torch.where(p.expand(-1,sz_[1],sz_[2],sz_[3]) < augment_ratio, img_, img)
def random_apply_pose3D(p = [.5,.1,.05,.1,.1,.1,.15], v = None):
	# p = [yaw, pitch, roll, tx, ty, tz, scale]
	batch = len(v) if v is not None and len(v.shape) >= 3 else 1
	if not isinstance(p, torch.Tensor):
		p = torch.Tensor(p)
	p = torch.abs(p.reshape(-1)[:7])
	if len(p) < 7:
		p = torch.cat((p, torch.zeros(7-len(p),dtype=p.dtype,device=p.device)))
	z = torch.normal(mean = 0, std = p.unsqueeze(0).expand(batch,-1))
	T = torch.cat(( \
		torch.exp(z[:,-1]).view(-1,1,1) * euler_mat(z[:,:3], 'yxz'), \
		z[:,3:6].view(-1,3,1)), -1)
	if v is not None:
		if v.is_cuda:
			T = T.to(v.device)
		return torch.matmul(v[...,:3].view(batch,-1,3), T[:,:3,:3]) + \
			T[:,:3,3:].view(-1,1,3)
	else:
		return T[0]
def mesh_point_normal(v, tri):
	va = torch.index_select(v, 1, tri[:,0])
	vb = torch.index_select(v, 1, tri[:,1])
	vc = torch.index_select(v, 1, tri[:,2])
	ab = vb - va
	ac = vc - va
	vn_= torch.cat(( \
		ab[:,:,1:2] * ac[:,:,2:3] - ab[:,:,2:3] * ac[:,:,1:2], \
		ab[:,:,2:3] * ac[:,:,0:1] - ab[:,:,0:1] * ac[:,:,2:3], \
		ab[:,:,0:1] * ac[:,:,1:2] - ab[:,:,1:2] * ac[:,:,0:1]), 2)
	vn = torch.zeros_like(v[:,:,:3])
	vn_ = vn_.permute(1,0,2).reshape(len(tri),-1)
	for j in range(3):
		i = torch.cat((tri[:,j].view(1,-1), \
			torch.LongTensor(range(len(tri))).view(1,-1).to(tri.device)), 0)
		if vn.dtype == torch.float32:
			I = torch.sparse.FloatTensor(i, torch.ones(len(tri),device=tri.device), \
				torch.Size([int(v.shape[1]),len(tri)]))
		elif vn.dtype == torch.float64:
			I = torch.sparse.DoubleTensor(i, torch.ones(len(tri),device=tri.device), \
				torch.Size([int(v.shape[1]),len(tri)]))
		else:
			raise ValueError('Not supported type')
		vnj = torch.sparse.mm(I, vn_)
		vn += vnj.view(int(v.shape[1]),-1,3).permute(1,0,2)
	return normalize(vn)
def save_obj(file_name, v, tri = [], vt = [], trit = [], vn = [], trin = []):
	if len(trit) == 0 and len(vt) == len(v):
		trit = tri
	elif len(trit) != len(tri):
		vt = []
		trit = []
	if len(trin) == 0 and len(vn) == len(v):
		trin = tri
	elif len(trin) != len(tri):
		vn = []
		trin = []
	with open(file_name, 'w') as fid:
		for i in range(len(v)):
			fid.write(('v'+' %f'*len(v[i])+'\n') % \
				tuple(v[i]))
		for i in range(len(vt)):
			fid.write(('vt'+' %f'*len(vt[i][:2])+'\n') % \
				tuple(vt[i][:2]))
		for i in range(len(vn)):
			fid.write(('vn'+' %f'*len(vn[i][:3])+'\n') % \
				tuple(vn[i][:3]))
		for i in range(len(tri)):
			if len(trit) > i and len(trit[i]) >= len(tri[i]):
				if len(trin) > i and len(trin[i]) >= len(tri[i]):
					fid.write(('f'+' %d/%d/%d'*len(tri[i])+'\n') % tuple([ \
						(trit[i][j]+1 if k==1 else trin[i][j]+1) if k else \
						tri[i][j]+1 for j in range(len(tri[i])) \
						for k in range(3)]))
				else:
					fid.write(('f'+' %d/%d'*len(tri[i])+'\n') % tuple([ \
						trit[i][j]+1 if k else tri[i][j]+1 \
						for j in range(len(tri[i])) for k in range(2)]))
			elif len(trin) > i and len(trin[i]) >= len(tri[i]):
				fid.write(('f'+' %d//%d'*len(tri[i])+'\n') % tuple([ \
					trin[i][j]+1 if k else tri[i][j]+1 \
					for j in range(len(tri[i])) for k in range(2)]))
			else:
				fid.write(('f'+' %d'*len(tri[i])+'\n') % tuple([ \
					tri[i][j]+1 for j in range(len(tri[i]))]))
	return os.path.exists(file_name)
if __name__ == '__main__':
	from face_model import load_bfm
	model, tri = load_bfm()
	z = model.random_input()
	v = model(z)
	v = random_apply_pose3D([1,0,0,0,0,0,0], v)
	n = mesh_point_normal(v, tri)
	if len(sys.argv) > 1:
		out_name = sys.argv[1]
		for i in range(len(v)):
			if out_name[-4:] == '.obj':
				v_  = v[i].detach().cpu().numpy()
				n_  = n[i].detach().cpu().numpy()
				tri_= tri.cpu().numpy()
				save_obj(out_name, v_, tri_, vn = n_)
			else:
				from op import rasterize
				import cv2
				nr = rasterize(v, n, tri, 256)
				n_ = nr[i].detach().cpu().numpy()
				cv2.imwrite(out_name, (n_[:,:,::-1]+1)*127.5)
