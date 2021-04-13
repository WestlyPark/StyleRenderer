import torch
import torch.nn as nn
import numpy as np
class LinearMorphableModel(nn.Module):
	def __init__(self, vertices_num, shape_dim = 0, expression_dim = 0, \
			vertices_mean = None, \
			w_shape_numpy = None, \
			w_expression_numpy = None, \
			sigma_shape = 1, \
			sigma_expression = .01, \
			learnable = False):
		super(LinearMorphableModel, self).__init__()
		vertices_num = max(int(vertices_num), 1)
		shape_dim = max(int(shape_dim), 0)
		expression_dim = max(int(expression_dim), 0)
		v = (np.random.rand(vertices_num * 3) \
			.astype(np.float32) * 2 - 1) * np.sqrt(shape_dim + expression_dim)
		w = (np.random.rand(shape_dim + expression_dim, v.shape[0]) \
			.astype(v.dtype) * 2 - 1) * np.sqrt(shape_dim + expression_dim)
		if vertices_mean is not None:
			vertices_mean = np.array(vertices_mean, np.float32)
			if vertices_mean.shape[0] == 3:
				vertices_mean = vertices_mean.reshape(3,-1).T
			elif len(vertices_mean.shape) > 1:
				vertices_mean = vertices_mean.reshape(-1,vertices_mean.shape[-1])
			else:
				vertices_mean = vertices_mean.reshape(-1,3)
			n = min(vertices_num, vertices_mean.shape[0])
			v[:3*n] = vertices_mean[:n,:3].reshape(-1)
		if w_shape_numpy is not None:
			w_shape_numpy = np.array(w_shape_numpy, np.float32)
			w_shape_numpy = w_shape_numpy.reshape( \
				(-1, w_shape_numpy.shape[-1]))
			if w_shape_numpy.shape[0] == w.shape[1] \
			and w_shape_numpy.shape[1] >= shape_dim:
				w_shape_numpy = w_shape_numpy.T
			d = min(shape_dim, w_shape_numpy.shape[0])
			n = min(vertices_num, w_shape_numpy.shape[1]//3)
			w[:d, :3*n] = w_shape_numpy[:d, :3*n]
		if w_expression_numpy is not None and expression_dim > 0:
			w_expression_numpy = np.array(w_expression_numpy, np.float32)
			w_expression_numpy = w_expression_numpy.reshape( \
				(-1, w_expression_numpy.shape[-1]))
			if w_expression_numpy.shape[0] == w.shape[1] \
			and w_expression_numpy.shape[1] >= expression_dim:
				w_expression_numpy = w_expression_numpy.T
			d = min(expression_dim, w_expression_numpy.shape[0])
			n = min(vertices_num, w_expression_numpy.shape[1]//3)
			w[shape_dim:shape_dim+d, :3*n] = w_expression_numpy[:d, :3*n]
		sigma_shape = [] if sigma_shape is None \
			else np.reshape(sigma_shape, -1)
		sigma_expression= [] if sigma_expression is None \
			else np.reshape(sigma_expression, -1)
		self.dim = [shape_dim, expression_dim, vertices_num * 3]
		self.fc = nn.Linear(shape_dim + expression_dim, vertices_num * 3, bias = True)
		self.sigma = nn.Parameter(torch.Tensor( \
			[abs(sigma_shape[i]) if len(sigma_shape) > i else \
			(abs(sigma_shape[-1])if len(sigma_shape) > 0 else 1) \
			for i in range(self.dim[0])] + \
			[abs(sigma_expression[i]) if len(sigma_expression) > i else \
			(abs(sigma_expression[-1])if len(sigma_expression) > 0 else 1) \
			for i in range(self.dim[1])]), requires_grad = False)
		with torch.no_grad():
			self.fc.weight.copy_(torch.from_numpy(w.T).float())
			self.fc.bias.copy_(torch.from_numpy(v).float())
		if not learnable:
			self.fc.weight.requires_grad = False
			self.fc.bias.requires_grad = False
	def random_input(self, batch_size = 1):
		return torch.normal(mean = 0, std = self.sigma.unsqueeze(0).expand(batch_size,-1))
	def forward(self, x):
		return torch.reshape(self.fc(x), (-1, self.dim[2]//3, 3))
	def regulation(self, x):
		return ((x / self.sigma[np.newaxis,:])**2).sum()
class BlendShapeModel(nn.Module):
	def __init__(self, vertices_num, shape_dim = 0, expression_dim = 0, \
			bs = None, \
			beta_shape = 1, \
			beta_expression = [1,10], \
			learnable = False):
		super(BlendShapeModel, self).__init__()
		vertices_num = max(int(vertices_num), 1)
		shape_dim = max(int(shape_dim), 0)
		expression_dim = max(int(expression_dim), 0)
		w = (np.random.rand(shape_dim+1, expression_dim+1, vertices_num * 3) \
			.astype(np.float32) * 2 - 1) * np.sqrt(shape_dim + expression_dim)
		if bs is not None:
			bs = np.array(bs, np.float32)
			if len(bs.shape) >= 3:
				bs = bs.reshape(bs.shape[0],bs.shape[1],-1)
				if bs.shape[0] == w.shape[-1]:
					bs = np.transpose(bs, [1,2,0])
				elif bs.shape[2] == w.shape[-1]:
					bs = bs
				d = [	min(bs.shape[0],w.shape[0]), \
					min(bs.shape[1],w.shape[1]), \
					min((bs.shape[2]//3)*3, w.shape[2])]
				w[:d[0],:d[1],:d[2]] = bs[:d[0],:d[1],:d[2]]
		beta_shape = [] if beta_shape is None \
			else np.reshape(beta_shape, -1)
		beta_expression = [] if beta_expression is None \
			else np.reshape(beta_expression, -1)
		self.dim = [shape_dim, expression_dim, vertices_num * 3]
		self.beta = nn.Parameter(torch.Tensor( \
			[abs(beta_shape[i]) if len(beta_shape) > i else \
			(abs(beta_shape[-1])if len(beta_shape) > 0 else 1) \
			for i in range(self.dim[0]+1)] + \
			[abs(beta_expression[2*i+j])if len(beta_expression) > 2*i+1 else \
			(abs(beta_expression[j-2])  if len(beta_expression) > 1 else 1) \
			for i in range(self.dim[1]) for j in range(2)]), requires_grad = False)
		self.sample =  \
			[torch.distributions.dirichlet.Dirichlet(self.beta[:self.dim[0]+1])] + \
			[torch.distributions.beta.Beta( \
				self.beta[self.dim[0]+2*i+1], \
				self.beta[self.dim[0]+2*i+2]) \
				for i in range(self.dim[1])]
		self.weight = nn.Parameter(torch.from_numpy(w).float())
		if not learnable:
			self.weight.requires_grad = False
	def random_input(self, batch_size = 1, eps = 1e-9):
		xs = self.sample[0].sample([batch_size])
		xs = torch.log(xs[:,:-1]/torch.clamp(xs[:,-1:], min = eps))
		xe = torch.cat([self.sample[i+1].sample([batch_size, 1]) \
			for i in range(self.dim[1])], 1)
		return	torch.cat(( \
			xs - torch.sum(xs, dim = 1, keepdim = True) / float(self.dim[0]), \
			torch.log(xe/torch.clamp(1-xe, min = eps))), 1)
	def forward(self, x):
		xs = nn.functional.softmax( \
			torch.cat((x[:,:self.dim[0]], \
			-torch.sum(x[:,:self.dim[0]], dim = 1, keepdim = True)), 1), dim = 1)
		xe = torch.sigmoid(x[:,self.dim[0]:])
		xe = torch.cat((1-torch.sum(xe, dim = 1, keepdim = True), xe), 1)
		return	torch.matmul(xe.view(-1,1,self.dim[1]+1), torch.matmul(xs, \
			self.weight.view(self.dim[0]+1,-1)).view(-1,self.dim[1]+1,self.dim[2])) \
			.view(-1, self.dim[2]//3, 3)
	def regulation(self, x):
		xs = torch.cat((x[:,:self.dim[0]], \
			-torch.sum(x[:,:self.dim[0]], dim = 1, keepdim = True)), 1)
		xe = x[:,self.dim[0]:]
		return-((xs * self.beta[np.newaxis,:self.dim[0]+1]).sum() - \
				torch.log(torch.exp(xs).sum(1)).sum() * \
				(self.beta[:self.dim[0]+1].sum()-self.dim[0]-1) + \
			(xe *(self.beta[np.newaxis,self.dim[0]+1:-1:2])-1).sum() - \
				(torch.log(torch.exp(xe) + 1) * \
				(self.beta[self.dim[0]+1:].view(1,-1,2).sum(2)-2)).sum())
class LinearBlendSkinningModel(nn.Module):
	def __init__(self, vertices_num, pose_nodes = 0, shape_dim = 0, \
			v_template = None, \
			J_regressor = None, \
			kintree_table = None, \
			weights = None, \
			posedirs = None, \
			shapedirs = None, \
			sigma_shape = 1, \
			sigma_pose = 1, \
			mean_pose = 0, \
			learnable = False):
		super(LinearBlendSkinningModel, self).__init__()
		vertices_num = max(int(vertices_num), 1)
		shape_dim = max(int(shape_dim), 0)
		pose_nodes = max(int(pose_nodes), 1)
		if kintree_table is not None:
			kintree_table = np.array(kintree_table, np.int32)
			if len(kintree_table.shape) == 1:
				if len(kintree_table) == pose_nodes-1:
					kintree_table = np.concatenate(([-1],kintree_table))
				kintree_table = np.vstack((kintree_table, np.arange(pose_nodes)))
			elif kintree_table.shape[1] == 2 and kintree_table.shape[0] == pose_nodes:
				kintree_table = kintree_table.T
			order = list(kintree_table[1,np.where(np.logical_or(kintree_table[0] < 0, \
					kintree_table[0] >= pose_nodes))[0]])
			i = 0; k = len(order)
			# ensure the parent joint is before the child joint
			while len(order) < pose_nodes:
				order += list(kintree_table[1,np.where( \
					kintree_table[0] == order[i])[0]])
				i += 1
			self.parent = kintree_table[0, order[k:]]
			if len(np.where(order != np.arange(len(order)))[0]) > 0:
				j = j[order,:]
				w = w[:,order]
				order3 = np.vstack((3*order,3*order+1,3*order+2)).T.reshape(-1)
				s[shape_dim:,:] = s[shape_dim+order3,:]
				inv_order = np.zeros_like(order)
				for i in range(len(order)):
					inv_order[order[i]] = i
				self.parent = inv_order(self.parent)
		else:
			self.parent = np.zeros(pose_nodes-1, np.int64)
		v = (np.random.rand(vertices_num * 3) \
			.astype(np.float32) * 2 - 1) * np.sqrt(shape_dim +len(self.parent)*9)
		s = (np.random.rand(shape_dim +(pose_nodes-1)*9, v.shape[0])
			.astype(np.float32) * 2 - 1) * np.sqrt(shape_dim +len(self.parent)*9)
		j = (np.random.rand(pose_nodes, vertices_num) \
			.astype(np.float32) * 2 - 1) * np.sqrt(pose_nodes)
		if v_template is not None:
			v_template = np.array(v_template, np.float32)
			if v_template.shape[0] == 3:
				v_template = v_template.reshape(3,-1).T
			elif len(v_template.shape) > 1:
				v_template = v_template.reshape(-1,v_template.shape[-1])
			else:
				v_template = v_template.reshape(-1,3)
			n = min(vertices_num, v_template.shape[0])
			v[:3*n] = v_template[:n,:3].reshape(-1)
		if shapedirs is not None:
			shapedirs = np.array(shapedirs, np.float32)
			shapedirs = shapedirs.reshape(-1, shapedirs.shape[-1])
			if shapedirs.shape[0] == s.shape[1] \
			and shapedirs.shape[1] >= shape_dim:
				shapedirs = shapedirs.T
			d = min(shape_dim, shapedirs.shape[0])
			n = min(vertices_num, shapedirs.shape[1]//3)
			s[:d, :3*n] = shapedirs[:d, :3*n]
		if posedirs is not None:
			posedirs = np.array(posedirs, np.float32)
			posedirs = posedirs.reshape(-1, posedirs.shape[-1])
			if posedirs.shape[0] == s.shape[1] \
			and posedirs.shape[1] >= len(self.parent)*9:
				posedirs = posedirs.T
			d = min(len(self.parent)*9, shapedirs.shape[0])
			n = min(vertices_num, posedirs.shape[1]//3)
			s[shape_dim:shape_dim+d, :3*n] = posedirs[:d, :3*n]
		if J_regressor is not None:
			import scipy.sparse as sp
			if isinstance(J_regressor, sp.csc_matrix):
				J_regressor = J_regressor.astype(np.float32).todense()
			else:
				J_regressor = np.array(J_regressor, np.float32)
			if J_regressor.shape[1] == pose_nodes \
			and J_regressor.shape[0] >= vertices_num:
				J_regressor = J_regressor.T
			m = min(pose_nodes, J_regressor.shape[0])
			n = min(vertices_num, J_regressor.shape[1])
			j[:m,:n] = J_regressor[:m,:n]
		w = np.zeros((vertices_num, pose_nodes), np.float32)
		if weights is not None:
			weights = np.array(weights, np.float32)
			if weights.shape[0] == pose_nodes \
			and weights.shape[1] >= vertices_num:
				weights = weights.T
			m = min(pose_nodes, weights.shape[1])
			n = min(vertices_num, weights.shape[0])
			w[:n,:m] = weights[:n,:m]
		else:
			from sklearn.neighbors import NearestNeighbors
			J = j.dot(v.reshape(-1,3))
			nbrs = NearestNeighbors(n_neighbors = 1, algorithm = 'kd_tree').fit(J)
			dis, idx = nbrs.kneighbors(v.reshape(-1,3))
			w.reshape(-1)[idx + np.arange(vertices_num).reshape(-1,1)*pose_nodes] = \
				np.exp(-dis*dis/(dis.max()*dis.max()))
		w = abs(w); w = w / np.maximum(w.sum(1).reshape(-1,1), 1e-5)
		sigma_shape = [] if sigma_shape is None \
			else np.reshape(sigma_shape, -1)
		sigma_pose = [] if sigma_pose is None \
			else np.reshape(sigma_pose, -1)
		mean_pose = [] if mean_pose is None \
			else np.reshape(mean_pose, -1)
		self.dim = [shape_dim, len(self.parent)* 3, vertices_num * 3]
		self.fc = [nn.Parameter(torch.from_numpy(s).float()), \
			nn.Parameter(torch.from_numpy(v).float())]
		self.weight = [nn.Parameter(torch.from_numpy(w).float()), \
			nn.Parameter(torch.from_numpy(j).float())]
		self.sigma = nn.Parameter(torch.Tensor( \
			[abs(sigma_shape[i]) if len(sigma_shape) > i else \
			(abs(sigma_shape[-1])if len(sigma_shape) > 0 else 1) \
			for i in range(self.dim[0])] + [1] * self.dim[1]), requires_grad = False)
		if len(mean_pose) <= len(self.parent):
			self.pose_mean = nn.Parameter(torch.cat([ \
				(mean_pose[i] if len(mean_pose) > i else \
				(mean_pose[-1]if len(mean_pose) > 0 else 0)) * \
				torch.ones(3,dtype=torch.float32) \
				for i in range(len(self.parent))], 0), requires_grad = False)
		else:
			self.pose_mean = nn.Parameter(torch.Tensor( \
				[mean_pose[i] if len(sigma_pose) > i else \
				(mean_pose[-1] if len(sigma_pose) > 0 else 0) \
				for i in range(self.dim[1])]))
		if len(sigma_pose) <= len(self.parent):
			self.pose_cov = nn.Parameter(torch.cat([ \
				(sigma_pose[i] if len(sigma_pose) > i else \
				(sigma_pose[-1]if len(sigma_pose) > 0 else 1)) * \
				torch.eye(3,3,dtype=torch.float32).view(1,3,3) \
				for i in range(len(self.parent))], 0), requires_grad = False)
		elif len(sigma_pose) <=len(self.parent)*3:
			self.pose_cov = torch.Tensor( \
				[sigma_pose[i] if len(sigma_pose) > i else \
				(sigma_pose[-1] if len(sigma_pose) > 0 else 1) \
				for i in range(self.dim[1])])
			self.pose_cov = nn.Parameter(torch.cat([ \
				torch.diag(self.pose_cov[3*i:3*i+3]).view(1,3,3) \
				for i in range(len(self.parent))], 0), requires_grad = False)
		else:
			self.pose_cov = torch.Tensor( \
				[sigma_pose[i] if len(sigma_pose) > i else \
				(sigma_pose[-1] if len(sigma_pose) > 0 else ((i%9)%4==0)) \
				for i in range(self.dim[1]*3)])
			self.pose_cov = nn.Parameter(torch.cat([ \
				self.pose_cov[9*i:9*i+9].view(1,3,3) \
				for i in range(len(self.parent))], 0), requires_grad = False)
		self.pose_inv = torch.inverse(self.pose_cov)
		if not learnable:
			for x in self.weight + self.fc:
				x.requires_grad = False
	def random_input(self, batch_size = 1):
		x = torch.normal(mean = 0, std = self.sigma.unsqueeze(0).expand(batch_size,-1))
		x = torch.cat([x[:,:self.dim[0]]] + [ \
			torch.matmul(x[:,self.dim[0]+3*i:self.dim[0]+3*i+3],self.pose_cov[i]) + \
			self.pose_mean[np.newaxis,3*i:3*i+3] \
			for i in range(self.dim[1]//3)], 1)
		return x
	def forward(self, x):
		from utils_3d import rodrigues
		v_shaped = torch.matmul(x[:,:self.dim[0]], self.fc[0][:self.dim[0],:]) + \
			self.fc[1].view(1,-1)
		R = rodrigues(x[:,self.dim[0]:].reshape(-1,3)).view(-1,self.dim[1]//3,3,3)
		J = torch.matmul(self.weight[1].view(1,-1,self.dim[2]//3).expand(x.shape[0],-1,-1), \
			v_shaped.view(-1,self.dim[2]//3,3))
		v_posed =(torch.matmul((R - torch.eye(3,3,dtype=x.dtype).view(1,1,3,3)) \
			.view(-1,self.dim[1]*3), self.fc[0][self.dim[0]:,:]) + \
			v_shaped).view(-1,self.dim[2]//3,3)
		T = [torch.cat(( \
			torch.eye(3,3,dtype = x.dtype).view(1,3,3).expand(x.shape[0],-1,-1), \
			J[:,i:i+1,:]), 1) for i in range(J.shape[1]-len(self.parent))]
		for i, p in enumerate(self.parent):
			T += [torch.cat(( \
				torch.matmul(R[:,i].permute([0,2,1]), T[p][:,:3,:3]), \
				torch.matmul(J[:,len(T):len(T)+1,:]-J[:,p:p+1,:],T[p][:,:3,:3])+ \
				T[p][:,3:4,:3]),1)]
		v = sum([self.weight[0][:,i].view(1,-1,1) * (\
			torch.matmul(v_posed-J[:,i:i+1,:], T[i][:,:3,:3]) + T[i][:,3:4,:3]) \
			for i in range(J.shape[1])])
		return v
	def regulation(self, x):
		l_shape = ((x[:,:self.dim[0]] / self.sigma[np.newaxis,:self.dim[0]])**2).sum()
		l_pose = sum([(( \
			torch.matmul(x[:,self.dim[0]+3*i:self.dim[0]+3*i+3], \
			self.pose_inv[i]))**2).sum() \
			for i in range(self.dim[1]//3)])
		return l_shape + l_pose
def load_bfm(file_name = '/data/BaselFaceModel.mat'):
	if isinstance(file_name, str):
		import scipy.io as sio
		data = sio.loadmat(file_name)
	else:
		data = file_name
	v = (data['v'] - data['v'].mean(1).reshape(-1,1)).T * 1e-5
	w_shape = data['w_shape'] * 1e-5
	w_exp = data['w_exp'] * 1e-5
	if 'sigma_shape' in data.keys():
		w_shape = w_shape.dot(np.diag(data['sigma_shape'].reshape(-1)))
	if 'sigma_exp' in data.keys():
		w_exp = w_exp.dot(np.diag(data['sigma_exp'].reshape(-1)))
	tri = (data['tri'][0,0]).astype(np.int64)
	tri = tri - tri.min()
	if tri.shape[0] == 3 and tri.shape[1] != 3:
		tri = tri.T
	model = LinearMorphableModel(len(v), \
		w_shape.shape[1], w_exp.shape[1], \
		v, w_shape, w_exp)
	return model, torch.from_numpy(tri)
def load_facewarehouse(file_name = '/data/FaceWareHouse.mat'):
	if isinstance(file_name, str):
		import scipy.io as sio
		data = sio.loadmat(file_name)
	else:
		data = file_name
	v_mean = np.tile(data['v'].mean(1).reshape(-1,1,1), \
		(data['v'].shape[1],1,1))
	bs = np.transpose(data['p'] - v_mean, [2,1,0])
	tri = (data['tri'] - data['tri'].min()).astype(np.int64)
	if tri.shape[0] == 3 and tri.shape[1] != 3:
		tri = tri.T
	model = BlendShapeModel(data['v'].shape[1], \
		bs.shape[0]-1, bs.shape[1]-1, bs, .01)
	return model, torch.from_numpy(tri)
def load_flame(file_name = '/data/flame/generic_model.mat'):
	if isinstance(file_name, str):
		if file_name[-4:] == '.pkl':
			import pickle
			with open(file_name, 'rb') as f:
				data = pickle.load(f, encoding = 'latin1')
		elif file_name[-4:] == '.mat':
			import scipy.io as sio
			data = sio.loadmat(file_name)
	else:
		data = file_name
	# pitch,yaw,roll, in degree
	neck = [10, 30, 5]
	jaw = [10,1,1]
	eye = [10,10,1e-5]
	model = LinearBlendSkinningModel(data['v_template'].shape[0], \
			data['posedirs'].shape[-1] // 9 + 1, \
			data['shapedirs'].shape[-1], \
			data['v_template'], \
			data['J_regressor'], \
			data['kintree_table'], \
			data['weights'], \
			data['posedirs'], \
			data['shapedirs'], 1, \
			[i*np.pi/180 for i in neck] + \
			[i*np.pi/180 for i in jaw] + \
			[i*np.pi/180 for i in eye] * 2)
	tri = (data['f'] - data['f'].min()).astype(np.int64)
	if tri.shape[0] == 3 and tri.shape[1] != 3:
		tri = tri.T
	return model, torch.from_numpy(tri)
import sys
if __name__ == '__main__':
	if len(sys.argv) > 1:
		if sys.argv[1][-4:] == '.mat':
			import scipy.io as sio
			data = sio.loadmat(sys.argv[1])
			if 'v' in data.keys() and 'w_shape' in data.keys() and 'w_exp' in data.keys():
				model, tri = load_bfm(data)
			elif 'p' in data.keys() and 'v' in data.keys():
				model, tri = load_facewarehouse(data)
			elif 'J_regressor' in data.keys():
				data['shapedirs'] = data['shapedirs'][0,0][0]
				model, tri = load_flame(data)
			else:
				model = None
		elif sys.argv[1][-4:] == '.pkl':
			model, tri = load_flame(sys.argv[1])
		if model is not None:
			z = model.random_input()
			v = model(z)
			if len(sys.argv) > 2 and sys.argv[2][-4:] == '.obj':
				v = v[0].detach().cpu().numpy()
				tri = tri.cpu().numpy()
				from utils_3d import save_obj
				save_obj(sys.argv[2], v, tri)
			else:
				l = model.regulation(z)
				print(l.cpu().numpy())

