import math
import pickle
import torch
from torch import distributed as dist
from torch.utils.data.sampler import Sampler
def get_rank():
	if not dist.is_available() \
	or not dist.is_initialized():
		return 0
	else:
		return dist.get_rank()
def get_world_size():
	if not dist.is_available() \
	or not dist.is_initialized():
		return 1
	else:
		return dist.get_world_size()
def synchronize():
	if get_world_size() > 1:
		dist.barrier()
def reduce_sum(tensor):
	if  dist.is_available() \
	and dist.is_initialized():
		tensor = tensor.clone()
		dist.all_reduce(tensor, op = dist.ReduceOp.SUM)
	return tensor
def gather_grad(params):
	world_size = get_world_size()
	if world_size <= 1:
		return
	for param in params:
		if param.grad is not None:
			dist.all_reduce(param.grad.data, op = dist.ReduceOp.SUM)
			param.grad.data.div_(world_size)
def all_gather(data):
	world_size = get_world_size()
	if world_size <= 1:
		return [data]
	buffer = pickle.dumps(data)
	storage = torch.ByteStorage.from_buffer(buffer)
	tensor = torch.ByteTensor(storage).to('cuda')

	local_size = torch.IntTensor([tensor.numel()]).to('cuda')
	size_list = [torch.IntTensor([0]).to('cuda') for _ in range(world_size)]
	dist.all_gather(size_list, local_size)
	size_list = [int(size.item()) for size in size_list]
	max_size = max(size_list)

	tensor_list = []
	for _ in size_list:
		tensor_list.append(torch.ByteTensor(size=(max_size,)).to('cuda'))

	if local_size != max_size:
		padding = torch.ByteTensor(size=(max_size - local_size,)).to('cuda')
		tensor = torch.cat((tensor, padding), 0)

	dist.all_gather(tensor_list, tensor)
	data_list = []
	for size, tensor in zip(size_list, tensor_list):
		buffer = tensor.cpu().numpy().tobytes()[:size]
		data_list.append(pickle.loads(buffer))
	return data_list
def reduce_loss_dict(loss_dict):
	world_size = get_world_size()
	if world_size <= 1:
		return loss_dict
	with torch.no_grad():
		keys = []
		losses = []
		for k in sorted(loss_dict.keys()):
			keys.append(k)
			losses.append(loss_dict[k])
		losses = torch.stack(losses, 0)
		dist.reduce(losses, dst = 0)
		if dist.get_rank() == 0:
			losses /= world_size
		reduced_losses = {k: v for k, v in zip(keys, losses)}
	return reduced_losses
def get_dataloader(dataset, batch = 1, shuffle = True):
	if dist.is_available():
		sampler = dist.DistributedSampler(dataset)
	elif shuffle:
		sampler = torch.utils.data.RandomSampler(dataset)
	else:
		sampler = torch.utils.data.SequentialSampler(dataset)
	return	data.DataLoader(dataset, \
		batch_size = batch, \
		sampler = sampler, \
		drop_last = True)
def initialize(args):
	if dist.is_available():
		torch.cuda.set_device(args.gpu[args.local_rank])
		if hasattr(args, 'seed'):
			torch.manual_seed(args.seed + dist.get_rank())
			torch.cuda.manual_seed(args.seed + dist.get_rank())
		dist.init_process_group(backend = 'nccl')
		synchronize()
def construct_ddp(model, args):
	if dist.is_available():
		from torch.nn.parallel import DistributedDataParallel as DDP
		model = DDP(model, \
			device_ids = args.gpu[args.local_rank:args.local_rank+1], \
			output_device = args.gpu[local_rank], \
			broadcast_buffers = False)
	return model
