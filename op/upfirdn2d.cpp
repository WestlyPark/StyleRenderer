#include <torch/extension.h>
struct UpFirDn2DKernelParams {
	int up_x;
	int up_y;
	int down_x;
	int down_y;
	int pad_x0;
	int pad_x1;
	int pad_y0;
	int pad_y1;

	int major_dim;
	int in_h;
	int in_w;
	int minor_dim;
	int kernel_h;
	int kernel_w;
	int out_h;
	int out_w;
	int loop_major;
	int loop_x;
};
extern bool upfirdn2d_op(float*,const float*,const float*,UpFirDn2DKernelParams&,int,int);
torch::Tensor upfirdn2d(const torch::Tensor& input, const torch::Tensor& kernel,
		int up_x, int up_y, int down_x, int down_y,
		int pad_x0, int pad_x1, int pad_y0, int pad_y1) {
	auto x = input.contiguous();
	auto k = kernel.contiguous();
	UpFirDn2DKernelParams p;
	p.major_dim = x.size(0);
	p.in_h = x.size(1);
	p.in_w = x.size(2);
	p.minor_dim = x.size(3);
	p.kernel_h = k.size(0);
	p.kernel_w = k.size(1);
	p.up_x = up_x;
	p.up_y = up_y;
	p.down_x = down_x;
	p.down_y = down_y;
	p.pad_x0 = pad_x0;
	p.pad_x1 = pad_x1;
	p.pad_y0 = pad_y0;
	p.pad_y1 = pad_y1;
	p.out_h = (p.in_h * p.up_y + p.pad_y0 + p.pad_y1 - p.kernel_h + p.down_y) / p.down_y;
	p.out_w = (p.in_w * p.up_x + p.pad_x0 + p.pad_x1 - p.kernel_w + p.down_x) / p.down_x;
	auto out = at::empty({p.major_dim, p.out_h, p.out_w, p.minor_dim}, x.options());
	int mode = -1;
	int tile_out_h = -1;
	int tile_out_w = -1;
	if(p.up_x == 1 && p.up_y == 1) {
		if(p.down_x == 1 && p.down_y == 1) {
			if(p.kernel_h <= 3 && p.kernel_w <= 3)
				mode = 2;
			else if(p.kernel_h <= 4 && p.kernel_w <= 4)
				mode = 1;
			tile_out_h = 16;
			tile_out_w = 64;

		} else if(p.down_x == 2 && p.down_y == 2) {
			if(p.kernel_h <= 2 && p.kernel_w <= 2)
				mode = 6;
			else if(p.kernel_h <= 4 && p.kernel_w <= 4)
				mode = 5;
			tile_out_h = 8;
			tile_out_w = 32;
		}
	} else if(p.up_x == 2 && p.up_y == 2)
		if(p.down_x == 1 && p.down_y == 1) {
			if(p.kernel_h <= 2 && p.kernel_w <= 2)
				mode = 4;
			else if(p.kernel_h <= 4 && p.kernel_w <= 4)
				mode = 3;
			tile_out_h = 16;
			tile_out_w = 64;
		}
	p.loop_major = tile_out_h;
	p.loop_x = tile_out_w;
	upfirdn2d_op(out.data<float>(),
		x.data<float>(),
		k.data<float>(),
		p, mode, x.type().is_cuda());
	return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("upfirdn2d", &upfirdn2d, "upfirdn2d (CUDA)");
}
