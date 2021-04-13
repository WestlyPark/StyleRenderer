#include <ATen/ATen.h>
#include <torch/extension.h>
extern bool fused_bias_act_op(float*,const float*,const float*,const float*,
	int,int,float,float,int,int,int,int,int,int); 
torch::Tensor fused_bias_act(
		const torch::Tensor& input,
		const torch::Tensor& bias,
		const torch::Tensor& refer,
		int act, int grad, float alpha, float scale) {
	auto x = input.contiguous();
	auto b = bias.contiguous();
	auto ref = refer.contiguous();
	int use_bias = b.numel() ? 1 : 0;
	int use_ref = ref.numel() ? 1 : 0;
	int size_x = x.numel();
	int size_b = b.numel();
	int step_b = 1;
	for(int i = 1 + 1; i < x.dim(); ++i)
		step_b *= x.size(i);
	auto y = torch::empty_like(x);
	fused_bias_act_op(
		y.data<float>(),
		x.data<float>(),
		b.data<float>(),
		ref.data<float>(),
		act, grad, (float)alpha, (float)scale,
		size_x, step_b, size_b, use_bias, use_ref,
		x.type().is_cuda());
	return y;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("fused_bias_act", &fused_bias_act, "fused bias act (CUDA)");
}
