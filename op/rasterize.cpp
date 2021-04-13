#include <vector>
#include <limits>
#include <stdlib.h>
#include <ATen/ATen.h>
#include <torch/extension.h>
#define CHECK_SIZE(x, sz) AT_ASSERTM(x.is_contiguous() && x.sizes() == sz, #x " input error")
template<typename scalar>
extern bool barycentric(scalar*,int64_t,int64_t,int64_t*,scalar*,scalar*,bool,scalar);
template<typename scalar>
extern bool barycentric_grad(const scalar*,scalar*,scalar,scalar,scalar*,bool,scalar);
template<typename scalar>
extern bool assign_buffer(const scalar*,scalar*,const scalar*,
	scalar*,scalar*,bool,const scalar*,scalar,scalar);
template<typename scalar, typename index>
extern bool rasterize_gpu(index,index,index,index,index,bool,bool,bool,
	const scalar*,const index*,index*,scalar*,scalar*,scalar);
template<typename scalar, typename index>
extern bool rasterize_gpu_backward(index,index,index,index,
        bool,bool,const scalar*,const index*,scalar*,scalar);
template<typename scalar, typename index>
index rasterize_cpu(index b, index nv, index nf, index h, index w,
		bool repeat_v, bool repeat_f, bool perspective,
		const scalar *v, const index *f, index *i, scalar *c,
		scalar *zB, scalar eps = 1e-6) {
	scalar Ainv[9], coeff[3], uv[2], det = 1;
	index count = 0;
	int64_t bbox[4];
	for(index batch = 0; batch < b; ++batch) {
		for(index t = 0; t < nf; ++t) {
			if(f == NULL || v == NULL
			|| f[3*t]>=nv || f[3*t+1]>=nv || f[3*t+2]>=nv
			|| f[3*t] < 0 || f[3*t+1] < 0 || f[3*t+2] < 0)
				continue;
			scalar v_[] = {
				v[3*f[3*t]],  v[3*f[3*t]+1],  v[3*f[3*t]+2],
				v[3*f[3*t+1]],v[3*f[3*t+1]+1],v[3*f[3*t+1]+2],
				v[3*f[3*t+2]],v[3*f[3*t+2]+1],v[3*f[3*t+2]+2]};
			if(barycentric<scalar>(v_, (int64_t)h, (int64_t)w, bbox, Ainv,
				&det, perspective, eps))
			for(index y = bbox[2]; y <= bbox[3]; ++y)
				for(index x = bbox[0]; x <= bbox[1]; ++x) {
					index ind = x + y * w;
					uv[0] = (scalar)x;
					uv[1] = (scalar)y;
					coeff[0] = Ainv[0] + Ainv[3] * uv[0] + Ainv[6] * uv[1];
					coeff[1] = Ainv[1] + Ainv[4] * uv[0] + Ainv[7] * uv[1];
					coeff[2] = Ainv[2] + Ainv[5] * uv[0] + Ainv[8] * uv[1];
					if(assign_buffer<scalar>(v_, coeff, uv,
						zB== NULL ? NULL : zB+ ind,
						c == NULL ? NULL : c + ind*3, 
						perspective, Ainv, det, eps)
					&& i != NULL) {
						i[ind*3]  = f[3*t]  + (repeat_v ? 0 : nv * batch);
						i[ind*3+1]= f[3*t+1]+ (repeat_v ? 0 : nv * batch);
						i[ind*3+2]= f[3*t+2]+ (repeat_v ? 0 : nv * batch);
					}
				}
			++count;
		}
		if(!repeat_v && v != NULL) v += nv * 3;
		if(!repeat_f && f != NULL) f += nf * 3;
		c = (c == NULL ? NULL : c + h * w * 3);
		i = (i == NULL ? NULL : i + h * w * 3);
		zB= (zB== NULL ? NULL : zB+ h * w);
	}
	return count;
}
template<typename scalar, typename index>
index rasterize_cpu_backward(index b, index n, index h, index w,
		bool repeat_v, bool perspective, const scalar *v,
		const index *i, scalar *dcoeff, scalar eps = 1e-6) {
	index count = 0;
	scalar uv[2];
	for(index batch = 0; batch < b; ++batch) {
		for(index t = 0; t < h * w; ++t) {
			if(i == NULL || v == NULL
			|| i[3*t] == i[3*t+1] || i[3*t] == i[3*t+2] || i[3*t+1] == i[3*t+2]
			|| i[3*t] < 0 || i[3*t+1] < 0 || i[3*t+2] < 0
	                || i[3*t]>=n*b|| i[3*t+1]>=n*b|| i[3*t+2]>=n*b)
				continue;
			scalar v_[] = {
				v[3*i[3*t]],  v[3*i[3*t]+1],  v[3*i[3*t]+2],
				v[3*i[3*t+1]],v[3*i[3*t+1]+1],v[3*i[3*t+1]+2],
				v[3*i[3*t+2]],v[3*i[3*t+2]+1],v[3*i[3*t+2]+2]};
			uv[0] = (scalar)(t % w);
			uv[1] = (scalar)(t / w);
			barycentric_grad<scalar>(v_, uv, (scalar)h, (scalar)w,
				dcoeff + t * 27, perspective, eps);
			++count;
		}
		if(dcoeff != NULL) dcoeff += h * w * 27;
		i = (i == NULL ? NULL : i + h * w * 3);
	}
	return count;
}
using namespace torch;
std::vector<torch::Tensor> rasterize_forward(
		const torch::Tensor &vertices,
		const torch::Tensor &triangles,
		int64_t height, int64_t width,
		bool perspective = false,
		double eps = 1e-9) {
	int64_t	b = (vertices.sizes().size() > 0 ? vertices.size(0) : 0),
		n = (vertices.sizes().size() > 1 ? vertices.size(1) : 0),
		f = (triangles.sizes().size() > 0 ? triangles.size(0) : 0),
		h = height<= 0 ? 1 : height,
		w = width <= 0 ? h : width;
	eps = (eps < 0 ? -eps : eps);
	std::vector<int64_t> sz;
	if(b > 0 && n == 3 && vertices.sizes().size() == 2) {
		n = b; b = 1; sz = std::vector<int64_t>{n, 3};
	} else
		sz = std::vector<int64_t>{b, n, 3};
	CHECK_SIZE(vertices, sz);
	if(triangles.sizes().size() == 3 && triangles.size(2) == 3
	&&(f == b || sz.size() == 2)) {
		b = f; f = triangles.size(1);
		sz = std::vector<int64_t>{b, f, 3};
	} else
		sz = std::vector<int64_t>{f, 3};
	CHECK_SIZE(triangles, sz);
	AT_ASSERTM(!(triangles.is_cuda() ^ vertices.is_cuda()), " cuda input error");
	sz = (vertices.sizes().size() == 2 && triangles.sizes().size() == 2 ?
		std::vector<int64_t>{h, w} : std::vector<int64_t>{b, h, w});
	torch::Tensor index, coefficient;
	if(vertices.type().scalarType() == torch::ScalarType::Float
	&& triangles.type().scalarType() == torch::ScalarType::Long) {
		torch::Tensor buffer= -std::numeric_limits<float>::max() * torch::ones(sz,
			vertices.is_cuda() ? CUDA(kFloat): CPU(kFloat));
		sz.push_back(3);
		index = torch::zeros(sz, vertices.is_cuda() ? CUDA(kLong) : CPU(kLong)),
		coefficient = torch::zeros(sz, vertices.is_cuda() ? CUDA(kFloat): CPU(kFloat));
		const float *v = vertices.data<float>();
		const int64_t *tri = triangles.data<int64_t>();
		int64_t *ind = index.data<int64_t>();
		float	*zB = buffer.data<float>(),
			*coeff = coefficient.data<float>();
		if(vertices.is_cuda())
			rasterize_gpu<float,int64_t>(b, n, f, h, w,
				vertices.sizes().size() == 2,
				triangles.sizes().size() == 2,
				perspective,
				v, tri, ind, coeff, zB, (float)eps);
		else
			rasterize_cpu<float,int64_t>(b, n, f, h, w,
				vertices.sizes().size() == 2,
				triangles.sizes().size() == 2,
				perspective,
				v, tri, ind, coeff, zB, (float)eps);
	} else if(vertices.type().scalarType() == torch::ScalarType::Double
	&& triangles.type().scalarType() == torch::ScalarType::Long) {
		torch::Tensor buffer= -std::numeric_limits<double>::max() * torch::ones(sz,
			vertices.is_cuda() ? CUDA(kDouble): CPU(kDouble));
		sz.push_back(3);
		index = torch::zeros(sz, vertices.is_cuda() ? CUDA(kLong) : CPU(kLong)),
		coefficient = torch::zeros(sz, vertices.is_cuda() ? CUDA(kDouble) : CPU(kDouble));
		const double *v = vertices.data<double>();
		const int64_t *tri = triangles.data<int64_t>();
		int64_t *ind = index.data<int64_t>();
		double	*zB = buffer.data<double>(),
			*coeff = coefficient.data<double>();
		if(vertices.is_cuda())
			rasterize_gpu<double,int64_t>(b, n, f, h, w,
				vertices.sizes().size() == 2,
				triangles.sizes().size() == 2,
				perspective,
				v, tri, ind, coeff, zB, eps);
		else
			rasterize_cpu<double,int64_t>(b, n, f, h, w,
				vertices.sizes().size() == 2,
				triangles.sizes().size() == 2,
				perspective,
				v, tri, ind, coeff, zB, (double)eps);
	} else {
		AT_ASSERTM(false, " type error");
	}
	return {index, coefficient};
}
torch::Tensor rasterize_backward(
		const torch::Tensor &vertices,
		const torch::Tensor &index,
		bool perspective = false,
		double eps = 1e-9) {
	int64_t	b = (index.sizes().size() > 0 ? index.size(0) : 0),
		n = (vertices.sizes().size() > 1 ? vertices.size(1) : 0),
		h = (index.sizes().size() > 1 ? index.size(1) : 0),
		w = (index.sizes().size() > 2 ? index.size(2) : 0);
	eps = (eps < 0 ? -eps : eps);
	std::vector<int64_t> sz;
	if(n == 3 && vertices.sizes().size() == 2) {
		n = vertices.size(0);
		sz = std::vector<int64_t>{n, 3};
	} else
		sz = std::vector<int64_t>{b, n, 3};
	CHECK_SIZE(vertices, sz);
	if(w == 3 && index.sizes().size() == 3 && vertices.sizes().size() == 2) {
		w = h; h = b; b = 1;
		sz = std::vector<int64_t>{h, w, 3};
	} else
		sz = std::vector<int64_t>{b, h, w, 3};
	CHECK_SIZE(index, sz);
	AT_ASSERTM(!(index.is_cuda() ^ vertices.is_cuda()), " cuda error");
	sz.push_back(9);
	torch::Tensor grad_coefficient;
	if(vertices.type().scalarType() == torch::ScalarType::Float
	&& index.type().scalarType() == torch::ScalarType::Long) {
		grad_coefficient = torch::zeros(sz, vertices.is_cuda() ? CUDA(kFloat):CPU(kFloat));
		const float *v = vertices.data<float>();
		const int64_t *ind = index.data<int64_t>();
		float *dcoeff = grad_coefficient.data<float>();
		if(vertices.is_cuda())
			rasterize_gpu_backward<float,int64_t>(b, n, h, w,
				vertices.sizes().size() == 2,
				perspective,
				v, ind, dcoeff, (float)eps);
		else
			rasterize_cpu_backward<float,int64_t>(b, n, h, w,
				vertices.sizes().size() == 2,
				perspective,
				v, ind, dcoeff, (float)eps);
	} else 	if(vertices.type().scalarType() == torch::ScalarType::Double
	&& index.type().scalarType() == torch::ScalarType::Long) {
		grad_coefficient = torch::zeros(sz, vertices.is_cuda() ? CUDA(kDouble):CPU(kDouble));
		const double *v = vertices.data<double>();
		const int64_t *ind = index.data<int64_t>();
		double *dcoeff = grad_coefficient.data<double>();
		if(vertices.is_cuda())
			rasterize_gpu_backward<double,int64_t>(b, n, h, w,
				vertices.sizes().size() == 2,
				perspective,
				v, ind, dcoeff, eps);
		else
			rasterize_cpu_backward<double,int64_t>(b, n, h, w,
				vertices.sizes().size() == 2,
				perspective,
				v, ind, dcoeff, eps);
	} else {
		AT_ASSERTM(false, " type error");
	}
	return grad_coefficient;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("forward", &rasterize_forward, "Rasterize Forward");
	m.def("backward",&rasterize_backward,"Rasterize Backward");
}
