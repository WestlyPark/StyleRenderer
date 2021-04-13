#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define THREADS_MAX 512
template<typename index>
__device__ inline bool divide_loop(index &start, index &end) {
	index num = (end - start + gridDim.x - 1) / gridDim.x;
	start = start + num * blockIdx.x;
	end = (start + num < end ? start + num : end);
	num = (end - start + blockDim.x - 1) / blockDim.x;
	start = start + num * threadIdx.x;
	end = (start + num < end ? start + num : end);
	return (end > start);
}
__device__ static float atomicMax(float* address, float val) {
	int* address_as_i = (int*) address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__float_as_int(fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}
__device__ static double atomicMax(double* address, double val) {
	unsigned long long* address_as_i = (unsigned long long*) address;
	unsigned long long old = *address_as_i, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_i, assumed,
			__double_as_longlong(max(val, __longlong_as_double(assumed))));
	} while (assumed != old);
	return __longlong_as_double(old);
}
__device__ static double atomicExch(double* address, double val) {
	return __longlong_as_double(atomicExch((unsigned long long*)address,
		__double_as_longlong(val)));
}
#include "rasterize.h"
template<typename scalar, typename index>
__global__ void rasterize_kernel(index b, index nv, index nf, index h, index w,
		bool repeat_v, bool repeat_f, bool perspective,
		const scalar *v0, const index *f0, index *i, scalar *c,
		scalar *zB, scalar eps = 1e-6) {
	scalar Ainv[9], coeff[3], uv[2], det = 1;
	index start = 0, end = b * nf;
	int64_t bbox[4];
	divide_loop(start, end);
	for(index t_ = start; t_ < end; ++t_) {
		index batch = t_ / nf, t = t_ % nf;
		const scalar *v = (repeat_v || v0 == NULL ? v0 : v0 + batch*nv*3);
		const index  *f = (repeat_f || f0 == NULL ? f0 : f0 + batch*nf*3);
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
				index ind = x + w * (y + h * batch);
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
	}
	__syncthreads();
}
template<typename scalar, typename index>
bool rasterize_gpu(index b, index nv, index nf, index h, index w,
		bool repeat_v, bool repeat_f, bool perspective,
		const scalar *v, const index *f, index *i, scalar *c,
		scalar *zB, scalar eps = 1e-6) {
	index threads = (THREADS_MAX < nf ? THREADS_MAX : nf);
	rasterize_kernel<scalar,index><<<b, threads>>>(
		b, nv, nf, h, w, repeat_v, repeat_f, perspective,
		v, f, i, c, zB, eps);
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(e));
		return false;
	} else
		return true;
}
template<typename scalar, typename index>
__global__ void rasterize_kernel_backward(index b, index n, index h, index w,
                bool repeat_v, bool perspective, const scalar *v,
                const index *i, scalar *dcoeff, scalar eps = 1e-6) {
	scalar uv[2];
	index start = 0, end = b * h * w;
	divide_loop(start, end);
	for(index t = start; t < end; ++t) {
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
		uv[1] = (scalar)((t%(h*w))/w);
		barycentric_grad<scalar>(v_, uv, (scalar)h, (scalar)w,
			dcoeff + t * 27, perspective, eps);
	}
	__syncthreads();
}
template<typename scalar, typename index>
bool rasterize_gpu_backward(index b, index n, index h, index w,
		bool repeat_v, bool perspective, const scalar *v,
		const index *i, scalar *dc, scalar eps = 1e-6) {
	index threads = (THREADS_MAX < h*w ? THREADS_MAX : h*w);
	rasterize_kernel_backward<scalar,index><<<b, threads>>>(
		b, n, h, w, repeat_v, perspective,
		v, i, dc, eps);
	cudaError_t e = cudaGetLastError();
	if(e != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(e));
		return false;
	} else
		return true;
}

template __host__ bool barycentric<float>(float*,int64_t,int64_t,
	int64_t*,float*,float*,bool,float);
template __host__ bool barycentric_grad<float>(const float*,float*,
	float,float,float*,bool,float);
template __host__ bool assign_buffer<float>(const float*,
	float*,const float*,float*,float*,bool,const float*,float,float);
template bool rasterize_gpu<float,int64_t>(int64_t,int64_t,int64_t,int64_t,int64_t,bool,bool,bool,
        const float*,const int64_t*,int64_t*,float*,float*,float);
template bool rasterize_gpu_backward<float,int64_t>(int64_t,int64_t,int64_t,int64_t,
	bool,bool,const float*,const int64_t*,float*,float);

template __host__ bool barycentric<double>(double*,int64_t,int64_t,
	int64_t*,double*,double*,bool,double);
template __host__ bool barycentric_grad<double>(const double*,double*,
	double,double,double*,bool,double);
template __host__ bool assign_buffer<double>(const double*,
	double*,const double*,double*,double*,bool,const double*,double,double);
template bool rasterize_gpu<double,int64_t>(int64_t,int64_t,int64_t,int64_t,int64_t,bool,bool,bool,
        const double*,const int64_t*,int64_t*,double*,double*,double);
template bool rasterize_gpu_backward<double,int64_t>(int64_t,int64_t,int64_t,int64_t,
	bool,bool,const double*,const int64_t*,double*,double);

