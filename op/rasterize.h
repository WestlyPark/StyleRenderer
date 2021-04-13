#ifndef _RASTERIZE_H_
#define _RASTERIZE_H_
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
template<typename scalar>
__device__ __host__ bool barycentric(scalar v[9], int64_t w, int64_t h, int64_t bbox[4],
		scalar Ainv[9], scalar *det_ = NULL, bool perspective = false, scalar eps = 1e-6) {
	scalar umin = (scalar)w, vmin = (scalar)h, umax = 0, vmax = 0;
	if(v != NULL)
		for(unsigned char i = 0; i < 3; ++i) {
			if(perspective) {
				if(v[3*i+2] >= -eps)
					return false;
				v[3*i]  /=-v[3*i+2];
				v[3*i+1]/=-v[3*i+2];
			}
			v[3*i]  = (1 + v[3*i])  *(scalar)w / 2 - .5;
			v[3*i+1]= (1 - v[3*i+1])*(scalar)h / 2 - .5;
			if(i == 0) {
				umax = umin = v[3*i];
				vmax = vmin = v[3*i+1];
			} else {
				if(umin > v[3*i])
					umin = v[3*i];
				else if(umax < v[3*i])
					umax = v[3*i];
				if(vmin > v[3*i+1])
					vmin = v[3*i+1];
				else if(vmax < v[3*i+1])
					vmax = v[3*i+1];
			}
		}
	else
		return false;
	if(bbox != NULL) {
		bbox[0] = (int64_t) ceil(umin);
		bbox[1] = (int64_t)floor(umax);
		bbox[2] = (int64_t) ceil(vmin);
		bbox[3] = (int64_t)floor(vmax);
		bbox[0] = bbox[0] < 0  ? 0  : bbox[0];
		bbox[1] = bbox[1] > w-1? w-1: bbox[1];
		bbox[2] = bbox[2] < 0  ? 0  : bbox[2];
		bbox[3] = bbox[3] > h-1? h-1: bbox[3];
		if(bbox[1] < bbox[0] || bbox[3] < bbox[2])
			return false;
	}
	if(Ainv != NULL) {
		Ainv[0] = v[3]*v[7] - v[4]*v[6];
		Ainv[1] = v[1]*v[6] - v[0]*v[7];
		Ainv[2] = v[0]*v[4] - v[1]*v[3];
		scalar det = Ainv[0] + Ainv[1] + Ainv[2];
		if(det > eps) return false;
		Ainv[3] = v[4] - v[7];
		Ainv[4] = v[7] - v[1];
		Ainv[5] = v[1] - v[4];
		Ainv[6] = v[6] - v[3];
		Ainv[7] = v[0] - v[6];
		Ainv[8] = v[3] - v[0];
		if(det_ != NULL)
			if(det < 0) {
				for(unsigned char i = 0; i < 9; ++i)
					Ainv[i] = -Ainv[i];
				*det_ =-det;
			} else	*det_ = det;
		else if(det < -eps)
			for(unsigned char i = 0; i < 9; ++i)
				Ainv[i] /= det;
		return true;
	} else
		return false;
}
template<typename scalar>
__device__ __host__ bool normalize_coeff(scalar coeff[3], const scalar v[9],
		const scalar uv[2], const scalar Ainv[9], scalar det = 1, scalar eps = 1e-6) {
	if(coeff == NULL || coeff[0] < -eps || coeff[1] < -eps || coeff[2] < -eps)
		return false;
	if(det > eps) {
		det = coeff[0] + coeff[1] + coeff[2];
		coeff[0] /= det;
		coeff[1] /= det;
		coeff[2] /= det;
		return true;
	} else if(v != NULL && uv != NULL) {
		scalar Ainv_[9];
		if(Ainv == NULL) {
			Ainv_[3] = v[4] - v[7];
			Ainv_[4] = v[7] - v[1];
			Ainv_[5] = v[1] - v[4];
			Ainv_[6] = v[6] - v[3];
			Ainv_[7] = v[0] - v[6];
			Ainv_[8] = v[3] - v[0];
			Ainv = Ainv_;
		}
		scalar l[] = {
			Ainv[3]*Ainv[3] + Ainv[6]*Ainv[6],
			Ainv[4]*Ainv[4] + Ainv[7]*Ainv[7],
			Ainv[5]*Ainv[5] + Ainv[8]*Ainv[8]};
		unsigned char i = (l[0] > l[1] ? 0 : 1);
		i = (l[i] > l[2] ? i : 2);
		unsigned char j = (i+1)%3, k = (j+1)%3;
		if(l[i] > eps) { // Triangle degenerate to a segment
			l[j] =-(uv[0]-v[3*k])*Ainv[6+i] + (uv[1]-v[3*k+1])*Ainv[3+i];
			l[k] = (uv[0]-v[3*j])*Ainv[6+i] - (uv[1]-v[3*j+1])*Ainv[3+i];
			l[i] = l[j] + l[k];
			coeff[i] = 0;
			coeff[j] = l[j] / l[i];
			coeff[k] = l[k] / l[i];
			return coeff[j] >= -eps && coeff[k] >= -eps;
		} else { // Triangle degenerate to a point
			coeff[j] = coeff[k] = 0;
			coeff[i] = 1;
			l[0] = uv[0] - v[3*i];
			l[1] = uv[1] - v[3*i+1];
			l[2] = l[0]*l[0] + l[1]*l[1];
			return l[2] < eps;
		}

	} else
		return false;
}
template<typename scalar>
__device__ __host__ bool assign_buffer(const scalar v[9], scalar coeff[3], const scalar uv[2],
		scalar *zB, scalar c[3], bool perspective,
		const scalar Ainv[9], scalar det = 1, scalar eps = 1e-6) {
	if(!normalize_coeff<scalar>(coeff, v, uv, Ainv, det, eps))
		return false;
	scalar z = 0;
	if(perspective) {
		coeff[0] /= v[2];
		coeff[1] /= v[5];
		coeff[2] /= v[8];
		z = coeff[0] + coeff[1] + coeff[2];
		if(z >= -eps) return false;
		coeff[0] *= z;
		coeff[1] *= z;
		coeff[2] *= z;
	} else
		z = coeff[0]*v[2] + coeff[1]*v[5] + coeff[2]*v[8];
#ifdef __CUDA_ARCH__
	if(zB != NULL)
		atomicMax(zB, z);
	if(zB == NULL || *zB == z) {
		if(c != NULL) {
			atomicExch(c,  coeff[0]);
			atomicExch(c+1,coeff[1]);
			atomicExch(c+2,coeff[2]);
		}
		return true;
	} else
		return false;
#else
	if(zB == NULL || *zB < z) {
		if(zB != NULL) *zB = z;
		if(c != NULL) {
			c[0] = coeff[0];
			c[1] = coeff[1];
			c[2] = coeff[2];
		}
		return true;
	} else
		return false;
#endif
}
template<typename scalar>
__device__ __host__ bool barycentric_grad(const scalar v[9], scalar uv[2],
		scalar w, scalar h, scalar dcoeff[27], bool perspective, scalar eps = 1e-6) {
	uv[0] = (uv[0] * 2 - w + 1) / w;
	uv[1] = (uv[1] *-2 + h - 1) / h;
	scalar Ainv[9]={v[3]*v[7] - v[4]*v[6],
			v[1]*v[6] - v[0]*v[7],
			v[0]*v[4] - v[1]*v[3]},
		coeff[3], det = 1;
	if(perspective) {
		if(v[2] >= -eps || v[5] >= -eps || v[8] >= -eps)
			return false;
		det = Ainv[0]*v[2] + Ainv[1]*v[5] + Ainv[2]*v[8];
		if(det >= -eps & det <= eps)
			return false;
		Ainv[0] = -Ainv[0];
		Ainv[1] = -Ainv[1];
		Ainv[2] = -Ainv[2];
		Ainv[3] = v[4]*v[8] - v[5]*v[7];
		Ainv[4] = v[2]*v[7] - v[1]*v[8];
		Ainv[5] = v[1]*v[5] - v[2]*v[4];
		Ainv[6] = v[5]*v[6] - v[3]*v[8];
		Ainv[7] = v[0]*v[8] - v[2]*v[6];
		Ainv[8] = v[2]*v[3] - v[0]*v[5];
	} else {
		det = Ainv[0] + Ainv[1] + Ainv[2];
		Ainv[3] = v[4] - v[7];
		Ainv[4] = v[7] - v[1];
		Ainv[5] = v[1] - v[4];
		Ainv[6] = v[6] - v[3];
		Ainv[7] = v[0] - v[6];
		Ainv[8] = v[3] - v[0];
	}
	if(det < -eps || det > eps) {
		for(unsigned char i = 0; i < 9; ++i)
			Ainv[i] /= det;
		scalar coeff[3] = {
			Ainv[0] + Ainv[3]*uv[0] + Ainv[6]*uv[1],
			Ainv[1] + Ainv[4]*uv[0] + Ainv[7]*uv[1],
			Ainv[2] + Ainv[5]*uv[0] + Ainv[8]*uv[1]};
		if(dcoeff != NULL) {
			for(unsigned char l = 0; l < 27; ++l) {
				unsigned char i = l/9, j = (l+1)%3, k = (l/3)%3;
				dcoeff[l] = -coeff[k]*Ainv[i+j*3];
			}
			if(perspective) {
				det = coeff[0] + coeff[1] + coeff[2];
				for(unsigned char l = 0; l < 9; ++l) {
					scalar dsum = dcoeff[l] + dcoeff[l+9] + dcoeff[l+18];
					for(unsigned char i = 0; i < 3; ++i)
						dcoeff[l+i*9] = (l % 3 == 2 ?
							(-dcoeff[l+i*9]-coeff[i]*dsum/det)/det :
							( dcoeff[l+i*9]-coeff[i]*dsum/det)/det);
				}
			} else
				for(unsigned char l = 0; l < 9; ++l)
					dcoeff[2+l*3] = 0;
		}
		return true;
	} else	return false;
}
#endif
