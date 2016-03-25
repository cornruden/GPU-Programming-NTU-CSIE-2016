#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

struct not_one{
    __host__ __device__
    bool operator()(const int x)
    {
        return (x!=1);
    }
};

struct not_new_line{
    __host__ __device__
        int operator()(const char x){
            return (x != '\n') ? 1:0;
        }
};

void CountPosition(const char *text, int *pos, int text_size)
{
    int *buffer;
    cudaMalloc(&buffer, sizeof(int)*text_size);
    thrust::device_ptr<const char> text_d(text), text_d_end(text+text_size);
    thrust::device_ptr<int> pos_d(pos), tmp(buffer),tmp_end(buffer+text_size);;
    thrust::transform(text_d,text_d_end, tmp, not_new_line());
    thrust::inclusive_scan_by_key(tmp, tmp_end, tmp, pos_d);
    cudaFree(buffer);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO
    thrust::sequence(flag_d,cumsum_d);
    cumsum_d = thrust::remove_if(flag_d,cumsum_d, pos_d, not_one());
    thrust::copy(flag_d, cumsum_d, head_d);
    nhead = cumsum_d - flag_d;
	cudaFree(buffer);

    return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
