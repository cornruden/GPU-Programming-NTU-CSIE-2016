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

__global__ void AlgorithmReduce(const int *input,unsigned int *output, const int input_size)
{
    const int tid = threadIdx.x;

    //Halve the number of blocks, tricky(!?)
    const int input_id = threadIdx.x + (blockDim.x*2)*blockIdx.x;
    __shared__ unsigned int local_sum[1024];
    local_sum[tid] = 0;

    //divergence!
    if( input_id  < input_size )
        local_sum[tid] = input[input_id];
    if( (input_id+blockDim.x) < input_size )
        local_sum[tid] += input[input_id+blockDim.x];

    __syncthreads();

    for(int i=blockDim.x>>1; i > 0; i = i>>1)
    {
        //divergence! use recusive(?)
        if( tid < i )
            local_sum[tid] += local_sum[tid+i];
        __syncthreads();
    }

    output[blockIdx.x] = local_sum[0];
}


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
    int numofblocks;
    numofblocks = (((text_size+1023)/1024)+1)/2;
    unsigned int *buffer;
    unsigned int *result;
    cudaMalloc(&buffer ,sizeof(unsigned int)*numofblocks);
    printf("For Part3: Use pos array implements parallel reduction\n");

    //Implement reduction.
    AlgorithmReduce<<<numofblocks,1024>>>(pos,buffer,text_size);
    text_size = numofblocks;

    //Dynamic 
    while( text_size > 1024 ){
        cudaMemcpy(pos, buffer ,sizeof(unsigned int)*text_size, cudaMemcpyDeviceToDevice);
        numofblocks = (((text_size+1023)/1024)+1)/2;
        AlgorithmReduce<<<numofblocks,1024>>>(pos,buffer,text_size);
        text_size = numofblocks;
    }

    result = (unsigned int*)malloc( sizeof(unsigned int)*text_size );
    cudaMemcpy(result, buffer, sizeof(unsigned int)*text_size, cudaMemcpyDeviceToHost);
    cudaFree(buffer);

    while( --text_size > 0  )
        result[0] += result[text_size];

    printf("total position  sum = %u\n",result[0]);
    free(result);

    /*
    unsigned int *ubuffer;
    cudaMalloc(&ubuffer, sizeof(unsigned int)*text_size);
    result = (unsigned int*)malloc( sizeof(unsigned int)*text_size );
    thrust::device_ptr<int> pos_d(pos), pos_d_end(pos+text_size);
    thrust::device_ptr<unsigned int> res_d(ubuffer);
    thrust::inclusive_scan( pos_d, pos_d_end, res_d);

    cudaMemcpy(result, ubuffer, sizeof(unsigned int)*text_size, cudaMemcpyDeviceToHost);
    printf("now total sum = %u\n",result[text_size-1]);
    free(result);
    */
}
