#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float kernels[8192];

__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    __shared__ float inputImage[72*72];
    /*
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */


// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define sharedIndex(i3, i2, i1) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1)]
#define x4d(i1, i0) inputImage[(i1) * 72 + i0]
#define k4d(i3, i2, i1, i0) kernels[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int numThreads = blockDim.x;
    const int imageSize = H * W;
    const int outputSize = H_out * W_out;

    int imageNumber = blockIdx.x;
    const int kernelNumber = blockIdx.y;
    const int numBlocks = gridDim.x

    // if the block is less than the batch size
    while (imageNumber < B && kernelNumber < M) { // for each image in the batch
        int index = threadIdx.x;
        while (index < outputSize) {
            int row = index / W_out;
            int col = index % W_out;
            y4d(imageNumber, kernelNumber, row, col) = 0;  // sum over all input feature maps
            for (int channelNumber = 0; channelNumber < C; channelNumber++) {

                // read one channel into shared memory
                int shared_index = threadIdx.x;
                while(shared_index < imageSize) {
                    int shared_row = shared_index / W;
                    int shared_col = shared_index % W;
                    x4d(shared_row, shared_col) = sharedIndex(imageNumber, channelNumber, shared_index);
                    shared_index += numThreads;
                }
                __syncthreads();

                for (int p = 0; p < K; p++) // KxK filter
                    for (int q = 0; q < K; q++)
                        y4d(imageNumber, kernelNumber, row, col) += x4d(row + p, col + q) * k4d(kernelNumber, channelNumber, p, q);                
            }
            index += numThreads;
        }
        imageNumber += numBlocks
    }

#undef y4d
#undef x4d
#undef k4d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   We only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // // Use mxnet's CHECK_EQ to do assertions.
    // CHECK_EQ(0, 1) 

    const int B = x.shape_[0];
    const int M = y.shape_[1]; // num_filter
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    dim3 gridDim((1024, M);
    dim3 blockDim(1024);

    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    // need to find count
    size_t w_size = M * C * K * K;
    cudaMemcpyToSymbol(kernels, w.dptr_, w_size * sizeof(float));

    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B, M, C, H, W, K);
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());
}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    assert(0 && "No forward implementation for other datatypes needed");
}
}
}

#endif
