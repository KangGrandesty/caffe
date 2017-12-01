#include <algorithm>
#include <vector>

#include "caffe/layers/pow_reu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PowReUForward(const int n, const Dtype* in,
    Dtype* out, Dtype powers) {
    CUDA_KERNEL_LOOP(index, n) {
        out[index] = in[index] > Dtype(0) ? (std::pow(in[index] + Dtype(1), powers) - Dtype(1)) : Dtype(0);
    }
}

template <typename Dtype>
void PowReULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    Dtype powers = this->layer_param_.pow_reu_param().powers();
    PowReUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, powers);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void PowReUBackward(const int n, const Dtype* in_diff, const Dtype* in_data,
    Dtype* out_diff, const Dtype* out_data, Dtype powers) {
    CUDA_KERNEL_LOOP(index, n) {
        out_diff[index] = in_diff[index] * (in_data[index] > Dtype(0) ? (powers * (out_data[index] + 1) / (in_data[index] + 1)) : Dtype(0));
    }
}

template <typename Dtype>
void PowReULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0]) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_data = top[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int count = bottom[0]->count();
        Dtype powers = this->layer_param_.pow_reu_param().powers();
        // NOLINT_NEXT_LINE(whitespace/operators)
        PowReUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, bottom_data, bottom_diff, top_data, powers);
        CUDA_POST_KERNEL_CHECK;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(PowReULayer);

} // namespace caffe
