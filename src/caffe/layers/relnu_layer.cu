#include <algorithm>
#include <vector>

#include "caffe/layers/relnu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLnUForward(const int n, const Dtype* in, Dtype* out,
    Dtype alpha, Dtype beta)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        out[index] = in[index] > Dtype(0) ? (beta * std::log(alpha * in[index] + 1)) : Dtype(0);
    }
}

template <typename Dtype>
void ReLnULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_.relnu_param().alpha();
    Dtype beta = this->layer_param_.relnu_param().beta();
    ReLnUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_data, alpha, beta);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ReLnUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype alpha, Dtype beta)
{
    CUDA_KERNEL_LOOP(index, n)
    {
        out_diff[index] = in_diff[index] * (in_data[index] > Dtype(0) ? (beta / (in_data[index] + Dtype(1) / alpha)) : Dtype(0));
    }
}

template <typename Dtype>
void ReLnULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0]) {
        const Dtype* bottom_data = bottom[0]->gpu_data();
        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
        const int count = bottom[0]->count();
        Dtype alpha = this->layer_param_.relnu_param().alpha();
        Dtype beta = this->layer_param_.relnu_param().beta();
        // NOLINT_NEXT_LINE(whitespace/operators)
        ReLnUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, bottom_data, bottom_diff, alpha, beta);
        CUDA_POST_KERNEL_CHECK;
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReLnULayer);

} // namespace caffe
