#include <algorithm>
#include <vector>

#include "caffe/layers/shift_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SReLUShift(Dtype* shifts, const Dtype* data, const int n,
                           const int offset) {
  CUDA_KERNEL_LOOP(index, n) {
    shifts[index / offset] +=
        data[index] > Dtype(0) ? data[index] : -data[index];
  }
}

template <typename Dtype>
__global__ void SReLUForward(const int n, const int offset, const Dtype* in,
                             Dtype* out, const Dtype negative_slope,
                             const Dtype* shifts) {
  CUDA_KERNEL_LOOP(index, n) {
    const Dtype shift = shifts[index / offset];
    const Dtype data = in[index] - shift;
    out[index] = data > Dtype(0) ? data : (data * negative_slope);
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype negative_slope =
      this->layer_param_.shift_relu_param().negative_slope();
  const Dtype shift = this->layer_param_.shift_relu_param().shift();
  const Dtype range = this->layer_param_.shift_relu_param().range();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN && range > Dtype(0)) {
    SReLUShift<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        mask_.mutable_gpu_data(), bottom_data, count, offset);
    caffe_gpu_scal(shift_.count(), Dtype(1) / offset, mask_.mutable_gpu_data());
    caffe_gpu_rng_uniform(shift_.count(), -range, range,
                          bias_.mutable_gpu_data());
    caffe_gpu_mul(shift_.count(), bias_.gpu_data(), mask_.gpu_data(),
                  shift_.mutable_gpu_data());
    caffe_gpu_add_scalar(shift_.count(), shift, shift_.mutable_gpu_data());
  } else {
    caffe_gpu_set(shift_.count(), shift, shift_.mutable_gpu_data());
  }
  SReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, offset, bottom_data, top_data, negative_slope, shift_.gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void SReLUBackward(const int n, const int offset,
                              const Dtype* in_diff, const Dtype* in_data,
                              Dtype* out_diff, const Dtype negative_slope,
                              const Dtype* shifts) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] =
        in_diff[index] *
        (in_data[index] > shifts[index / offset] ? Dtype(1) : negative_slope);
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                     const vector<bool>& propagate_down,
                                     const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype negative_slope =
        this->layer_param_.shift_relu_param().negative_slope();
    const int count = bottom[0]->count();
    const Dtype* shifts = shift_.mutable_gpu_data();
    SReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, offset, top_diff, bottom_data, bottom_diff, negative_slope,
        shifts);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SReLULayer);

}  // namespace caffe
