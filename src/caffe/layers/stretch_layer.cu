#include <vector>

#include "caffe/layers/stretch_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void StretchForward(const int n, const Dtype *in, Dtype *out,
                               const int channels_total, const int width) {
  CUDA_KERNEL_LOOP(index, n) {
    int offset = (index / width) % 2;
    int count_channels = index / channels_total;
    out[(count_channels * 2 + (index + offset) % 2) * channels_total +
        index % channels_total] = in[index];
  }
}

template <typename Dtype>
void StretchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const Dtype *in_data = bottom[0]->gpu_data();
  const int count_ = bottom[0]->count();
  Dtype *out_data = top[0]->mutable_gpu_data();
  const vector<int> shape = bottom[0]->shape();
  const int channels_total = shape[2] * shape[3];
  caffe_gpu_set(count_ * 2, Dtype(0), out_data);
  StretchForward<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
      count_, in_data, out_data, channels_total, shape[3]);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void StretchBackward(const int n, const Dtype *in, Dtype *out,
                                const int channels_total, const int width) {
  CUDA_KERNEL_LOOP(index, n) {
    int offset = (index / width) % 2;
    int count_channels = index / channels_total;
    out[index] =
        in[(count_channels * 2 + (index + offset) % 2) * channels_total +
               index % channels_total];
  }
}

template <typename Dtype>
void StretchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *in_diff = top[0]->gpu_diff();
    const int count_ = bottom[0]->count();
    Dtype *out_diff = bottom[0]->mutable_gpu_diff();
    const vector<int> shape = bottom[0]->shape();
    const int channels_total = shape[2] * shape[3];
    StretchBackward<
        Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
        count_, in_diff, out_diff, channels_total, shape[3]);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(StretchLayer);

}  // namespace caffe
