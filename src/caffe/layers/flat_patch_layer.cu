//
// Created by admins on 17-10-8.
//
#include <vector>

#include "caffe/layers/flat_patch_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void FlatPatchForeWard(const int n, const Dtype *in, Dtype *out,
                                  const int num_total, const int channels_total,
                                  const int width, const int size_h,
                                  const int size_w) {
  const int channels_total_ = channels_total / size_w / size_h;
  const int width_out = width / size_w;
  CUDA_KERNEL_LOOP(index, n) {
    int count_num = index / num_total;
    int count_channels = (index % num_total) / channels_total;
    int height_data = (index % channels_total) / width;
    int width_data = index % width;
    int seat_height = height_data / size_h;
    int count_height = height_data % size_h;
    int seat_width = width_data / size_w;
    int count_width = width_data % size_w;
    out[count_num * num_total +
        ((count_channels * size_h + count_height) * size_w + count_width) *
            channels_total_ +
        seat_height * width_out + seat_width] = in[index];
  }
}

template <typename Dtype>
void FlatPatchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const vector<int> shape = bottom[0]->shape();
  const int count_ = bottom[0]->count();
  const Dtype *in_data = bottom[0]->gpu_data();
  Dtype *out_data = top[0]->mutable_gpu_data();
  FlatPatchForeWard<
      Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
      count_, in_data, out_data, shape[1] * shape[2] * shape[3],
      shape[2] * shape[3], shape[3], size_h, size_w);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void FlatPatchBackWard(const int n, const Dtype *in, Dtype *out,
                                  const int num_total, const int channels_total,
                                  const int width, const int size_h,
                                  const int size_w) {
  const int channels_total_ = channels_total / size_w / size_h;
  const int width_out = width / size_w;
  CUDA_KERNEL_LOOP(index, n) {
    int count_num = index / num_total;
    int count_channels = (index % num_total) / channels_total;
    int height_data = (index % channels_total) / width;
    int width_data = index % width;
    int seat_height = height_data / size_h;
    int count_height = height_data % size_h;
    int seat_width = width_data / size_w;
    int count_width = width_data % size_w;
    out[index] =
        in[count_num * num_total +
           ((count_channels * size_h + count_height) * size_w + count_width) *
               channels_total_ +
           seat_height * width_out + seat_width];
  }
}

template <typename Dtype>
void FlatPatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const vector<int> shape = bottom[0]->shape();
    const int count_ = bottom[0]->count();
    const Dtype *in_diff = top[0]->gpu_diff();
    Dtype *out_diff = bottom[0]->mutable_gpu_diff();
    FlatPatchBackWard<
        Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
        count_, in_diff, out_diff, shape[1] * shape[2] * shape[3],
        shape[2] * shape[3], shape[3], size_h, size_w);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(FlatPatchLayer);

}  // namespace caffe
