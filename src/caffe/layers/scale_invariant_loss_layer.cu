//  Create on: 2016/10/24 ShanghaiTech
//  Author:    Yingying Zhang

#include <vector>

#include "caffe/layers/scale_invariant_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScaleInvariantLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // 1. sum(d_i ^ 2) / 2n
  int count = bottom[0]->count();
  caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
                diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  // 2. lambda  * (sum(d_i) ^ 2 / 2n^2
  caffe_gpu_set(count, Dtype(1), ones_.mutable_gpu_data());
  caffe_gpu_dot(count, diff_.gpu_data(), ones_.gpu_data(), &sum_di_);
  loss -= lambda_ * sum_di_ * sum_di_ / bottom[0]->num() / bottom[0]->num() /
          Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ScaleInvariantLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      // 1. gradient for sum(d_i ^ 2) / 2n
      const Dtype sign = (i == 0) ? 1 : -1;
      Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(bottom[i]->count(), alpha, diff_.gpu_data(), Dtype(0),
                      bottom[i]->mutable_gpu_diff());
      // 2. graidient for lambda  * (sum(d_i) ^ 2 / 2n^2
      alpha = -sign * top[0]->cpu_diff()[0] * sum_di_ * lambda_ /
              bottom[i]->num() / bottom[i]->num();
      caffe_gpu_axpby(bottom[i]->count(), alpha, ones_.gpu_data(),
                      Dtype(1),  // accumulate gradient
                      bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ScaleInvariantLossLayer);

}  // namespace caffe
