
#include <vector>

#include "caffe/layers/polarize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PolarizeForward(const int n, const Dtype* in,
                                const unsigned int* mask,
                                const unsigned int threshold,
                                const Dtype zoom_up_, const Dtype zoom_down_,
                                Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = mask[index] > threshold ? (in[index] * zoom_up_)
                                         : (in[index] * zoom_down_);
    //    out[index] = in[index] * zoom[mask[index] > threshold];
  }
}

template <typename Dtype>
void PolarizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);
    //    const Dtype zoom[2] = {zoom_[0], zoom_[1]};
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    PolarizeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, mask, uint_thres_, zoom_[1], zoom_[0], top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void PolarizeBackward(const int n, const Dtype* in_diff,
                                 const unsigned int* mask,
                                 const unsigned int threshold,
                                 const Dtype zoom_up_, const Dtype zoom_down_,
                                 Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = mask[index] > threshold ? (in_diff[index] * zoom_up_)
                                              : (in_diff[index] * zoom_down_);
    //    out_diff[index] = in_diff[index] * zoom[mask[index] > threshold];
  }
}

template <typename Dtype>
void PolarizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const unsigned int* mask =
        static_cast<const unsigned int*>(rand_vec_.gpu_data());
    const int count = bottom[0]->count();
    //    const Dtype zoom[2] = {zoom_[0], zoom_[1]};
    // NOLINT_NEXT_LINE(whitespace/operators)
    PolarizeBackward<
        Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, uint_thres_, zoom_[1], zoom_[0], bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PolarizeLayer);

}  // namespace caffe
