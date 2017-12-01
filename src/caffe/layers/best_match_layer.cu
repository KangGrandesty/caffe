#include <numeric>
#include <vector>

#include "caffe/layers/best_match_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BestMatchForward(const int n, const Dtype *in1,
                                 const Dtype *in2, const Dtype *match,
                                 Dtype *out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = ((in1[index] + in2[index] - Dtype(2) * match[index]) *
                  (in1[index] - in2[index])) > Dtype(0)
                     ? in2[index]
                     : in1[index];
  }
}

template <typename Dtype>
void BestMatchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *cand_data1 = bottom[0]->gpu_data();
  const Dtype *cand_data2 = bottom[1]->gpu_data();
  const Dtype *match_data = bottom[2]->gpu_data();
  const int count = bottom[0]->count();
  Dtype *top_data = top[0]->mutable_gpu_data();
  if (this->phase_ == TRAIN) {
    BestMatchForward<
        Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, cand_data1, cand_data2, match_data, top_data);
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, cand_data1, top_data);
    caffe_gpu_axpy(count, Dtype(1), cand_data2, top_data);
  }
}

template <typename Dtype>
__global__ void BestMatchBackward(const int n, const Dtype *in1,
                                  const Dtype *in2, const Dtype *data,
                                  const Dtype *diff, Dtype *out1, Dtype *out2) {
  CUDA_KERNEL_LOOP(index, n) {
    out1[index] = (in1[index] == data[index]) * diff[index];
    out2[index] = (in2[index] == data[index]) * diff[index];
  }
}

template <typename Dtype>
void BestMatchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  const Dtype *cand_data1 = bottom[0]->gpu_data();
  const Dtype *cand_data2 = bottom[1]->gpu_data();
  const Dtype *top_diff = top[0]->gpu_diff();
  const Dtype *top_data = top[0]->gpu_data();
  const int count = bottom[0]->count();
  Dtype *cand_diff1 = bottom[0]->mutable_gpu_diff();
  Dtype *cand_diff2 = bottom[1]->mutable_gpu_diff();
  Dtype *match_diff2 = bottom[2]->mutable_gpu_diff();
  caffe_gpu_set(count, Dtype(0), match_diff2);
  BestMatchBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, cand_data1, cand_data2, top_data, top_diff, cand_diff1,
      cand_diff2);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(BestMatchLayer);

}  // namespace caffe
