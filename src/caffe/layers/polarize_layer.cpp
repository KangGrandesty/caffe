#include <vector>

#include "caffe/layers/polarize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PolarizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.polarize_param().ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  zoom_.clear();
  zoom_.push_back(this->layer_param_.polarize_param().zoom());
  zoom_.push_back((1. - threshold_ * zoom_[0]) / (1. - threshold_));
}

template <typename Dtype>
void PolarizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  rand_vec_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void PolarizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      top_data[i] = bottom_data[i] * zoom_[mask[i]];
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void PolarizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const unsigned int* mask = rand_vec_.cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * zoom_[mask[i]];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PolarizeLayer);
#endif

INSTANTIATE_CLASS(PolarizeLayer);
REGISTER_LAYER_CLASS(Polarize);

}  // namespace caffe
