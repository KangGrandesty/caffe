#include "caffe/layers/shift_relu_layer.hpp"
#include <algorithm>
#include <vector>

namespace caffe {

template <typename Dtype>
void SReLULayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top) {
  top[0]->ReshapeLike(*bottom[0]);
  const vector<int> shape = bottom[0]->shape();
  const bool map = this->layer_param_.shift_relu_param().map();
  if (map) {
    offset = shape[2] * shape[3];
    shift_.Reshape(shape[0], shape[1], 1, 1);
  } else {
    offset = shape[1] * shape[2] * shape[3];
    shift_.Reshape(shape[0], 1, 1, 1);
  }
  bias_.ReshapeLike(shift_);
  mask_.ReshapeLike(shift_);
}

template <typename Dtype>
void SReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const Dtype negative_slope =
      this->layer_param_.shift_relu_param().negative_slope();
  const Dtype shift = this->layer_param_.shift_relu_param().shift();
  const Dtype range = this->layer_param_.shift_relu_param().range();
  Dtype *shifts = shift_.mutable_cpu_data();
  if (this->phase_ == TRAIN && range > Dtype(0)) {
    Dtype *mask = mask_.mutable_cpu_data();
    for (int i = 0; i < shift_.count(); ++i) {
      mask[i] = caffe_cpu_asum(offset, bottom_data + i * offset) / offset;
    }
    caffe_rng_uniform(shift_.count(), -range, range, bias_.mutable_cpu_data());
    caffe_mul(shift_.count(), bias_.cpu_data(), mask_.cpu_data(), shifts);
    caffe_add_scalar(shift_.count(), shift, shifts);
  } else {
    caffe_set(shift_.count(), shift, shifts);
  }
  const bool type = this->layer_param_.shift_relu_param().type();
  if (type) {
    for (int i = 0; i < bottom[0]->count(); ++i) {
      const Dtype shift = shifts[i / offset];
      const Dtype data = bottom_data[i] + type * shift;
      top_data[i] = data > Dtype(0) ? data : (data * negative_slope);
    }
  } else {
    for (int i = 0; i < bottom[0]->count(); ++i) {
      const Dtype shift = shifts[i / offset];
      top_data[i] = bottom_data[i] > shift
                        ? bottom_data[i]
                        : ((bottom_data[i] - shift) * negative_slope + shift);
    }
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                     const vector<bool> &propagate_down,
                                     const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const Dtype *bottom_data = bottom[0]->cpu_data();
    const Dtype *top_diff = top[0]->cpu_diff();
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype negative_slope =
        this->layer_param_.shift_relu_param().negative_slope();
    Dtype *shifts = shift_.mutable_cpu_data();
    const bool type = this->layer_param_.shift_relu_param().type();
    if (type) {
      for (int i = 0; i < bottom[0]->count(); ++i) {
        const Dtype shift = shifts[i / offset];
        const Dtype data = bottom_data[i] + type * shift;
        bottom_diff[i] =
            top_diff[i] * (data > Dtype(0) ? Dtype(1) : negative_slope);
      }
    } else {
      for (int i = 0; i < bottom[0]->count(); ++i) {
        bottom_diff[i] =
            top_diff[i] *
            (bottom_data[i] > shifts[i / offset] ? Dtype(1) : negative_slope);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SReLULayer);
#endif

INSTANTIATE_CLASS(SReLULayer);

REGISTER_LAYER_CLASS(SReLU);

}  // namespace caffe
