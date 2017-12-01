#include <vector>

#include "caffe/layers/stretch_layer.hpp"

namespace caffe {

template <typename Dtype>
void StretchLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  const vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape[0], shape[1] * 2, shape[2], shape[3]);
}

template <typename Dtype>
void StretchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                      const vector<Blob<Dtype> *> &top) {
  const int count_ = bottom[0]->count();
  const Dtype *in_data = bottom[0]->cpu_data();
  Dtype *out_data = top[0]->mutable_cpu_data();
  caffe_set(count_ * 2, Dtype(0), out_data);
  const vector<int> shape = bottom[0]->shape();
  const int channels_total = shape[2] * shape[3];
  for (int i = 0; i < count_; ++i) {
    int offset = (i / shape[3]) % 2;
    int count_channels = i / channels_total;
    out_data[(count_channels * 2 + (i + offset) % 2) * channels_total +
             i % channels_total] = in_data[i];
  }
}

template <typename Dtype>
void StretchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                       const vector<bool> &propagate_down,
                                       const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const int count_ = bottom[0]->count();
    const Dtype *in_diff = top[0]->cpu_diff();
    Dtype *out_diff = bottom[0]->mutable_cpu_diff();
    const vector<int> shape = bottom[0]->shape();
    const int channels_total = shape[2] * shape[3];
    for (int i = 0; i < count_; ++i) {
      int offset = (i / shape[3]) % 2;
      int count_channels = i / channels_total;
      out_diff[i] =
          in_diff[(count_channels * 2 + (i + offset) % 2) * channels_total +
                      i % channels_total];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(StretchLayer);
#endif

INSTANTIATE_CLASS(StretchLayer);

REGISTER_LAYER_CLASS(Stretch);

}  // namespace caffe
