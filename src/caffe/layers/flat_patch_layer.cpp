//
// Created by admins on 17-10-8.
//
#include <vector>

#include "caffe/layers/flat_patch_layer.hpp"

namespace caffe {

template <typename Dtype>
void FlatPatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom.size(), 1);
  CHECK_EQ(top.size(), 1);
  int size = this->layer_param().flat_patch_param().size();
  CHECK_GE(size, 1);
  const vector<int> shape = bottom[0]->shape();
  size_h = this->layer_param().flat_patch_param().size_h();
  if (size_h == 0) {
    size_h = size;
  }
  CHECK_EQ(0, shape[2] % size_h);
  size_w = this->layer_param().flat_patch_param().size_w();
  if (size_w == 0) {
    size_w = size;
  }
  CHECK_EQ(0, shape[3] % size_w);
}

template <typename Dtype>
void FlatPatchLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  const vector<int> shape = bottom[0]->shape();
  top[0]->Reshape(shape[0], shape[1] * size_h * size_w, shape[2] / size_h,
                  shape[3] / size_w);
}

template <typename Dtype>
void FlatPatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const vector<int> shape = bottom[0]->shape();
  const int num_total = shape[1] * shape[2] * shape[3];
  const int channels_total = shape[2] * shape[3];
  const int count_ = bottom[0]->count();
  const Dtype *in_data = bottom[0]->cpu_data();
  Dtype *out_data = top[0]->mutable_cpu_data();
  const int channels_total_ = channels_total / size_h / size_w;
  const int width_out = shape[3] / size_w;
  for (int i = 0; i < count_; ++i) {
    int count_num = i / num_total;
    int count_channels = (i % num_total) / channels_total;
    int height_data = (i % channels_total) / shape[3];
    int width_data = i % shape[3];
    int seat_height = height_data / size_h;
    int count_height = height_data % size_h;
    int seat_width = width_data / size_w;
    int count_width = width_data % size_w;
    out_data[count_num * num_total +
             ((count_channels * size_h + count_height) * size_w + count_width) *
                 channels_total_ +
             seat_height * width_out + seat_width] = in_data[i];
  }
}

template <typename Dtype>
void FlatPatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0]) {
    const vector<int> shape = bottom[0]->shape();
    const int num_total = shape[1] * shape[2] * shape[3];
    const int channels_total = shape[2] * shape[3];
    const Dtype *in_diff = top[0]->cpu_diff();
    const int count_ = bottom[0]->count();
    Dtype *out_diff = bottom[0]->mutable_cpu_diff();
    const int channels_total_ = channels_total / size_h / size_w;
    const int width_out = shape[3] / size_w;
    for (int i = 0; i < count_; ++i) {
      int count_num = i / num_total;
      int count_channels = (i % num_total) / channels_total;
      int height_data = (i % channels_total) / shape[3];
      int width_data = i % shape[3];
      int seat_height = height_data / size_h;
      int count_height = height_data % size_h;
      int seat_width = width_data / size_w;
      int count_width = width_data % size_w;
      out_diff[i] = in_diff[count_num * num_total +
                            ((count_channels * size_h + count_height) * size_w +
                             count_width) *
                                channels_total_ +
                            seat_height * width_out + seat_width];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(FlatPatchLayer);
#endif

INSTANTIATE_CLASS(FlatPatchLayer);

REGISTER_LAYER_CLASS(FlatPatch);

}  // namespace caffe
