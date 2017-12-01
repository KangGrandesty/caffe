#include <numeric>
#include <vector>

#include "caffe/layers/best_match_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BestMatchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                       const vector<Blob<Dtype> *> &top) {
  CHECK_EQ(bottom.size(), 3U) << "bottom length is not correct!";
  CHECK_EQ(top.size(), 1U) << "top length is not correct!";
}

template <typename Dtype>
void BestMatchLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                    const vector<Blob<Dtype> *> &top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void BestMatchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  const Dtype *cand_data1 = bottom[0]->cpu_data();
  const Dtype *cand_data2 = bottom[1]->cpu_data();
  const int count = bottom[0]->count();
  Dtype *top_data = top[0]->mutable_cpu_data();
  if (this->phase_ == TRAIN) {
    const Dtype *match_data = bottom[2]->cpu_data();
    for (int i = 0; i < count; ++i) {
      top_data[i] =
          ((cand_data1[i] + cand_data2[i] - Dtype(2) * match_data[i]) *
           (cand_data1[i] - cand_data2[i])) > Dtype(0)
              ? cand_data2[i]
              : cand_data1[i];
    }
  } else {
    caffe_copy(count, cand_data1, top_data);
    caffe_axpy(count, Dtype(1), cand_data2, top_data);
  }
}

template <typename Dtype>
void BestMatchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  const Dtype *cand_data1 = bottom[0]->cpu_data();
  const Dtype *cand_data2 = bottom[1]->cpu_data();
  const Dtype *match_data = bottom[2]->cpu_data();
  const Dtype *top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  Dtype *cand_diff1 = bottom[0]->mutable_cpu_diff();
  Dtype *cand_diff2 = bottom[1]->mutable_cpu_diff();
  Dtype *match_diff2 = bottom[2]->mutable_cpu_diff();
  caffe_set(count, Dtype(0), match_diff2);
  for (int i = 0; i < count; ++i) {
    const int mark =
        ((cand_data1[i] + cand_data2[i] - Dtype(2) * match_data[i]) *
         (cand_data1[i] - cand_data2[i])) > Dtype(0);
    cand_diff1[i] = mark * top_diff[i];
    cand_diff2[i] = (Dtype(1) - mark) * top_diff[i];
  }
}

#ifdef CPU_ONLY
STUB_GPU(BestMatchLayer);
#endif

INSTANTIATE_CLASS(BestMatchLayer);

REGISTER_LAYER_CLASS(BestMatch);

}  // namespace caffe
