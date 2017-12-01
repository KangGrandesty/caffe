#include <numeric>
#include <vector>

#include "caffe/layers/switch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  switch_map.clear();
  for (int i = 0; i < BLOB_SIZE; ++i) {
    switch_map.push_back(i);
  }
  if (this->phase_ == TRAIN) {
    if (stochastic) {
      for (int i = 0; i < BLOB_SIZE; ++i) {
        unsigned int choice_1 = caffe_rng_rand() % BLOB_SIZE;
        unsigned int choice_2 = caffe_rng_rand() % BLOB_SIZE;
        unsigned int temp = switch_map[choice_1];
        switch_map[choice_1] = switch_map[choice_2];
        switch_map[choice_2] = temp;
      }
    } else {
      vector<Dtype> switch_buff;
      switch_buff.clear();
      for (int i = 0; i < BLOB_SIZE; ++i) {
        for (int j = 0; j < BLOB_SIZE; ++j) {
          switch_buff.push_back(switch_path[j * BLOB_SIZE + i]);
        }
      }
      typename vector<Dtype>::const_iterator iter;
      for (int i = 0; i < BLOB_SIZE; ++i) {
        iter = switch_buff.begin();
        iter += (i * BLOB_SIZE);
        Dtype switcher = std::accumulate(iter, iter + BLOB_SIZE, Dtype(0)) *
                         caffe_rng_rand() / UINT_MAX;
        unsigned int switch_mark = BLOB_SIZE;
        while (switcher > Dtype(0) && switch_mark > 0) {
          switcher -= switch_buff[switch_mark + i * BLOB_SIZE - 1];
          --switch_mark;
        }
        switch_map[i] = switch_mark;
        for (int j = 0; j < BLOB_SIZE; ++j) {
          switch_buff[j * BLOB_SIZE + switch_mark] = Dtype(0);
        }
      }
    }
  }
  int count_ = bottom[0]->count();
  for (int i = 0; i < BLOB_SIZE; ++i) {
    caffe_copy(count_, bottom[switch_map[i]]->gpu_data(),
               top[i]->mutable_gpu_data());
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  int count_ = bottom[0]->count();
  for (int i = 0; i < BLOB_SIZE; ++i) {
    if (propagate_down[i]) {
      caffe_copy(count_, top[i]->gpu_diff(),
                 bottom[switch_map[i]]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SwitchLayer);

}  // namespace caffe
