#include <numeric>
#include <vector>

#include "caffe/layers/switch_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SwitchLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), top.size());
  BLOB_SIZE = bottom.size();
  SwitchParameter param = this->layer_param().switch_param();
  switch_path.clear();
  std::copy(param.switch_path().begin(), param.switch_path().end(),
            std::back_inserter(switch_path));
  if (switch_path.size() == BLOB_SIZE * BLOB_SIZE) {
    stochastic = false;
    // normalize all of the ratio from one src(bottom) to all the dest(top)
    typename vector<Dtype>::iterator iter;
    iter = switch_path.begin();
    for (int i = 0; i < BLOB_SIZE; ++i) {
      Dtype sum = std::accumulate(iter, iter + BLOB_SIZE, Dtype(0));
      CHECK_GT(sum, Dtype(0));
      for (int j = 0; j < BLOB_SIZE; ++j, ++iter) {
        *iter /= sum;
      }
    }
  } else {
    stochastic = true;
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < BLOB_SIZE; ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  for (int i = 0; i < BLOB_SIZE; ++i) {
    top[i]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  // initialize one-to-one match between bottom and top
  switch_map.clear();
  for (int i = 0; i < BLOB_SIZE; ++i) {
    switch_map.push_back(i);
  }
  if (this->phase_ == TRAIN) {
    if (stochastic) {
      // set a stochastic match
      for (int i = 0; i < BLOB_SIZE; ++i) {
        unsigned int choice_1 = caffe_rng_rand() % BLOB_SIZE;
        unsigned int choice_2 = caffe_rng_rand() % BLOB_SIZE;
        unsigned int temp = switch_map[choice_1];
        switch_map[choice_1] = switch_map[choice_2];
        switch_map[choice_2] = temp;
      }
    } else {
      // gather all the ratio of src to one dest
      vector<Dtype> switch_buff;
      switch_buff.clear();
      for (int dest = 0; dest < BLOB_SIZE; ++dest) {
        for (int src = 0; src < BLOB_SIZE; ++src) {
          switch_buff.push_back(switch_path[src * BLOB_SIZE + dest]);
        }
      }

      // decide each pair of src-dest through the ratio
      typename vector<Dtype>::const_iterator iter;
      for (int dest = 0; dest < BLOB_SIZE; ++dest) {
        iter = switch_buff.begin();
        iter += (dest * BLOB_SIZE);
        Dtype switcher = std::accumulate(iter, iter + BLOB_SIZE, Dtype(0)) *
                         caffe_rng_rand() / UINT_MAX;
        unsigned int switch_mark = BLOB_SIZE;
        while (switcher > Dtype(0) && switch_mark > 0) {
          switcher -= switch_buff[switch_mark + dest * BLOB_SIZE - 1];
          --switch_mark;
        }
        // dest has a source switch_mark
        switch_map[dest] = switch_mark;
        // set the No.switch_mark src's ratio 0
        // guarantee the one-to-one match
        for (int src = 0; src < BLOB_SIZE; ++src) {
          switch_buff[src * BLOB_SIZE + switch_mark] = Dtype(0);
        }
      }
    }
  }
  int count_ = bottom[0]->count();
  for (int i = 0; i < BLOB_SIZE; ++i) {
    caffe_copy(count_, bottom[switch_map[i]]->cpu_data(),
               top[i]->mutable_cpu_data());
  }
}

template <typename Dtype>
void SwitchLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  int count_ = bottom[0]->count();
  for (int i = 0; i < BLOB_SIZE; ++i) {
    if (propagate_down[i]) {
      caffe_copy(count_, top[i]->cpu_diff(),
                 bottom[switch_map[i]]->mutable_cpu_diff());
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SwitchLayer);
#endif

INSTANTIATE_CLASS(SwitchLayer);
REGISTER_LAYER_CLASS(Switch);

}  // namespace caffe
