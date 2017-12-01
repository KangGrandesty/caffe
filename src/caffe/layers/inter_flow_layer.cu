#include <vector>

#include "caffe/layers/inter_flow_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void InterFlowLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
        if (this->phase_ == TRAIN) {
            maps_.clear();
            path_map.clear();
            for (int i = 0; i < top.size(); ++i) {
                vector<int> maps_i;
                maps_i.clear();
                for (int j = 0; j < bottom.size(); ++j) {
                    if ((path_ratio[i * bottom.size() + j] * UINT_MAX) > caffe_rng_rand()) {
                        maps_i.push_back(j);
                    }
                }
                maps_.push_back(maps_i);
            }
            int count_ = bottom[0]->count();
            for (int i = 0; i < top.size(); ++i) {
                if (maps_[i].size() == 0) {
                    maps_[i].push_back(caffe_rng_rand() % bottom.size());
                }
                caffe_gpu_set(count_, Dtype(0), top[i]->mutable_gpu_data());
            }
            for (int i = 0; i < top.size(); ++i) {
                for (int j = 0; j < maps_[i].size(); ++j) {
                    caffe_gpu_axpy(count_, Dtype(1), //Dtype(bottom.size()) / Dtype(maps_[i].size()),
                                   bottom[maps_[i][j]]->gpu_data(),
                                   top[i]->mutable_gpu_data());
                }
            }
        } else {
            int count_ = bottom[0]->count();
            caffe_gpu_set(count_, Dtype(0), top[0]->mutable_gpu_data());
            for (int i = 0; i < bottom.size(); ++i) {
                caffe_gpu_axpy(count_, Dtype(1), bottom[i]->gpu_data(),
                               top[0]->mutable_gpu_data());
            }
            for (int i = 1; i < top.size(); ++i) {
                caffe_copy(count_, top[0]->gpu_data(), top[i]->mutable_gpu_data());
            }
        }
    }

    template<typename Dtype>
    void InterFlowLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                             const vector<bool> &propagate_down,
                                             const vector<Blob<Dtype> *> &bottom) {
        int count_ = bottom[0]->count();
        for (int i = 0; i < bottom.size(); ++i) {
            if (propagate_down[i]) {
                caffe_gpu_set(count_, Dtype(0), bottom[i]->mutable_gpu_diff());
            }
        }
        for (int i = 0; i < top.size(); ++i) {
            for (int j = 0; j < maps_[i].size(); ++j) {
                if (propagate_down[maps_[i][j]]) {
                    caffe_gpu_axpy(count_, Dtype(1), //Dtype(bottom.size()) / Dtype(maps_[i].size()),
                                   top[i]->gpu_diff(),
                                   bottom[maps_[i][j]]->mutable_gpu_diff());
                }
            }
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(InterFlowLayer);

}  // namespace caffe
