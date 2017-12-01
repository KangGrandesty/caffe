#include <vector>

#include "caffe/layers/inter_flow_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

    template<typename Dtype>
    void InterFlowLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
        InterFlowParameter param = this->layer_param().inter_flow_param();
        CHECK_GE(bottom.size(), 2U);
        CHECK_GE(top.size(), 2U);
        path_ratio.clear();
        std::copy(param.path().begin(), param.path().end(),
                  std::back_inserter(path_ratio));
        if (path_ratio.size() == 0) {
            for (int i = 0; i < bottom.size() * top.size(); ++i) {
                path_ratio.push_back(0.0);
            }
        } else if (path_ratio.size() == 1) {
            for (int i = 0; i < bottom.size() * top.size() - 1; ++i) {
                path_ratio.push_back(path_ratio[0]);
            }
        } else if (path_ratio.size() == bottom.size()) {
            for (int i = 0; i < top.size() - 1; ++i) {
                for (int j = 0; j < bottom.size(); ++j) {
                    path_ratio.push_back(path_ratio[j]);
                }
            }
        } else if (path_ratio.size() != bottom.size() * top.size()) {
            LOG(FATAL) << "ratio number error " << path_ratio.size() << " "
                       << bottom.size() * top.size();
        }
        CHECK_EQ(path_ratio.size(), bottom.size() * top.size());
    }

    template<typename Dtype>
    void InterFlowLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
        for (int i = 1; i < bottom.size(); ++i) {
            CHECK(bottom[i]->shape() == bottom[0]->shape());
        }
        for (int i = 0; i < top.size(); ++i) {
            top[i]->ReshapeLike(*bottom[0]);
        }
    }

    template<typename Dtype>
    void InterFlowLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
        if (this->phase_ == TRAIN) {
            maps_.clear();
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
                caffe_set(count_, Dtype(0), top[i]->mutable_cpu_data());
            }
            for (int i = 0; i < top.size(); ++i) {
                for (int j = 0; j < maps_[i].size(); ++j) {
                    caffe_axpy(count_, Dtype(1), //Dtype(bottom.size()) / Dtype(maps_[i].size()),
                               bottom[maps_[i][j]]->cpu_data(),
                               top[i]->mutable_cpu_data());
                }
            }
        } else {
            int count_ = bottom[0]->count();
            caffe_set(count_, Dtype(0), top[0]->mutable_cpu_data());
            for (int i = 0; i < bottom.size(); ++i) {
                caffe_axpy(count_, Dtype(1), bottom[i]->cpu_data(),
                           top[0]->mutable_cpu_data());
            }
            for (int i = 1; i < top.size(); ++i) {
                caffe_copy(count_, top[0]->cpu_data(), top[i]->mutable_cpu_data());
            }
        }
    }

    template<typename Dtype>
    void InterFlowLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                             const vector<bool> &propagate_down,
                                             const vector<Blob<Dtype> *> &bottom) {
        int count_ = bottom[0]->count();
        for (int i = 0; i < bottom.size(); ++i) {
            if (propagate_down[i]) {
                caffe_set(count_, Dtype(0), bottom[i]->mutable_cpu_diff());
            }
        }
        for (int i = 0; i < top.size(); ++i) {
            for (int j = 0; j < maps_[i].size(); ++j) {
                if (propagate_down[maps_[i][j]]) {
                    caffe_axpy(count_, Dtype(1), //Dtype(bottom.size()) / Dtype(maps_[i].size()),
                               top[i]->cpu_diff(),
                               bottom[maps_[i][j]]->mutable_cpu_diff());
                }
            }
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(InterFlowLayer);
#endif

    INSTANTIATE_CLASS(InterFlowLayer);

    REGISTER_LAYER_CLASS(InterFlow);

}  // namespace caffe
